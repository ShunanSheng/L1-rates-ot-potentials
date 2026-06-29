#!/usr/bin/env python
import argparse
import glob
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otexp.io import ensure_dir, save_csv_atomic  # noqa: E402

SPEC_COLS = ["n", "n_eval_source", "n_solve", "base_seed"]


def _read_csvs(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _present_spec_cols(df):
    return [col for col in SPEC_COLS if col in df.columns]


def _critical_values(calibration, alpha):
    rows = []
    if calibration.empty:
        return pd.DataFrame(rows)
    group_cols = [*_present_spec_cols(calibration), "null_name", "statistic"]
    for keys, group in calibration.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        values = np.sort(group["value"].dropna().to_numpy(dtype=float))
        if len(values):
            idx = int(np.ceil((1.0 - alpha) * len(values))) - 1
            critical_value = float(values[np.clip(idx, 0, len(values) - 1)])
        else:
            critical_value = np.nan
        row = {
            "null_name": key_map["null_name"],
            "statistic": key_map["statistic"],
            "alpha": alpha,
            "critical_value": critical_value,
            "calibration_B": int(len(values)),
        }
        row.update({col: key_map[col] for col in _present_spec_cols(calibration)})
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_reject(df, critical_values):
    if df.empty:
        return df
    if critical_values.empty:
        df = df.copy()
        df["critical_value"] = np.nan
        df["reject"] = np.nan
        return df

    merge_cols = [*_present_spec_cols(df), "null_name", "statistic"]
    merged = df.drop(columns=["critical_value", "reject"], errors="ignore").merge(
        critical_values[[*merge_cols, "critical_value", "alpha"]],
        on=merge_cols,
        how="left",
    )
    missing = merged["critical_value"].isna() | merged["value"].isna()
    reject = (merged["value"] > merged["critical_value"]).astype(object)
    reject[missing] = np.nan
    merged["reject"] = reject
    return merged


def _rejection_summary(df, quantity):
    rows = []
    if df.empty:
        return pd.DataFrame(rows)

    spec_cols = _present_spec_cols(df)
    group_cols = [*spec_cols, "null_name", "scenario", "alt_type", "level", "statistic"]
    for keys, group in df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        reject = group["reject"].dropna().astype(bool)
        estimate = float(reject.mean()) if len(reject) else np.nan
        row = {
            "quantity": quantity,
            "null_name": key_map["null_name"],
            "scenario": key_map["scenario"],
            "alt_type": key_map["alt_type"],
            "level": float(key_map["level"]) if pd.notna(key_map["level"]) else np.nan,
            "statistic": key_map["statistic"],
            "estimate": estimate,
            "mc_se": float(np.sqrt(estimate * (1.0 - estimate) / len(reject))) if len(reject) else np.nan,
            "num_replicates": int(len(reject)),
            "alpha": float(group["alpha"].dropna().iloc[0]) if "alpha" in group and group["alpha"].notna().any() else np.nan,
            "critical_value": float(group["critical_value"].dropna().iloc[0]) if group["critical_value"].notna().any() else np.nan,
            "success_rate": float(group["success"].mean()) if "success" in group else np.nan,
            "error_rate": float(group["error_message"].fillna("").astype(bool).mean()) if "error_message" in group else np.nan,
            "mean_value": float(group["value"].mean()),
            "median_value": float(group["value"].median()),
        }
        row.update({col: key_map[col] for col in spec_cols})
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_power_differences(power):
    rows = []
    if power.empty:
        return pd.DataFrame(rows)

    pairs = [("potential", "w2"), ("potential", "mmd"), ("w2", "mmd")]
    spec_cols = _present_spec_cols(power)
    group_cols = [*spec_cols, "null_name", "alt_type", "level"]
    for keys, group in power.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        wide = group.pivot_table(
            index="replicate",
            columns="statistic",
            values="reject",
            aggfunc="first",
        )
        for left, right in pairs:
            if left not in wide or right not in wide:
                continue
            paired = wide[[left, right]].dropna().astype(float)
            if paired.empty:
                estimate = np.nan
                mc_se = np.nan
            else:
                diff = paired[left] - paired[right]
                estimate = float(diff.mean())
                mc_se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) >= 2 else 0.0
            row = {
                "null_name": key_map["null_name"],
                "alt_type": key_map["alt_type"],
                "level": float(key_map["level"]),
                "left_statistic": left,
                "right_statistic": right,
                "estimate": estimate,
                "mc_se": mc_se,
                "num_pairs": int(len(paired)),
            }
            row.update({col: key_map[col] for col in spec_cols})
            rows.append(row)
    return pd.DataFrame(rows)


def _runtime_summary(all_raw):
    if all_raw.empty:
        return pd.DataFrame()
    group_cols = [*_present_spec_cols(all_raw), "null_name", "scenario", "statistic"]
    return (
        all_raw.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            num_rows=("runtime_seconds", "size"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            max_runtime_seconds=("runtime_seconds", "max"),
            success_rate=("success", "mean"),
        )
    )


def aggregate_gof(outdir="results_gof", alpha=0.05):
    outdir = Path(outdir)
    summary_dir = ensure_dir(outdir / "summary")

    calibration = _read_csvs(str(outdir / "raw" / "calibration" / "**" / "chunk=*.csv"))
    size = _read_csvs(str(outdir / "raw" / "size" / "**" / "chunk=*.csv"))
    power = _read_csvs(str(outdir / "raw" / "power" / "**" / "chunk=*.csv"))

    critical_values = _critical_values(calibration, alpha)
    size = _attach_reject(size, critical_values)
    power = _attach_reject(power, critical_values)

    size_summary = _rejection_summary(size, "empirical size")
    power_summary = _rejection_summary(power, "empirical power")
    paired = _paired_power_differences(power)
    all_raw = pd.concat([df for df in (calibration, size, power) if not df.empty], ignore_index=True) if any(
        not df.empty for df in (calibration, size, power)
    ) else pd.DataFrame()
    runtime = _runtime_summary(all_raw)

    outputs = {
        "critical_values.csv": critical_values,
        "size_summary.csv": size_summary,
        "power_summary.csv": power_summary,
        "paired_power_differences.csv": paired,
        "runtime_summary.csv": runtime,
    }
    for name, df in outputs.items():
        save_csv_atomic(df, summary_dir / name)
        print(f"Saved {summary_dir / name}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Aggregate GOF raw chunk files.")
    parser.add_argument("--outdir", type=str, default="results_gof")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    aggregate_gof(args.outdir, args.alpha)


if __name__ == "__main__":
    main()
