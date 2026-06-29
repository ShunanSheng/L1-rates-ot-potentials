from pathlib import Path
import time

import numpy as np
import pandas as pd

from .core import (
    evaluate_phi_hat_and_assignment,
    evaluate_w2_from_assignment,
    solve_weights,
)
from .io import ensure_dir, file_exists_and_nonempty, load_json, save_csv_atomic, save_json
from .mmd import compute_mmd_unbiased, median_bandwidth, precompute_mmd_reference
from .sampling import GOF_NULLS, phi_true, sample_alternative, sample_null


ALT_LEVELS = {
    "location_shift": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    "scale": (1.025, 1.05, 1.075, 1.10, 1.15, 1.20),
    "mixture_contamination": (0.01, 0.02, 0.05, 0.10, 0.15, 0.20),
}
STATISTICS = ("potential", "w2", "mmd")


def get_null_config(null_name):
    if null_name not in GOF_NULLS:
        raise ValueError(f"Unknown null distribution: {null_name}")
    if null_name == "uniform_ball":
        return {"name": null_name, "d": 3, "description": "Unif(B_3(0,1))"}
    if null_name == "truncated_gaussian":
        return {"name": null_name, "d": 3, "radius": 2.0, "description": "N_3(0,I) truncated to ||Z|| <= 2"}
    return {
        "name": null_name,
        "d": 3,
        "radius": 2.0,
        "nu": 5,
        "sigma_diag": [1.0, 1.5**2, 0.7**2],
        "description": "Elliptical t_5 truncated to Y^T Sigma^{-1}Y <= 4",
    }


def default_gof_config(**overrides):
    config = {
        "d": 3,
        "n": 100,
        "B_cal": 500,
        "N_size": 500,
        "N_alt": 500,
        "alpha": 0.05,
        "n_solve": 5000,
        "n_eval_source": 5000,
        "seed": 2026,
        "use_qmc_source": True,
        "max_iter": 500,
        "chunk_size": 1000,
        "gtol": 1e-5,
        "ftol": 1e-10,
        "outdir": "results_gof",
        "overwrite": False,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def _spec_label(config):
    return (
        f"n={int(config['n'])}"
        f"_eval={int(config['n_eval_source'])}"
        f"_solve={int(config['n_solve'])}"
        f"_seed={int(config['seed'])}"
    )


def _reference_dir(outdir, null_name, config):
    return Path(outdir) / "references" / _spec_label(config) / f"null={null_name}"


def _reference_paths(outdir, null_name, config):
    ref_dir = _reference_dir(outdir, null_name, config)
    return ref_dir / "references.npz", ref_dir / "metadata.json"


def _raw_chunk_label(chunk_id):
    return "all" if chunk_id is None else str(int(chunk_id))


def _format_level(level):
    text = f"{float(level):g}"
    return text.replace("-", "m").replace(".", "p")


def _replicate_indices(total, chunk_id=None, num_chunks=None):
    total = int(total)
    if chunk_id is None or num_chunks is None:
        return list(range(total))
    chunk_id = int(chunk_id)
    num_chunks = int(num_chunks)
    if not 0 <= chunk_id < num_chunks:
        raise ValueError(f"chunk_id must be in [0, num_chunks), got {chunk_id}")
    return [idx for idx in range(total) if idx % num_chunks == chunk_id]


def _seed_for(null_name, scenario, replicate, seed, alt_type=None, level=None):
    null_offset = {name: i for i, name in enumerate(GOF_NULLS)}[null_name] * 10_000_000
    scenario_offset = {"calibration": 100_000, "size": 200_000, "power": 300_000}[scenario]
    alt_offset = 0
    if alt_type is not None:
        alt_offset += (list(ALT_LEVELS).index(alt_type) + 1) * 1_000_000
    if level is not None:
        alt_offset += int(round(float(level) * 100000))
    return int(seed + null_offset + scenario_offset + alt_offset + replicate)


def make_gof_references(null_name, config, rng):
    """
    Create null-specific reference objects.

    The same X_eval cloud is used for the potential statistic, W2, and MMD.
    """
    n_solve = int(config["n_solve"])
    n_eval_source = int(config["n_eval_source"])
    seed = int(config["seed"])
    use_qmc_source = bool(config.get("use_qmc_source", True))
    chunk_size = int(config.get("chunk_size", 2048))

    X_solve = sample_null(
        null_name,
        n_solve,
        rng=rng,
        use_qmc=use_qmc_source,
        seed=seed + 101,
    )
    X_eval = sample_null(
        null_name,
        n_eval_source,
        rng=rng,
        use_qmc=use_qmc_source,
        seed=seed + 202,
    )
    sigma0 = median_bandwidth(X_eval, rng, max_points=min(2000, n_eval_source))
    ref_ref_term = precompute_mmd_reference(X_eval, sigma0, chunk_size=chunk_size)

    return {
        "X_solve": X_solve,
        "X_eval": X_eval,
        "sigma0": float(sigma0),
        "ref_ref_term": float(ref_ref_term),
    }


def save_gof_references(null_name, references, config):
    path, meta_path = _reference_paths(config["outdir"], null_name, config)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        X_solve=references["X_solve"],
        X_eval=references["X_eval"],
        sigma0=np.asarray(references["sigma0"]),
        ref_ref_term=np.asarray(references["ref_ref_term"]),
    )
    save_json({
        "null_name": null_name,
        "null_config": get_null_config(null_name),
        "n": int(config["n"]),
        "n_solve": int(config["n_solve"]),
        "n_eval_source": int(config["n_eval_source"]),
        "mmd_reference": "X_eval",
        "seed": int(config["seed"]),
        "use_qmc_source": bool(config.get("use_qmc_source", True)),
        "sigma0": float(references["sigma0"]),
        "ref_ref_term": float(references["ref_ref_term"]),
    }, meta_path)


def load_gof_references(null_name, outdir, config):
    path, meta_path = _reference_paths(outdir, null_name, config)
    if not file_exists_and_nonempty(path):
        raise FileNotFoundError(f"Missing GOF references for {null_name}: {path}")
    data = np.load(path)
    metadata = load_json(meta_path) if meta_path.exists() else {}
    if metadata and metadata.get("mmd_reference") != "X_eval":
        raise ValueError(
            f"GOF references at {path} were created with mmd_reference="
            f"{metadata.get('mmd_reference')!r}. Recreate them with --overwrite."
        )
    references = {
        "X_solve": data["X_solve"],
        "X_eval": data["X_eval"],
        "sigma0": float(data["sigma0"]),
        "ref_ref_term": float(data["ref_ref_term"]),
    }
    if metadata:
        references["metadata"] = metadata
    return references


def prepare_gof_references(null_name, config):
    path, _ = _reference_paths(config["outdir"], null_name, config)
    if file_exists_and_nonempty(path) and not config.get("overwrite", False):
        return load_gof_references(null_name, config["outdir"], config)
    rng = np.random.default_rng(int(config["seed"]) + 17)
    references = make_gof_references(null_name, config, rng)
    save_gof_references(null_name, references, config)
    return references


def compute_potential_and_w2(X, references, solver_config):
    h, res = solve_weights(
        references["X_solve"],
        X,
        max_iter=int(solver_config.get("max_iter", 200)),
        chunk_size=int(solver_config.get("chunk_size", 2048)),
        gtol=float(solver_config.get("gtol", 1e-5)),
        ftol=float(solver_config.get("ftol", 1e-10)),
    )
    phi_hat, assignment = evaluate_phi_hat_and_assignment(
        references["X_eval"],
        X,
        h,
        chunk_size=int(solver_config.get("chunk_size", 2048)),
    )
    diff = phi_hat - phi_true(references["X_eval"])
    centered_l1 = float(np.mean(np.abs(diff - np.median(diff))))
    potential = float(np.sqrt(len(X)) * centered_l1)
    w2 = evaluate_w2_from_assignment(references["X_eval"], X, assignment)
    diagnostics = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "njev": int(res.njev),
        "fun": float(res.fun),
        "grad_inf": float(np.max(np.abs(res.jac))),
    }
    return {"potential": potential, "w2": w2, "centered_l1": centered_l1, **diagnostics}


def compute_gof_statistics(X, null_name, references, solver_config):
    """
    Compute potential, W2, and MMD statistics for one dataset.
    """
    pot_w2 = compute_potential_and_w2(X, references, solver_config)
    mmd = compute_mmd_unbiased(
        X,
        references["X_eval"],
        references["sigma0"],
        references["ref_ref_term"],
        chunk_size=int(solver_config.get("chunk_size", 2048)),
    )
    return {
        "potential": pot_w2["potential"],
        "w2": pot_w2["w2"],
        "mmd": float(mmd),
        "diagnostics": {key: pot_w2[key] for key in (
            "success",
            "status",
            "message",
            "nit",
            "nfev",
            "njev",
            "fun",
            "grad_inf",
        )},
    }


def _error_rows(null_name, scenario, alt_type, level, replicate, seed, start_time, exc, critical_values=None):
    rows = []
    critical_values = critical_values or {}
    for statistic in STATISTICS:
        critical_value = critical_values.get(statistic, np.nan)
        rows.append({
            "null_name": null_name,
            "scenario": scenario,
            "alt_type": alt_type,
            "level": level,
            "replicate": replicate,
            "seed": seed,
            "statistic": statistic,
            "value": np.nan,
            "critical_value": critical_value,
            "reject": np.nan,
            "runtime_seconds": time.time() - start_time,
            "success": False,
            "status": -999,
            "message": "exception",
            "nit": np.nan,
            "nfev": np.nan,
            "njev": np.nan,
            "fun": np.nan,
            "grad_inf": np.nan,
            "error_message": repr(exc),
        })
    return rows


def _stat_rows(null_name, scenario, alt_type, level, replicate, seed, start_time, stats, critical_values=None):
    rows = []
    critical_values = critical_values or {}
    diagnostics = stats["diagnostics"]
    for statistic in STATISTICS:
        critical_value = critical_values.get(statistic, np.nan)
        value = float(stats[statistic])
        rows.append({
            "null_name": null_name,
            "scenario": scenario,
            "alt_type": alt_type,
            "level": level,
            "replicate": replicate,
            "seed": seed,
            "statistic": statistic,
            "value": value,
            "critical_value": critical_value,
            "reject": bool(value > critical_value) if np.isfinite(critical_value) else np.nan,
            "runtime_seconds": time.time() - start_time,
            "success": bool(diagnostics["success"]) if statistic in ("potential", "w2") else True,
            "status": int(diagnostics["status"]) if statistic in ("potential", "w2") else 0,
            "message": str(diagnostics["message"]) if statistic in ("potential", "w2") else "",
            "nit": int(diagnostics["nit"]) if statistic in ("potential", "w2") else 0,
            "nfev": int(diagnostics["nfev"]) if statistic in ("potential", "w2") else 0,
            "njev": int(diagnostics["njev"]) if statistic in ("potential", "w2") else 0,
            "fun": float(diagnostics["fun"]) if statistic in ("potential", "w2") else np.nan,
            "grad_inf": float(diagnostics["grad_inf"]) if statistic in ("potential", "w2") else np.nan,
            "error_message": "",
        })
    return rows


def _add_config_columns(df, config):
    df = df.copy()
    df["n"] = int(config["n"])
    df["n_eval_source"] = int(config["n_eval_source"])
    df["n_solve"] = int(config["n_solve"])
    df["base_seed"] = int(config["seed"])
    return df


def _calibration_path(outdir, null_name, config, chunk_id):
    return (
        Path(outdir)
        / "raw"
        / "calibration"
        / _spec_label(config)
        / f"null={null_name}"
        / f"chunk={_raw_chunk_label(chunk_id)}.csv"
    )


def _size_path(outdir, null_name, config, chunk_id):
    return (
        Path(outdir)
        / "raw"
        / "size"
        / _spec_label(config)
        / f"null={null_name}"
        / f"chunk={_raw_chunk_label(chunk_id)}.csv"
    )


def _power_path(outdir, null_name, alt_type, level, config, chunk_id):
    return (
        Path(outdir)
        / "raw"
        / "power"
        / _spec_label(config)
        / f"null={null_name}"
        / f"alt={alt_type}"
        / f"level={_format_level(level)}"
        / f"chunk={_raw_chunk_label(chunk_id)}.csv"
    )


def _write_or_skip(df, path, overwrite=False):
    path = Path(path)
    if file_exists_and_nonempty(path) and not overwrite:
        return pd.read_csv(path)
    ensure_dir(path.parent)
    save_csv_atomic(df, path)
    return df


def _load_critical_values(outdir, null_name, alpha, config):
    raw_dir = Path(outdir) / "raw" / "calibration" / _spec_label(config) / f"null={null_name}"
    files = sorted(raw_dir.glob("chunk=*.csv"))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    values = {}
    for statistic, group in df.groupby("statistic"):
        stat_values = np.sort(group["value"].dropna().to_numpy(dtype=float))
        if len(stat_values):
            idx = int(np.ceil((1.0 - alpha) * len(stat_values))) - 1
            values[statistic] = float(stat_values[np.clip(idx, 0, len(stat_values) - 1)])
    return values


def run_gof_calibration(null_name, config, chunk_id=None, num_chunks=None):
    references = prepare_gof_references(null_name, config)
    indices = _replicate_indices(config["B_cal"], chunk_id, num_chunks)
    rows = []
    for replicate in indices:
        seed = _seed_for(null_name, "calibration", replicate, config["seed"])
        rng = np.random.default_rng(seed)
        start_time = time.time()
        try:
            X = sample_null(null_name, config["n"], rng=rng, use_qmc=False)
            stats = compute_gof_statistics(X, null_name, references, config)
            rows.extend(_stat_rows(null_name, "calibration", "null", 0.0, replicate, seed, start_time, stats))
        except Exception as exc:
            rows.extend(_error_rows(null_name, "calibration", "null", 0.0, replicate, seed, start_time, exc))
    df = _add_config_columns(pd.DataFrame(rows), config)
    return _write_or_skip(
        df,
        _calibration_path(config["outdir"], null_name, config, chunk_id),
        config.get("overwrite", False),
    )


def run_gof_size(null_name, config, chunk_id=None, num_chunks=None):
    references = prepare_gof_references(null_name, config)
    critical_values = _load_critical_values(config["outdir"], null_name, config["alpha"], config)
    indices = _replicate_indices(config["N_size"], chunk_id, num_chunks)
    rows = []
    for replicate in indices:
        seed = _seed_for(null_name, "size", replicate, config["seed"])
        rng = np.random.default_rng(seed)
        start_time = time.time()
        try:
            X = sample_null(null_name, config["n"], rng=rng, use_qmc=False)
            stats = compute_gof_statistics(X, null_name, references, config)
            rows.extend(_stat_rows(null_name, "size", "null", 0.0, replicate, seed, start_time, stats, critical_values))
        except Exception as exc:
            rows.extend(_error_rows(null_name, "size", "null", 0.0, replicate, seed, start_time, exc, critical_values))
    df = _add_config_columns(pd.DataFrame(rows), config)
    return _write_or_skip(
        df,
        _size_path(config["outdir"], null_name, config, chunk_id),
        config.get("overwrite", False),
    )


def run_gof_power(null_name, alt_type, level, config, chunk_id=None, num_chunks=None):
    if alt_type not in ALT_LEVELS:
        raise ValueError(f"Unknown alternative type: {alt_type}")
    references = prepare_gof_references(null_name, config)
    critical_values = _load_critical_values(config["outdir"], null_name, config["alpha"], config)
    indices = _replicate_indices(config["N_alt"], chunk_id, num_chunks)
    rows = []
    for replicate in indices:
        seed = _seed_for(null_name, "power", replicate, config["seed"], alt_type=alt_type, level=level)
        rng = np.random.default_rng(seed)
        start_time = time.time()
        try:
            X = sample_alternative(null_name, alt_type, level, config["n"], rng=rng)
            stats = compute_gof_statistics(X, null_name, references, config)
            rows.extend(_stat_rows(null_name, "power", alt_type, float(level), replicate, seed, start_time, stats, critical_values))
        except Exception as exc:
            rows.extend(_error_rows(null_name, "power", alt_type, float(level), replicate, seed, start_time, exc, critical_values))
    df = _add_config_columns(pd.DataFrame(rows), config)
    return _write_or_skip(
        df,
        _power_path(config["outdir"], null_name, alt_type, level, config, chunk_id),
        config.get("overwrite", False),
    )


def run_gof_all(null_name, config):
    prepare_gof_references(null_name, config)
    outputs = [run_gof_calibration(null_name, config)]
    outputs.append(run_gof_size(null_name, config))
    for alt_type, levels in ALT_LEVELS.items():
        for level in levels:
            outputs.append(run_gof_power(null_name, alt_type, level, config))
    return outputs
