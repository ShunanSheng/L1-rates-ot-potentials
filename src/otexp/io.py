from pathlib import Path
import json
import tempfile
import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv_atomic(df: pd.DataFrame, path):
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp, index=False)
    tmp_path.replace(path)


def save_json(obj: dict, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def file_exists_and_nonempty(path):
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0
