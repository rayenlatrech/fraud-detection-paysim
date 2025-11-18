from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, **kwargs)
