from pathlib import Path
import pandas as pd


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}. See docs/DATA_SETUP.md."
        )
    return pd.read_csv(path, low_memory=False)


def write_table(frame: pd.DataFrame, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_stem.with_suffix(".csv"), index=False)
    frame.to_parquet(output_stem.with_suffix(".parquet"), index=False)
