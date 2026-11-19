#!/usr/bin/env python
"""
WR-MovementIQ — audit.py

Compares the reconstructed Combine player profiles against original reference
Parquet files. Also checks whether duplicate-looking reference files are
byte-identical.

Run from the project root:
    python audit.py

Expected layout:
    data/reference/wr_combine_player_profiles.parquet
    data/reference/wr_combine_player_profiles (1).parquet
    data/features/wr_combine_player_profiles.parquet
"""

from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "data" / "reference"
FEATURES_DIR = ROOT / "data" / "features"

REFERENCE_FILES = [
    REFERENCE_DIR / "wr_combine_player_profiles.parquet",
    REFERENCE_DIR / "wr_combine_player_profiles (1).parquet",
]
RECONSTRUCTED_FILE = FEATURES_DIR / "wr_combine_player_profiles.parquet"

KEYS = ["draft_year", "nfl_id"]
ABS_TOL = 1e-8
REL_TOL = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def check_keys(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing key columns: {missing}")
    duplicates = int(df.duplicated(KEYS).sum())
    print(f"{label} duplicate key rows: {duplicates:,}")


def compare_frames(reference: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print("REFERENCE VS RECONSTRUCTED COMBINE PLAYER PROFILE")
    print("=" * 76)
    print(f"Reference rows/columns:     {reference.shape}")
    print(f"Reconstructed rows/columns: {rebuilt.shape}")

    check_keys(reference, "Reference")
    check_keys(rebuilt, "Reconstructed")

    ref_keys = reference[KEYS].drop_duplicates()
    new_keys = rebuilt[KEYS].drop_duplicates()
    key_audit = ref_keys.merge(
        new_keys, on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    print(f"Keys in both:               {(key_audit['_merge'] == 'both').sum():,}")
    print(f"Reference only:             {(key_audit['_merge'] == 'left_only').sum():,}")
    print(f"Reconstructed only:         {(key_audit['_merge'] == 'right_only').sum():,}")

    common_columns = sorted((set(reference.columns) & set(rebuilt.columns)) - set(KEYS))
    ref_only = sorted(set(reference.columns) - set(rebuilt.columns))
    rebuilt_only = sorted(set(rebuilt.columns) - set(reference.columns))
    print(f"\nCommon non-key columns:     {len(common_columns)}")
    print(f"Reference-only columns:     {ref_only}")
    print(f"Reconstructed-only columns: {rebuilt_only}")

    merged = reference.merge(
        rebuilt,
        on=KEYS,
        how="inner",
        suffixes=("_ref", "_new"),
        validate="one_to_one",
    )

    numeric_results = []
    categorical_results = []

    for col in common_columns:
        left = merged[f"{col}_ref"]
        right = merged[f"{col}_new"]

        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)

            both_nan = np.isnan(a) & np.isnan(b)
            one_nan = np.isnan(a) ^ np.isnan(b)
            finite = np.isfinite(a) & np.isfinite(b)

            equal = np.zeros(len(a), dtype=bool)
            equal[both_nan] = True
            equal[finite] = np.isclose(
                a[finite], b[finite], atol=ABS_TOL, rtol=REL_TOL
            )
            failed = ~(equal)
            max_abs = (
                float(np.max(np.abs(a[finite] - b[finite])))
                if finite.any()
                else float("nan")
            )
            numeric_results.append(
                (col, int(failed.sum()), int(one_nan.sum()), max_abs)
            )
        else:
            a = left.astype("string").fillna("<NA>")
            b = right.astype("string").fillna("<NA>")
            mismatches = int((a != b).sum())
            categorical_results.append((col, mismatches))

    print("\nNUMERIC FEATURE COMPARISON")
    if numeric_results:
        print(f"{'Feature':48} {'Failures':>9} {'Null diff':>10} {'Max abs diff':>15}")
        for col, failures, null_diff, max_abs in numeric_results:
            print(f"{col:48} {failures:9,d} {null_diff:10,d} {max_abs:15.8g}")
    else:
        print("No common numeric features.")

    print("\nNON-NUMERIC FEATURE COMPARISON")
    if categorical_results:
        for col, mismatches in categorical_results:
            print(f"{col:48} mismatches={mismatches:,}")
    else:
        print("No common non-numeric features.")

    key_pass = (
        (key_audit["_merge"] == "left_only").sum() == 0
        and (key_audit["_merge"] == "right_only").sum() == 0
        and reference.duplicated(KEYS).sum() == 0
        and rebuilt.duplicated(KEYS).sum() == 0
    )
    values_pass = (
        all(item[1] == 0 for item in numeric_results)
        and all(item[1] == 0 for item in categorical_results)
    )
    columns_pass = not ref_only and not rebuilt_only

    print("\n" + "-" * 76)
    if key_pass and values_pass and columns_pass:
        print("AUDIT STATUS: PASS — keys, columns, and compared values match.")
    else:
        print("AUDIT STATUS: NOT A FULL PASS")
        print(f"  Key reconciliation: {'PASS' if key_pass else 'FAIL'}")
        print(f"  Feature values:     {'PASS' if values_pass else 'FAIL'}")
        print(f"  Column sets:        {'PASS' if columns_pass else 'DIFFER'}")
    print("-" * 76)


def main() -> int:
    print("=" * 76)
    print("WR-MovementIQ — COMBINE PLAYER PROFILE REPRODUCTION AUDIT")
    print("=" * 76)

    existing_refs = [p for p in REFERENCE_FILES if p.exists()]
    if not existing_refs:
        print("ERROR: No original reference Parquet file found in data/reference/.")
        return 1

    for path in existing_refs:
        print(f"\nREFERENCE: {path.relative_to(ROOT)}")
        print(f"Size:      {path.stat().st_size:,} bytes")
        print(f"SHA-256:   {sha256(path)}")

    if len(existing_refs) == 2:
        same_hash = sha256(existing_refs[0]) == sha256(existing_refs[1])
        print(f"\nDuplicate reference files byte-identical: {same_hash}")

    if not RECONSTRUCTED_FILE.exists():
        print(f"\nERROR: Reconstructed file not found: {RECONSTRUCTED_FILE}")
        return 1

    # Prefer the canonical filename; otherwise use the first available reference.
    reference_path = (
        REFERENCE_FILES[0] if REFERENCE_FILES[0].exists() else existing_refs[0]
    )

    reference = load_parquet(reference_path)
    reconstructed = load_parquet(RECONSTRUCTED_FILE)

    print(f"\nUsing reference:      {reference_path.name}")
    print(f"Using reconstruction: {RECONSTRUCTED_FILE.name}")

    compare_frames(reference, reconstructed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nAUDIT ERROR: {type(exc).__name__}: {exc}")
        raise
