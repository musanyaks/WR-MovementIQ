#!/usr/bin/env python
"""
WR-MovementIQ — audit_player_season.py

Compare the original player-season reference with the locally reconstructed
Notebook 05 Cell 29 dataset.

Run from the project root:
    python audit_player_season.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "data" / "reference" / "wr_game_player_season_profiles.parquet"
RECONSTRUCTED = ROOT / "data" / "features" / "wr_game_player_season_profiles.parquet"

KEYS = ["draft_year", "season", "nfl_id"]
ABS_TOL = 1e-8
REL_TOL = 1e-7


def main():
    print("=" * 80)
    print("WR-MovementIQ — GAME PLAYER-SEASON REPRODUCTION AUDIT")
    print("=" * 80)

    for path, label in [(REFERENCE, "ORIGINAL REFERENCE"),
                        (RECONSTRUCTED, "RECONSTRUCTED DATASET")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        print(f"{label}: {path.relative_to(ROOT)}")
        print(f"  Size: {path.stat().st_size:,} bytes")

    ref = pd.read_parquet(REFERENCE)
    new = pd.read_parquet(RECONSTRUCTED)

    print("\nDATASET SHAPES")
    print(f"Original:       {ref.shape}")
    print(f"Reconstructed:  {new.shape}")

    missing_keys = [k for k in KEYS if k not in ref.columns or k not in new.columns]
    if missing_keys:
        raise ValueError(f"Missing required key columns: {missing_keys}")

    ref_dup = int(ref.duplicated(KEYS).sum())
    new_dup = int(new.duplicated(KEYS).sum())
    print("\nKEY AUDIT")
    print(f"Original duplicate keys:      {ref_dup}")
    print(f"Reconstructed duplicate keys: {new_dup}")

    key_audit = ref[KEYS].merge(
        new[KEYS], on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    both = int((key_audit["_merge"] == "both").sum())
    ref_only = int((key_audit["_merge"] == "left_only").sum())
    new_only = int((key_audit["_merge"] == "right_only").sum())
    print(f"Matching player-season keys:   {both}")
    print(f"Original-only keys:            {ref_only}")
    print(f"Reconstructed-only keys:       {new_only}")

    ref_cols = set(ref.columns)
    new_cols = set(new.columns)
    common = sorted((ref_cols & new_cols) - set(KEYS))
    ref_only_cols = sorted(ref_cols - new_cols)
    new_only_cols = sorted(new_cols - ref_cols)

    print("\nCOLUMN AUDIT")
    print(f"Common non-key columns: {len(common)}")
    print(f"Original-only columns: {ref_only_cols}")
    print(f"Reconstructed-only columns: {new_only_cols}")

    joined = ref.merge(
        new, on=KEYS, how="inner", suffixes=("_original", "_rebuilt"),
        validate="one_to_one"
    )

    numeric = []
    categorical = []

    for col in common:
        left = joined[f"{col}_original"]
        right = joined[f"{col}_rebuilt"]

        if (pd.api.types.is_numeric_dtype(left)
                and pd.api.types.is_numeric_dtype(right)):
            a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)

            both_nan = np.isnan(a) & np.isnan(b)
            one_nan = np.isnan(a) ^ np.isnan(b)
            finite = np.isfinite(a) & np.isfinite(b)

            equal = both_nan.copy()
            equal[finite] = np.isclose(
                a[finite], b[finite], atol=ABS_TOL, rtol=REL_TOL
            )
            failures = int((~equal).sum())
            max_abs = (
                float(np.max(np.abs(a[finite] - b[finite])))
                if finite.any() else float("nan")
            )
            numeric.append((col, failures, int(one_nan.sum()), max_abs))
        else:
            a = left.astype("string").fillna("<NA>")
            b = right.astype("string").fillna("<NA>")
            categorical.append((col, int((a != b).sum())))

    print("\nNUMERIC FEATURE COMPARISON")
    if numeric:
        print(f"{'Feature':52} {'Failures':>9} {'Null diff':>10} {'Max abs diff':>15}")
        for col, failures, null_diff, max_abs in numeric:
            print(f"{col:52} {failures:9,d} {null_diff:10,d} {max_abs:15.8g}")
    else:
        print("No common numeric features.")

    print("\nNON-NUMERIC FEATURE COMPARISON")
    if categorical:
        for col, mismatches in categorical:
            print(f"{col:52} mismatches={mismatches:,}")
    else:
        print("No common non-numeric features.")

    keys_pass = (
        ref_dup == 0 and new_dup == 0 and ref_only == 0 and new_only == 0
    )
    values_pass = (
        all(x[1] == 0 for x in numeric)
        and all(x[1] == 0 for x in categorical)
    )
    columns_pass = not ref_only_cols and not new_only_cols

    print("\n" + "-" * 80)
    if keys_pass and values_pass and columns_pass:
        print("AUDIT STATUS: PASS — keys, columns, and feature values match.")
    else:
        print("AUDIT STATUS: NOT A FULL PASS")
        print(f"  Keys:    {'PASS' if keys_pass else 'FAIL'}")
        print(f"  Values:  {'PASS' if values_pass else 'FAIL'}")
        print(f"  Columns: {'PASS' if columns_pass else 'DIFFER'}")
    print("-" * 80)


if __name__ == "__main__":
    main()
