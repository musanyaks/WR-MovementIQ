from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "data" / "reference" / "wr_combine_attempt_features_enriched.parquet"
REBUILT = ROOT / "data" / "features" / "wr_combine_attempt_features_enriched.parquet"

KEYS = [
    "draft_year", "nfl_id", "event_id",
    "drill_type", "drill_name", "attempt"
]
ABS_TOL = 1e-9
REL_TOL = 1e-9

def stop(message):
    raise SystemExit(f"\nAUDIT STATUS: FAIL — {message}")

def main():
    for path in (ORIGINAL, REBUILT):
        if not path.exists():
            stop(f"file not found: {path}")

    original = pd.read_parquet(ORIGINAL)
    rebuilt = pd.read_parquet(REBUILT)

    print("=" * 88)
    print("COMBINE ATTEMPT-LEVEL ENRICHED FEATURE AUDIT")
    print("=" * 88)
    print(f"Original:     {ORIGINAL} ({ORIGINAL.stat().st_size:,} bytes)")
    print(f"Reconstructed:{REBUILT} ({REBUILT.stat().st_size:,} bytes)")
    print(f"Original shape:      {original.shape}")
    print(f"Reconstructed shape: {rebuilt.shape}")

    missing_keys = [c for c in KEYS if c not in original.columns or c not in rebuilt.columns]
    if missing_keys:
        stop(f"required key columns missing: {missing_keys}")

    for name, df in (("original", original), ("reconstructed", rebuilt)):
        dupes = int(df.duplicated(KEYS).sum())
        print(f"{name} duplicate key rows: {dupes:,}")
        if dupes:
            examples = df.loc[df.duplicated(KEYS, keep=False), KEYS].head(10)
            print(examples.to_string(index=False))
            stop(f"{name} contains duplicate six-column keys")

    original_keys = pd.MultiIndex.from_frame(original[KEYS])
    rebuilt_keys = pd.MultiIndex.from_frame(rebuilt[KEYS])
    original_key_set = set(original_keys)
    rebuilt_key_set = set(rebuilt_keys)
    only_original = original_key_set - rebuilt_key_set
    only_rebuilt = rebuilt_key_set - original_key_set

    print(f"\nOriginal unique attempts:      {len(original_key_set):,}")
    print(f"Reconstructed unique attempts:{len(rebuilt_key_set):,}")
    print(f"Original-only attempts:        {len(only_original):,}")
    print(f"Reconstructed-only attempts:   {len(only_rebuilt):,}")

    if only_original:
        print("\nExamples original-only keys:")
        print(pd.DataFrame(list(only_original), columns=KEYS).head(10).to_string(index=False))
    if only_rebuilt:
        print("\nExamples reconstructed-only keys:")
        print(pd.DataFrame(list(only_rebuilt), columns=KEYS).head(10).to_string(index=False))

    original_cols = set(original.columns)
    rebuilt_cols = set(rebuilt.columns)
    original_only_cols = sorted(original_cols - rebuilt_cols)
    rebuilt_only_cols = sorted(rebuilt_cols - original_cols)
    common_cols = [c for c in original.columns if c in rebuilt_cols]

    print("\nColumn audit:")
    print(f"  Original columns:      {len(original_cols)}")
    print(f"  Reconstructed columns: {len(rebuilt_cols)}")
    print(f"  Common columns:        {len(common_cols)}")
    print(f"  Original-only columns: {original_only_cols}")
    print(f"  Reconstructed-only columns: {rebuilt_only_cols}")

    # Align records by their natural composite key before comparing values.
    left = original.set_index(KEYS).sort_index()
    right = rebuilt.set_index(KEYS).sort_index()
    common_index = left.index.intersection(right.index)
    left = left.loc[common_index]
    right = right.loc[common_index]

    numeric_passes = 0
    numeric_failures = []
    categorical_failures = []
    null_failures = []
    numeric_details = []

    compare_cols = [c for c in common_cols if c not in KEYS]
    for col in compare_cols:
        a = left[col]
        b = right[col]
        null_a = a.isna()
        null_b = b.isna()
        null_diff = int((null_a != null_b).sum())
        if null_diff:
            null_failures.append((col, null_diff))

        # Treat columns as numeric if both can be converted for all non-null values.
        an = pd.to_numeric(a, errors="coerce")
        bn = pd.to_numeric(b, errors="coerce")
        nonnull_a = a.notna()
        nonnull_b = b.notna()
        numeric_like = (
            (an.notna() | ~nonnull_a).all()
            and (bn.notna() | ~nonnull_b).all()
            and (pd.api.types.is_numeric_dtype(a.dtype)
                 or pd.api.types.is_numeric_dtype(b.dtype))
        )

        if numeric_like:
            av = an.to_numpy(dtype=float)
            bv = bn.to_numpy(dtype=float)
            both_finite = np.isfinite(av) & np.isfinite(bv)
            equal_nan = np.isnan(av) & np.isnan(bv)
            valid = both_finite | equal_nan
            close = np.zeros(len(av), dtype=bool)
            close[both_finite] = np.isclose(
                av[both_finite], bv[both_finite],
                atol=ABS_TOL, rtol=REL_TOL
            )
            close[equal_nan] = True
            failures = int((~close).sum())
            finite_diff = np.abs(av[both_finite] - bv[both_finite])
            max_abs = float(finite_diff.max()) if finite_diff.size else 0.0
            numeric_details.append((col, failures, max_abs))
            if failures:
                numeric_failures.append((col, failures, max_abs))
            else:
                numeric_passes += 1
        else:
            # Normalize values to strings while preserving missing values.
            aa = a.astype("string")
            bb = b.astype("string")
            mismatch = (aa != bb) & ~(aa.isna() & bb.isna())
            failures = int(mismatch.sum())
            if failures:
                categorical_failures.append((col, failures))

    print("\nValue comparison:")
    print(f"  Numeric columns passed: {numeric_passes}")
    print(f"  Numeric columns failed: {len(numeric_failures)}")
    print(f"  Categorical columns failed: {len(categorical_failures)}")
    print(f"  Columns with null-pattern differences: {len(null_failures)}")

    if numeric_details:
        print("\nNumeric feature diagnostics (column, failed rows, max absolute error):")
        for item in numeric_details:
            print(f"  {item[0]}: {item[1]:,}, {item[2]:.12g}")

    if numeric_failures:
        print("\nNumeric failures:")
        for item in numeric_failures:
            print(f"  {item}")
    if categorical_failures:
        print("\nCategorical failures:")
        for item in categorical_failures:
            print(f"  {item}")
    if null_failures:
        print("\nNull-pattern failures:")
        for item in null_failures:
            print(f"  {item}")

    passed = (
        not only_original
        and not only_rebuilt
        and not original_only_cols
        and not rebuilt_only_cols
        and not numeric_failures
        and not categorical_failures
        and not null_failures
    )

    print("\n" + "-" * 88)
    if passed:
        print("AUDIT STATUS: PASS — keys, columns, null patterns, and feature values match.")
    else:
        print("AUDIT STATUS: FAIL — inspect the differences reported above.")
        raise SystemExit(1)
    print("-" * 88)

if __name__ == "__main__":
    main()
