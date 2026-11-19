from __future__ import annotations

import argparse
from pathlib import Path
import json

import duckdb
import pandas as pd


KEYS = ["game_id", "play_id", "nfl_id"]
TOLERANCE = 1e-8


def sql_string(value: str) -> str:
    return "'" + value.replace("\\", "/").replace("'", "''") + "'"


def ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def describe_parquet(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    return con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string(str(path.resolve()))})"
    ).df()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare original and locally reproduced WR-MovementIQ Parquet outputs."
    )
    parser.add_argument(
        "--original",
        default="data/reference/wr_game_movement_features_original.parquet",
    )
    parser.add_argument(
        "--local",
        default="data/features/wr_game_movement_features.parquet",
    )
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--output-dir", default="reports/data_audit/reproduction")
    args = parser.parse_args()

    original = Path(args.original).resolve()
    local = Path(args.local).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (original, local):
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='4GB'")

    try:
        original_sql = sql_string(str(original))
        local_sql = sql_string(str(local))
        con.execute(
            f"CREATE VIEW original_data AS SELECT * FROM read_parquet({original_sql})"
        )
        con.execute(
            f"CREATE VIEW local_data AS SELECT * FROM read_parquet({local_sql})"
        )

        original_schema = describe_parquet(con, original)
        local_schema = describe_parquet(con, local)

        original_cols = original_schema["column_name"].tolist()
        local_cols = local_schema["column_name"].tolist()
        common_cols = [c for c in original_cols if c in local_cols]
        original_only = [c for c in original_cols if c not in local_cols]
        local_only = [c for c in local_cols if c not in original_cols]

        missing_keys = [k for k in KEYS if k not in common_cols]
        if missing_keys:
            raise ValueError(
                f"Cannot compare receiver-play keys; missing shared columns: {missing_keys}"
            )

        original_count = con.execute("SELECT COUNT(*) FROM original_data").fetchone()[0]
        local_count = con.execute("SELECT COUNT(*) FROM local_data").fetchone()[0]

        key_select = ", ".join(ident(k) for k in KEYS)
        original_dupes = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {key_select}, COUNT(*) AS n
                FROM original_data GROUP BY {key_select} HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        local_dupes = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {key_select}, COUNT(*) AS n
                FROM local_data GROUP BY {key_select} HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        key_join = " AND ".join(
            f"o.{ident(k)} = l.{ident(k)}" for k in KEYS
        )
        original_only_count = con.execute(
            f"""
            SELECT COUNT(*) FROM original_data o
            ANTI JOIN local_data l ON {key_join}
            """
        ).fetchone()[0]
        local_only_count = con.execute(
            f"""
            SELECT COUNT(*) FROM local_data l
            ANTI JOIN original_data o ON {key_join}
            """
        ).fetchone()[0]
        matched_count = con.execute(
            f"""
            SELECT COUNT(*) FROM original_data o
            INNER JOIN local_data l ON {key_join}
            """
        ).fetchone()[0]

        key_summary = pd.DataFrame([{
            "original_rows": original_count,
            "local_rows": local_count,
            "original_duplicate_key_groups": original_dupes,
            "local_duplicate_key_groups": local_dupes,
            "matched_receiver_plays": matched_count,
            "original_only_receiver_plays": original_only_count,
            "local_only_receiver_plays": local_only_count,
            "keys_match_exactly": (
                original_count == local_count
                and original_dupes == 0
                and local_dupes == 0
                and original_only_count == 0
                and local_only_count == 0
            ),
        }])

        # Persist key-level differences for inspection.
        con.execute(f"""
            COPY (
                SELECT o.{ident('game_id')}, o.{ident('play_id')}, o.{ident('nfl_id')}
                FROM original_data o
                ANTI JOIN local_data l ON {key_join}
            ) TO {sql_string(str(output_dir / 'original_only_keys.csv'))}
            (HEADER, DELIMITER ',')
        """)
        con.execute(f"""
            COPY (
                SELECT l.{ident('game_id')}, l.{ident('play_id')}, l.{ident('nfl_id')}
                FROM local_data l
                ANTI JOIN original_data o ON {key_join}
            ) TO {sql_string(str(output_dir / 'local_only_keys.csv'))}
            (HEADER, DELIMITER ',')
        """)

        original_types = dict(zip(original_schema["column_name"], original_schema["column_type"]))
        local_types = dict(zip(local_schema["column_name"], local_schema["column_type"]))

        numeric_tokens = (
            "INT", "UINT", "FLOAT", "DOUBLE", "DECIMAL", "REAL", "HUGEINT", "BIGINT",
            "SMALLINT", "TINYINT"
        )
        numeric_results = []
        categorical_results = []

        for col in common_cols:
            if col in KEYS:
                continue

            o_type = str(original_types[col]).upper()
            l_type = str(local_types[col]).upper()
            is_numeric = any(t in o_type for t in numeric_tokens) and any(
                t in l_type for t in numeric_tokens
            )

            if is_numeric:
                expr_o = f"TRY_CAST(o.{ident(col)} AS DOUBLE)"
                expr_l = f"TRY_CAST(l.{ident(col)} AS DOUBLE)"
                stats = con.execute(
                    f"""
                    SELECT
                        COUNT(*) AS matched_rows,
                        COUNT(*) FILTER (
                            WHERE ({expr_o} IS NULL) <> ({expr_l} IS NULL)
                        ) AS null_status_mismatches,
                        COUNT(*) FILTER (
                            WHERE {expr_o} IS NOT NULL AND {expr_l} IS NOT NULL
                              AND ABS({expr_o} - {expr_l}) <= {args.tolerance}
                        ) AS rows_within_tolerance,
                        COUNT(*) FILTER (
                            WHERE {expr_o} IS NOT NULL AND {expr_l} IS NOT NULL
                              AND ABS({expr_o} - {expr_l}) > {args.tolerance}
                        ) AS rows_outside_tolerance,
                        MAX(ABS({expr_o} - {expr_l})) FILTER (
                            WHERE {expr_o} IS NOT NULL AND {expr_l} IS NOT NULL
                        ) AS max_absolute_error,
                        AVG(ABS({expr_o} - {expr_l})) FILTER (
                            WHERE {expr_o} IS NOT NULL AND {expr_l} IS NOT NULL
                        ) AS mean_absolute_error,
                        QUANTILE_CONT(ABS({expr_o} - {expr_l}), 0.95) FILTER (
                            WHERE {expr_o} IS NOT NULL AND {expr_l} IS NOT NULL
                        ) AS p95_absolute_error
                    FROM original_data o
                    INNER JOIN local_data l ON {key_join}
                    """
                ).fetchone()

                numeric_results.append({
                    "feature": col,
                    "original_type": original_types[col],
                    "local_type": local_types[col],
                    "matched_rows": stats[0],
                    "null_status_mismatches": stats[1],
                    "rows_within_tolerance": stats[2],
                    "rows_outside_tolerance": stats[3],
                    "max_absolute_error": stats[4],
                    "mean_absolute_error": stats[5],
                    "p95_absolute_error": stats[6],
                    "feature_matches": stats[1] == 0 and stats[3] == 0,
                })
            else:
                stats = con.execute(
                    f"""
                    SELECT
                        COUNT(*) AS matched_rows,
                        COUNT(*) FILTER (
                            WHERE o.{ident(col)} IS DISTINCT FROM l.{ident(col)}
                        ) AS differing_values
                    FROM original_data o
                    INNER JOIN local_data l ON {key_join}
                    """
                ).fetchone()
                categorical_results.append({
                    "column": col,
                    "original_type": original_types[col],
                    "local_type": local_types[col],
                    "matched_rows": stats[0],
                    "differing_values": stats[1],
                    "column_matches": stats[1] == 0,
                })

        numeric_df = pd.DataFrame(numeric_results)
        categorical_df = pd.DataFrame(categorical_results)

        key_summary.to_csv(output_dir / "key_comparison.csv", index=False)
        numeric_df.to_csv(output_dir / "numeric_feature_comparison.csv", index=False)
        categorical_df.to_csv(output_dir / "categorical_feature_comparison.csv", index=False)

        schema_summary = {
            "original_columns": original_cols,
            "local_columns": local_cols,
            "shared_columns": common_cols,
            "original_only_columns": original_only,
            "local_only_columns": local_only,
            "numeric_features_compared": len(numeric_results),
            "categorical_columns_compared": len(categorical_results),
            "tolerance": args.tolerance,
        }
        (output_dir / "schema_comparison.json").write_text(
            json.dumps(schema_summary, indent=2), encoding="utf-8"
        )

        print("=" * 78)
        print("WR-MOVEMENTIQ NUMERICAL REPRODUCIBILITY AUDIT")
        print("=" * 78)
        print(f"Original: {original} ({original.stat().st_size:,} bytes)")
        print(f"Local:    {local} ({local.stat().st_size:,} bytes)")
        print("\nKEY COMPARISON")
        print(key_summary.to_string(index=False))
        print("\nSCHEMA DIFFERENCES")
        print(f"Shared columns:       {len(common_cols)}")
        print(f"Original-only cols:   {original_only}")
        print(f"Local-only cols:      {local_only}")
        print(f"Numeric comparisons:  {len(numeric_results)}")
        print(f"Categorical checks:   {len(categorical_results)}")

        if not numeric_df.empty:
            print("\nNUMERIC FEATURE COMPARISON")
            display_cols = [
                "feature", "rows_outside_tolerance", "null_status_mismatches",
                "max_absolute_error", "mean_absolute_error", "feature_matches"
            ]
            print(numeric_df[display_cols].to_string(index=False))

        if not categorical_df.empty:
            print("\nCATEGORICAL FEATURE COMPARISON")
            print(categorical_df.to_string(index=False))

        keys_match = bool(key_summary.iloc[0]["keys_match_exactly"])
        numeric_match = numeric_df.empty or bool(numeric_df["feature_matches"].all())
        categorical_match = (
            categorical_df.empty or bool(categorical_df["column_matches"].all())
        )

        print("\nFINAL AUDIT STATUS")
        if keys_match and numeric_match and categorical_match:
            print("PASS: All shared keys and compared columns match within tolerance.")
        else:
            print("NOT YET A PASS: Differences require investigation.")
            print(f"  Keys match: {keys_match}")
            print(f"  Numeric features match: {numeric_match}")
            print(f"  Categorical columns match: {categorical_match}")

        print(f"\nDetailed reports saved to: {output_dir}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
