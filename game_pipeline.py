from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def qpath(path: Path) -> str:
    """Return a DuckDB-safe SQL string literal for a filesystem path."""
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce WR-MovementIQ snap-to-throw game movement features."
    )
    parser.add_argument(
        "--config", default="config/project.yaml",
        help="Path to project YAML configuration."
    )
    parser.add_argument(
        "--memory-limit", default="8GB",
        help="DuckDB memory limit (default: 8GB for a 16GB machine)."
    )
    parser.add_argument(
        "--threads", type=int, default=2,
        help="DuckDB worker threads (default: 2 for a dual-core CPU)."
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    config = load_config(project_root / args.config)

    raw_dir = project_root / config["paths"]["raw"]
    feature_dir = project_root / config["paths"]["features"]
    audit_dir = project_root / config["paths"]["audit"]
    processed_dir = project_root / config["paths"]["processed"]
    temp_dir = processed_dir / "duckdb_tmp"

    feature_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    required = config["files"]
    game_files = [raw_dir / name for name in required["game_tracking"]]
    player_play_file = raw_dir / required["player_play"]
    games_file = raw_dir / required["games"]

    missing = [p for p in [*game_files, player_play_file, games_file] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n" +
            "\n".join(f"  - {p}" for p in missing)
        )

    db_path = processed_dir / "wr_game_movement.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET threads = {max(1, args.threads)}")
    con.execute(f"SET temp_directory = '{qpath(temp_dir)}'")
    con.execute("SET preserve_insertion_order = false")

    try:
        print("=" * 78)
        print("WR-MOVEMENTIQ — ORIGINAL SNAP-TO-THROW REPRODUCTION")
        print("=" * 78)
        print(f"DuckDB version: {duckdb.__version__}")
        print(f"Database: {db_path}")
        print(f"Memory limit: {args.memory_limit}; threads: {args.threads}")

        # The source notebook's all_game_tracking view unions the three
        # tracking seasons and retains the complete movement/event schema.
        tracking_sql = []
        for season, file_path in zip((2023, 2024, 2025), game_files):
            tracking_sql.append(
                f"""
                SELECT
                    {season}::INTEGER AS tracking_season,
                    game_id, play_id, nfl_id, time,
                    x, y, s, a, dis, o, dir, event
                FROM read_csv_auto(
                    '{qpath(file_path)}',
                    header = true,
                    sample_size = 200000,
                    nullstr = ['NA', 'NaN', '']
                )
                """
            )
        con.execute(
            "CREATE OR REPLACE VIEW all_game_tracking AS\n" +
            "\nUNION ALL\n".join(tracking_sql)
        )

        con.execute(f"""
            CREATE OR REPLACE VIEW player_play_source AS
            SELECT *
            FROM read_csv_auto(
                '{qpath(player_play_file)}',
                header = true,
                sample_size = 200000,
                nullstr = ['NA', 'NaN', '']
            )
        """)
        con.execute(f"""
            CREATE OR REPLACE VIEW games_source AS
            SELECT *
            FROM read_csv_auto(
                '{qpath(games_file)}',
                header = true,
                sample_size = 100000,
                nullstr = ['NA', 'NaN', '']
            )
        """)

        # Source Notebook 01, Cell 15: exact receiver eligibility definition.
        # TRY_CAST handles VARCHAR numeric fields and common "NA" strings.
        con.execute("""
            CREATE OR REPLACE TEMP VIEW targeted_wr_regular AS
            SELECT DISTINCT
                pp.game_id,
                pp.play_id,
                pp.nfl_id,
                g.season,
                UPPER(TRIM(pp.lined_up_position)) AS position,
                pp.route_ran,
                TRY_CAST(
                    NULLIF(TRIM(CAST(pp.separation_at_pass_forward AS VARCHAR)), 'NA')
                    AS DOUBLE
                ) AS separation_at_pass_forward,
                TRY_CAST(
                    NULLIF(TRIM(CAST(pp.rec_yards AS VARCHAR)), 'NA')
                    AS DOUBLE
                ) AS receiving_yards
            FROM player_play_source pp
            JOIN games_source g USING (game_id)
            WHERE pp.target = TRUE
              AND g.season_type = 'REG'
              AND UPPER(TRIM(pp.lined_up_position)) IN ('WR', 'SLOT_WR')
        """)

        # Source Notebook 01, Cell 15: event timestamps grouped by
        # targeted receiver-play and event label.
        con.execute("""
            CREATE OR REPLACE TEMP VIEW receiver_play_events AS
            SELECT
                t.season,
                t.game_id,
                t.play_id,
                t.nfl_id,
                t.position,
                LOWER(TRIM(CAST(tr.event AS VARCHAR))) AS event_name,
                MIN(tr.time) AS event_time,
                COUNT(*) AS marker_rows
            FROM targeted_wr_regular t
            JOIN all_game_tracking tr
              ON t.game_id = tr.game_id
             AND t.play_id = tr.play_id
             AND t.nfl_id = tr.nfl_id
            WHERE tr.event IS NOT NULL
              AND LOWER(TRIM(CAST(tr.event AS VARCHAR)))
                    NOT IN ('na', 'null', '')
            GROUP BY 1, 2, 3, 4, 5, 6
        """)

        # Source Notebook 01, Cell 17, step 1: one valid event window
        # per eligible receiver-play.
        con.execute("""
            CREATE OR REPLACE TEMP VIEW receiver_play_windows AS
            SELECT
                t.season,
                t.game_id,
                t.play_id,
                t.nfl_id,
                t.position,
                t.route_ran,
                t.separation_at_pass_forward,
                t.receiving_yards,
                MIN(CASE WHEN e.event_name = 'ball_snap'
                    THEN e.event_time END) AS snap_time,
                MIN(CASE WHEN e.event_name = 'pass_forward'
                    THEN e.event_time END) AS pass_forward_time
            FROM targeted_wr_regular t
            LEFT JOIN receiver_play_events e
              ON t.season = e.season
             AND t.game_id = e.game_id
             AND t.play_id = e.play_id
             AND t.nfl_id = e.nfl_id
            GROUP BY ALL
            HAVING snap_time IS NOT NULL
               AND pass_forward_time IS NOT NULL
               AND pass_forward_time > snap_time
        """)

        # Source Notebook 01, Cell 17, step 2: tracking observations
        # inside the inclusive snap-to-pass-forward interval.
        con.execute("""
            CREATE OR REPLACE TEMP VIEW receiver_tracking_windows AS
            SELECT
                w.season,
                w.game_id,
                w.play_id,
                w.nfl_id,
                w.position,
                w.route_ran,
                w.separation_at_pass_forward,
                w.receiving_yards,
                w.snap_time,
                w.pass_forward_time,
                tr.time,
                tr.x,
                tr.y,
                tr.s,
                tr.a,
                tr.dis,
                tr.dir
            FROM receiver_play_windows w
            JOIN all_game_tracking tr
              ON w.game_id = tr.game_id
             AND w.play_id = tr.play_id
             AND w.nfl_id = tr.nfl_id
            WHERE tr.time >= w.snap_time
              AND tr.time <= w.pass_forward_time
        """)

        # Source Notebook 01, Cell 17, step 3: preserve the original
        # LAG-based gap, coordinate-step and circular direction-change logic.
        con.execute("""
            CREATE OR REPLACE TEMP VIEW receiver_tracking_deltas AS
            SELECT
                *,
                date_diff(
                    'microseconds',
                    LAG(time) OVER (
                        PARTITION BY game_id, play_id, nfl_id
                        ORDER BY time
                    ),
                    time
                ) / 1000000.0 AS gap_seconds,

                SQRT(
                    POWER(x - LAG(x) OVER (
                        PARTITION BY game_id, play_id, nfl_id ORDER BY time
                    ), 2)
                    +
                    POWER(y - LAG(y) OVER (
                        PARTITION BY game_id, play_id, nfl_id ORDER BY time
                    ), 2)
                ) AS coordinate_step,

                CASE
                    WHEN dir IS NOT NULL
                     AND LAG(dir) OVER (
                        PARTITION BY game_id, play_id, nfl_id ORDER BY time
                     ) IS NOT NULL
                    THEN ABS(
                        ((dir - LAG(dir) OVER (
                            PARTITION BY game_id, play_id, nfl_id ORDER BY time
                        ) + 540) % 360) - 180
                    )
                END AS direction_change_raw
            FROM receiver_tracking_windows
        """)

        # Source Notebook 01, Cell 17, step 4: exact feature names and
        # aggregation definitions. Keep output in DuckDB until export.
        con.execute("""
            CREATE OR REPLACE TABLE wr_game_movement_features AS
            WITH aggregated AS (
                SELECT
                    season,
                    game_id,
                    play_id,
                    nfl_id,
                    position,
                    route_ran,
                    separation_at_pass_forward,
                    receiving_yards,

                    date_diff(
                        'microseconds', snap_time, pass_forward_time
                    ) / 1000000.0 AS snap_to_throw_seconds,

                    COUNT(*) AS tracking_observations,

                    AVG(s) AS mean_speed_raw,
                    MAX(s) AS peak_speed_raw,
                    quantile_cont(s, 0.95) AS p95_speed_raw,

                    AVG(a) AS mean_acceleration_raw,
                    MAX(a) AS peak_acceleration_raw,
                    MIN(a) AS minimum_acceleration_raw,

                    SUM(dis) AS summed_recorded_displacement_raw,
                    SUM(coordinate_step) AS coordinate_path_length_raw,

                    SUM(direction_change_raw) AS total_direction_change_raw,
                    COUNT(direction_change_raw) AS direction_change_observations,

                    MAX(gap_seconds) AS maximum_sampling_gap_seconds,
                    COUNT(*) FILTER (
                        WHERE gap_seconds > 0.15
                    ) AS gaps_over_0_15_seconds,

                    COUNT(*) FILTER (
                        WHERE s IS NULL OR a IS NULL OR x IS NULL OR y IS NULL
                    ) AS rows_with_missing_movement_values,

                    MIN(x) AS min_x,
                    MAX(x) AS max_x,
                    MIN(y) AS min_y,
                    MAX(y) AS max_y,

                    ARG_MIN(x, time) AS start_x,
                    ARG_MIN(y, time) AS start_y,
                    ARG_MAX(x, time) AS end_x,
                    ARG_MAX(y, time) AS end_y

                FROM receiver_tracking_deltas
                GROUP BY ALL
            ),
            final AS (
                SELECT *,
                    SQRT(
                        POWER(end_x - start_x, 2) +
                        POWER(end_y - start_y, 2)
                    ) AS net_displacement_raw,

                    CASE
                        WHEN coordinate_path_length_raw > 0
                        THEN SQRT(
                            POWER(end_x - start_x, 2) +
                            POWER(end_y - start_y, 2)
                        ) / coordinate_path_length_raw
                    END AS path_efficiency
                FROM aggregated
            )
            SELECT *
            FROM final
        """)

        # Materialize row/key/coverage diagnostics from the generated table.
        duplicate_summary = con.execute("""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT (game_id, play_id, nfl_id)) AS unique_receiver_plays,
                COUNT(*) - COUNT(DISTINCT (game_id, play_id, nfl_id))
                    AS duplicate_key_rows
            FROM wr_game_movement_features
        """).df()

        feature_quality = con.execute("""
            SELECT
                season,
                COUNT(*) AS receiver_plays,
                COUNT(DISTINCT nfl_id) AS unique_receivers,
                MEDIAN(tracking_observations) AS median_observations,
                MEDIAN(snap_to_throw_seconds) AS median_duration,
                MEDIAN(peak_speed_raw) AS median_peak_speed,
                MEDIAN(peak_acceleration_raw) AS median_peak_acceleration,
                MEDIAN(coordinate_path_length_raw) AS median_path_length,
                COUNT(*) FILTER (
                    WHERE gaps_over_0_15_seconds > 0
                ) AS plays_with_large_gaps,
                COUNT(*) FILTER (
                    WHERE rows_with_missing_movement_values > 0
                ) AS plays_with_missing_values
            FROM wr_game_movement_features
            GROUP BY season
            ORDER BY season
        """).df()

        feature_distribution = con.execute("""
            SELECT
                COUNT(*) AS records,
                MIN(snap_to_throw_seconds) AS min_duration,
                MEDIAN(snap_to_throw_seconds) AS median_duration,
                MAX(snap_to_throw_seconds) AS max_duration,
                MIN(peak_speed_raw) AS min_peak_speed,
                QUANTILE_CONT(peak_speed_raw, 0.50) AS median_peak_speed,
                QUANTILE_CONT(peak_speed_raw, 0.95) AS p95_peak_speed,
                QUANTILE_CONT(peak_speed_raw, 0.99) AS p99_peak_speed,
                MAX(peak_speed_raw) AS max_peak_speed,
                MIN(peak_acceleration_raw) AS min_peak_acceleration,
                MEDIAN(peak_acceleration_raw) AS median_peak_acceleration,
                MAX(peak_acceleration_raw) AS max_peak_acceleration,
                MEDIAN(coordinate_path_length_raw) AS median_path_length,
                MAX(coordinate_path_length_raw) AS max_path_length,
                MEDIAN(net_displacement_raw) AS median_displacement,
                MEDIAN(path_efficiency) AS median_path_efficiency,
                COUNT(*) FILTER (
                    WHERE path_efficiency < 0 OR path_efficiency > 1
                ) AS path_efficiency_outside_0_1
            FROM wr_game_movement_features
        """).df()

        # Event/eligibility coverage makes it possible to distinguish
        # receiver plays from plays that survive event-window and tracking joins.
        event_coverage = con.execute("""
            SELECT
                t.season,
                COUNT(*) AS eligible_targeted_receiver_plays,
                COUNT(*) FILTER (
                    WHERE w.snap_time IS NOT NULL
                ) AS plays_with_snap_and_valid_window,
                COUNT(*) FILTER (
                    WHERE f.game_id IS NOT NULL
                ) AS plays_with_tracking_features
            FROM targeted_wr_regular t
            LEFT JOIN receiver_play_windows w
              ON t.season = w.season
             AND t.game_id = w.game_id
             AND t.play_id = w.play_id
             AND t.nfl_id = w.nfl_id
            LEFT JOIN wr_game_movement_features f
              ON t.season = f.season
             AND t.game_id = f.game_id
             AND t.play_id = f.play_id
             AND t.nfl_id = f.nfl_id
            GROUP BY t.season
            ORDER BY t.season
        """).df()

        # Write Parquet directly from DuckDB (no large Pandas materialization).
        parquet_path = feature_dir / "wr_game_movement_features.parquet"
        csv_path = feature_dir / "wr_game_movement_features.csv"
        con.execute(
            f"COPY wr_game_movement_features TO '{qpath(parquet_path)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"COPY wr_game_movement_features TO '{qpath(csv_path)}' "
            "(FORMAT CSV, HEADER, DELIMITER ',')"
        )

        feature_quality.to_csv(
            audit_dir / "game_movement_feature_quality.csv", index=False
        )
        event_coverage.to_csv(
            audit_dir / "game_movement_event_coverage.csv", index=False
        )
        duplicate_summary.to_csv(
            audit_dir / "game_movement_key_validation.csv", index=False
        )
        feature_distribution.to_csv(
            audit_dir / "game_movement_feature_distribution.csv", index=False
        )

        print("\n" + "=" * 78)
        print("PIPELINE RESULTS")
        print("=" * 78)
        print(duplicate_summary.to_string(index=False))
        print("\nEVENT COVERAGE BY SEASON")
        print(event_coverage.to_string(index=False))
        print("\nFEATURE QUALITY BY SEASON")
        print(feature_quality.to_string(index=False))
        print("\nFEATURE DISTRIBUTION / BOUNDARY CHECKS")
        print(feature_distribution.to_string(index=False))
        print("\nOUTPUTS")
        print(f"  Parquet: {parquet_path}")
        print(f"  CSV:     {csv_path}")
        print(f"  Audits:  {audit_dir}")
        print("\nPipeline completed. Compare row counts and feature values")
        print("against the original Colab wr_game_movement_features.parquet.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
