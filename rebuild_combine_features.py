from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATTEMPT_KEYS = [
    "draft_year", "nfl_id", "drill_type",
    "drill_name", "attempt", "event_id"
]

PRIMARY_DRILLS = [
    "GAUNTLET_DRILL",
    "DAGGER_ROUTE_LEFT",
    "SLANT_ROUTE_LEFT",
    "CURL_ROUTE_RIGHT",
    "SLOT_STRIKE_ROUTE_LEFT",
    "OVER_SHOULDER_ADJUST",
    "RED_ZONE_FADE_RIGHT",
    "COMEBACK_ROUTE_RIGHT",
    "SLOT_SAIL_ROUTE_RIGHT",
    "GO_ROUTE_RIGHT",
    "POST_CORNER_ROUTE_RIGHT",
]


def circular_difference(a: pd.Series, b: pd.Series) -> pd.Series:
    """Absolute smallest angular difference in degrees."""
    return ((a - b + 180.0) % 360.0 - 180.0).abs()


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def save_table(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(stem.with_suffix(".parquet"), index=False)
    df.to_csv(stem.with_suffix(".csv"), index=False)


def build_attempt_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Notebook 04 Cell 31 attempt-level features."""
    required = ATTEMPT_KEYS + [
        "entity_type", "time", "x", "y", "s", "a", "dis", "dir"
    ]
    require_columns(raw, required, "combine_tracking")

    df = raw.copy()
    for col in ["x", "y", "s", "a", "dis", "dir"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.loc[
        df["entity_type"].astype(str).str.upper().eq("PLAYER")
    ].dropna(subset=ATTEMPT_KEYS + ["time"])

    df = df.sort_values(ATTEMPT_KEYS + ["time"], kind="mergesort").copy()
    grouped = df.groupby(ATTEMPT_KEYS, sort=False, dropna=False)

    df["prev_time"] = grouped["time"].shift(1)
    df["prev_x"] = grouped["x"].shift(1)
    df["prev_y"] = grouped["y"].shift(1)
    df["prev_dir"] = grouped["dir"].shift(1)

    df["dt"] = (df["time"] - df["prev_time"]).dt.total_seconds()

    dx = df["x"] - df["prev_x"]
    dy = df["y"] - df["prev_y"]
    df["step_distance"] = np.sqrt(dx.pow(2) + dy.pow(2))

    df["recorded_direction_change"] = circular_difference(
        df["dir"], df["prev_dir"]
    )

    grouped = df.groupby(ATTEMPT_KEYS, sort=False, dropna=False)

    rows = []
    for key, g in grouped:
        g = g.sort_values("time", kind="mergesort")
        first = g.iloc[0]
        last = g.iloc[-1]

        duration = (last["time"] - first["time"]).total_seconds()
        net = np.nan
        if pd.notna(first["x"]) and pd.notna(first["y"]) and pd.notna(last["x"]) and pd.notna(last["y"]):
            net = float(np.hypot(last["x"] - first["x"], last["y"] - first["y"]))

        path_length = g["step_distance"].sum(min_count=1)
        record = dict(zip(ATTEMPT_KEYS, key))
        record.update({
            "observations": int(len(g)),
            "start_time": first["time"],
            "end_time": last["time"],
            "duration_seconds": duration,
            "start_x": first["x"],
            "start_y": first["y"],
            "end_x": last["x"],
            "end_y": last["y"],
            "mean_speed": g["s"].mean(),
            "peak_speed": g["s"].max(),
            "p95_speed": g["s"].quantile(0.95),
            "speed_std": g["s"].std(ddof=0),
            "mean_acceleration": g["a"].mean(),
            "peak_acceleration": g["a"].max(),
            "minimum_acceleration": g["a"].min(),
            "acceleration_std": g["a"].std(ddof=0),
            "coordinate_path_length": path_length,
            "summed_recorded_displacement": g["dis"].sum(min_count=1),
            "total_direction_change": g["recorded_direction_change"].sum(min_count=1),
            "median_sampling_interval": g["dt"].median(),
            "maximum_sampling_interval": g["dt"].max(),
            "large_sampling_gaps": int((g["dt"] > 0.15).sum()),
            "net_displacement": net,
        })
        record["path_efficiency"] = (
            net / path_length
            if pd.notna(net) and pd.notna(path_length) and path_length > 0
            else np.nan
        )
        rows.append(record)

    result = pd.DataFrame(rows)
    if result.duplicated(ATTEMPT_KEYS).any():
        raise AssertionError("Attempt feature output has duplicate six-part keys.")
    return result


def build_turning_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Notebook 04 Cells 35–37 at speed thresholds 1.0 and 2.0."""
    required = ATTEMPT_KEYS + [
        "entity_type", "time", "x", "y", "s", "dir"
    ]
    require_columns(raw, required, "combine_tracking")

    df = raw.copy()
    for col in ["x", "y", "s", "dir"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.loc[
        df["entity_type"].astype(str).str.upper().eq("PLAYER")
    ].dropna(subset=ATTEMPT_KEYS + ["time"])

    df = df.sort_values(ATTEMPT_KEYS + ["time"], kind="mergesort").copy()
    grouped = df.groupby(ATTEMPT_KEYS, sort=False, dropna=False)

    df["dx"] = grouped["x"].diff()
    df["dy"] = grouped["y"].diff()
    df["step_distance"] = np.sqrt(df["dx"].pow(2) + df["dy"].pow(2))
    df["coordinate_heading"] = (
        np.degrees(np.arctan2(df["dx"], df["dy"])) % 360.0
    )
    df["previous_heading"] = grouped["coordinate_heading"].shift(1)
    df["previous_dir"] = grouped["dir"].shift(1)
    df["previous_speed"] = grouped["s"].shift(1)
    df["previous_time"] = grouped["time"].shift(1)
    df["pair_dt"] = (
        df["time"] - df["previous_time"]
    ).dt.total_seconds()

    df["coordinate_turn_pair"] = circular_difference(
        df["coordinate_heading"], df["previous_heading"]
    )
    df["recorded_turn_pair"] = circular_difference(
        df["dir"], df["previous_dir"]
    )

    valid_pairs = (
        df["pair_dt"].between(0.05, 0.15)
        & df["s"].notna()
        & df["previous_speed"].notna()
        & df["coordinate_turn_pair"].notna()
        & df["recorded_turn_pair"].notna()
    )

    attempt_info = (
        df.groupby(ATTEMPT_KEYS, sort=False, dropna=False)
        .agg(
            attempt_duration_seconds=(
                "time", lambda x: (x.max() - x.min()).total_seconds()
            ),
            total_observations=("time", "size"),
        )
        .reset_index()
    )

    outputs = []
    for threshold in [1.0, 2.0]:
        selected = df.loc[
            valid_pairs
            & (df["s"] >= threshold)
            & (df["previous_speed"] >= threshold)
        ].copy()

        features = (
            selected.groupby(ATTEMPT_KEYS, sort=False, dropna=False)
            .agg(
                valid_turn_pairs=("coordinate_turn_pair", "size"),
                coordinate_total_turn=("coordinate_turn_pair", "sum"),
                coordinate_mean_turn=("coordinate_turn_pair", "mean"),
                coordinate_median_turn=("coordinate_turn_pair", "median"),
                coordinate_p95_turn=(
                    "coordinate_turn_pair",
                    lambda x: x.quantile(0.95),
                ),
                coordinate_max_turn=("coordinate_turn_pair", "max"),
                recorded_total_turn=("recorded_turn_pair", "sum"),
                recorded_mean_turn=("recorded_turn_pair", "mean"),
            )
            .reset_index()
        )

        features["speed_threshold"] = threshold
        features = features.merge(
            attempt_info,
            on=ATTEMPT_KEYS,
            how="left",
            validate="one_to_one",
        )
        features["turning_rate_degrees_per_second"] = (
            features["coordinate_total_turn"]
            / features["attempt_duration_seconds"]
        )
        denominator = features["total_observations"] - 1
        features["valid_pair_coverage"] = (
            features["valid_turn_pairs"] / denominator
        )
        outputs.append(features)

    result = pd.concat(outputs, ignore_index=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild WR-MovementIQ Combine attempt and turning features."
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Directory containing combine_results.csv and combine_tracking.csv",
    )
    parser.add_argument(
        "--feature-dir", default="data/features",
        help="Output feature directory",
    )
    parser.add_argument(
        "--audit-dir", default="reports/data_audit",
        help="Output audit directory",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    feature_dir = Path(args.feature_dir)
    audit_dir = Path(args.audit_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    combine_path = raw_dir / "combine_results.csv"
    tracking_path = raw_dir / "combine_tracking.csv"
    if not combine_path.exists() or not tracking_path.exists():
        raise FileNotFoundError(
            "Expected data/raw/combine_results.csv and "
            "data/raw/combine_tracking.csv. Check --raw-dir."
        )

    print("Reading Combine source files...")
    combine = pd.read_csv(combine_path, low_memory=False)
    raw = pd.read_csv(tracking_path, low_memory=False)

    require_columns(combine, ["nfl_id", "combine_position"], "combine_results")
    wr_ids = set(
        combine.loc[
            combine["combine_position"].astype(str).str.strip().str.upper().eq("WR"),
            "nfl_id",
        ].dropna()
    )
    raw = raw.loc[raw["nfl_id"].isin(wr_ids)].copy()
    print(f"Raw tracking rows for Combine WR population: {len(raw):,}")
    print(f"Combine WR population: {len(wr_ids):,}")

    print("\nBuilding corrected attempt features (Notebook 04 Cell 31)...")
    attempts = build_attempt_features(raw)
    save_table(
        attempts,
        feature_dir / "wr_combine_attempt_features_v2",
    )

    print("Building turning candidates (Notebook 04 Cells 35–37)...")
    turning = build_turning_candidates(raw)
    save_table(
        turning,
        feature_dir / "wr_combine_turning_candidate_features",
    )

    # Reproduce Notebook 04 Cell 41: select threshold 1 and enrich attempts.
    primary_turning = turning.loc[
        turning["speed_threshold"].eq(1.0)
    ].copy()

    if attempts.duplicated(ATTEMPT_KEYS).any():
        raise AssertionError("Attempt feature keys are not unique.")
    if primary_turning.duplicated(ATTEMPT_KEYS).any():
        raise AssertionError("Threshold-1 turning keys are not unique.")

    attempt_key_set = set(map(tuple, attempts[ATTEMPT_KEYS].to_numpy()))
    turning_key_set = set(map(tuple, primary_turning[ATTEMPT_KEYS].to_numpy()))
    only_attempts = attempt_key_set - turning_key_set
    only_turning = turning_key_set - attempt_key_set

    print("\nAttempt/turning key reconciliation:")
    print(f"  Attempt rows: {len(attempts):,}")
    print(f"  Threshold-1 turning rows: {len(primary_turning):,}")
    print(f"  Keys only in attempt features: {len(only_attempts):,}")
    print(f"  Keys only in turning features: {len(only_turning):,}")

    if only_attempts or only_turning:
        raise ValueError(
            "Attempt and threshold-1 turning keys do not match. "
            "Do not proceed to player-level aggregation until investigated."
        )

    turning_columns = ATTEMPT_KEYS + [
        "valid_turn_pairs",
        "coordinate_total_turn",
        "coordinate_mean_turn",
        "coordinate_median_turn",
        "coordinate_p95_turn",
        "coordinate_max_turn",
        "recorded_total_turn",
        "recorded_mean_turn",
        "speed_threshold",
        "turning_rate_degrees_per_second",
        "valid_pair_coverage",
    ]

    enriched = attempts.merge(
        primary_turning[turning_columns],
        on=ATTEMPT_KEYS,
        how="left",
        validate="one_to_one",
    )
    if len(enriched) != len(attempts):
        raise AssertionError("Enriched attempt row count changed.")
    if enriched["coordinate_total_turn"].isna().any():
        raise ValueError(
            "Some threshold-1 coordinate turning values are missing. "
            "This conflicts with the original Cell 41 assertion."
        )

    enriched["turning_coverage_flag"] = np.select(
        [
            enriched["valid_pair_coverage"] < 0.70,
            enriched["valid_pair_coverage"] < 0.80,
        ],
        ["LOW_BELOW_70", "MODERATE_70_TO_80"],
        default="GOOD_80_OR_ABOVE",
    )

    save_table(
        enriched,
        feature_dir / "wr_combine_attempt_features_enriched",
    )

    # Reproduce Cell 14 attempt quality audit.
    quality = raw.copy()
    for col in ["x", "y", "s", "a", "dis"]:
        quality[col] = pd.to_numeric(quality[col], errors="coerce")
    quality["time"] = pd.to_datetime(quality["time"], errors="coerce")
    quality = quality.loc[
        quality["entity_type"].astype(str).str.upper().eq("PLAYER")
    ].dropna(subset=ATTEMPT_KEYS + ["time"])
    quality = quality.sort_values(ATTEMPT_KEYS + ["time"], kind="mergesort")
    qgroup = quality.groupby(ATTEMPT_KEYS, sort=False, dropna=False)
    quality["previous_time"] = qgroup["time"].shift(1)
    quality["time_gap_seconds"] = (
        quality["time"] - quality["previous_time"]
    ).dt.total_seconds()

    quality_audit = (
        quality.groupby(ATTEMPT_KEYS, sort=False, dropna=False)
        .agg(
            observations=("time", "size"),
            start_time=("time", "min"),
            end_time=("time", "max"),
            missing_coordinates=(
                "x", lambda x: int(x.isna().sum())
            ),
            missing_speed=("s", lambda x: int(x.isna().sum())),
            missing_acceleration=("a", lambda x: int(x.isna().sum())),
            missing_displacement=("dis", lambda x: int(x.isna().sum())),
            minimum_time_gap=("time_gap_seconds", "min"),
            median_time_gap=("time_gap_seconds", "median"),
            maximum_time_gap=("time_gap_seconds", "max"),
            gaps_over_015_seconds=(
                "time_gap_seconds", lambda x: int((x > 0.15).sum())
            ),
            minimum_speed=("s", "min"),
            maximum_speed=("s", "max"),
            minimum_acceleration=("a", "min"),
            maximum_acceleration=("a", "max"),
        )
        .reset_index()
    )
    save_table(
        quality_audit,
        audit_dir / "combine_attempt_quality",
    )

    print("\nREBUILD SUMMARY")
    print(f"Attempt features: {len(attempts):,} rows")
    print(f"Turning candidates: {len(turning):,} rows")
    print(f"Enriched attempts: {len(enriched):,} rows")
    print(f"Players in attempt features: {attempts['nfl_id'].nunique():,}")
    print(
        "Thresholds:",
        sorted(turning["speed_threshold"].dropna().unique().tolist()),
    )
    print("\nCoverage flags:")
    print(enriched["turning_coverage_flag"].value_counts(dropna=False))
    print("\nPrimary drill coverage:")
    print(
        enriched.loc[enriched["drill_name"].isin(PRIMARY_DRILLS)]
        .groupby("drill_name")["nfl_id"]
        .agg(attempts="size", players="nunique")
        .sort_values("attempts", ascending=False)
    )

    print("\nFILES WRITTEN:")
    for name in [
        "wr_combine_attempt_features_v2",
        "wr_combine_turning_candidate_features",
        "wr_combine_attempt_features_enriched",
    ]:
        print(f"  {feature_dir / (name + '.parquet')}")
        print(f"  {feature_dir / (name + '.csv')}")
    print(f"  {audit_dir / 'combine_attempt_quality.parquet'}")
    print(f"  {audit_dir / 'combine_attempt_quality.csv'}")


if __name__ == "__main__":
    main()
