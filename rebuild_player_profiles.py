from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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

MOVEMENT_FEATURES = [
    "duration_seconds",
    "mean_speed",
    "peak_speed",
    "p95_speed",
    "speed_std",
    "mean_acceleration",
    "peak_acceleration",
    "minimum_acceleration",
    "coordinate_path_length",
    "net_displacement",
    "path_efficiency",
    "coordinate_total_turn",
    "coordinate_mean_turn",
    "coordinate_p95_turn",
    "turning_rate_degrees_per_second",
]

QUALITY_FEATURES = [
    "observations",
    "valid_turn_pairs",
    "valid_pair_coverage",
    "large_sampling_gaps",
]

PROFILE_FEATURES = [
    "mean_speed",
    "peak_speed",
    "p95_speed",
    "mean_acceleration",
    "peak_acceleration",
    "coordinate_path_length",
    "net_displacement",
    "path_efficiency",
    "coordinate_total_turn",
    "coordinate_mean_turn",
    "coordinate_p95_turn",
    "turning_rate_degrees_per_second",
]

GAME_NUMERIC_CANDIDATES = [
    "mean_speed_raw",
    "peak_speed_raw",
    "p95_speed_raw",
    "mean_acceleration_raw",
    "peak_acceleration_raw",
    "minimum_acceleration_raw",
    "summed_recorded_displacement_raw",
    "coordinate_path_length_raw",
    "total_direction_change_raw",
    "path_efficiency",
    "snap_to_throw_seconds",
    "separation_at_pass_forward",
    "receiving_yards",
    "maximum_sampling_gap_seconds",
    "gaps_over_0_15_seconds",
    "rows_with_missing_movement_values",
    "tracking_observations",
]

GAME_MOVEMENT_FEATURES = [
    "mean_speed_raw",
    "peak_speed_raw",
    "p95_speed_raw",
    "mean_acceleration_raw",
    "peak_acceleration_raw",
    "minimum_acceleration_raw",
    "summed_recorded_displacement_raw",
    "coordinate_path_length_raw",
    "total_direction_change_raw",
    "path_efficiency",
    "snap_to_throw_seconds",
]


def save_pair(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_parquet(out_dir / f"{stem}.parquet", index=False)
    df.to_csv(out_dir / f"{stem}.csv", index=False)


def build_combine_profiles(features_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = features_dir / "wr_combine_attempt_features_enriched.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Run rebuild_combine_features.py first."
        )

    combine = pd.read_parquet(source)

    required = {
        "draft_year", "nfl_id", "drill_name", "attempt",
        *MOVEMENT_FEATURES, *QUALITY_FEATURES,
    }
    missing = sorted(required - set(combine.columns))
    if missing:
        raise ValueError(f"Enriched Combine file is missing columns: {missing}")

    available_drills = set(combine["drill_name"].dropna().unique())
    missing_drills = sorted(set(PRIMARY_DRILLS) - available_drills)
    if missing_drills:
        raise ValueError(f"Expected primary drills missing: {missing_drills}")

    primary = combine.loc[combine["drill_name"].isin(PRIMARY_DRILLS)].copy()

    group_keys = ["draft_year", "nfl_id", "drill_name"]
    aggregation = {feature: ["median", "std"] for feature in MOVEMENT_FEATURES}
    aggregation.update({feature: ["median"] for feature in QUALITY_FEATURES})

    player_drill = primary.groupby(group_keys).agg(aggregation)
    player_drill.columns = [
        f"{feature}_{stat}" for feature, stat in player_drill.columns
    ]
    player_drill = player_drill.reset_index()

    attempt_counts = (
        primary.groupby(group_keys)
        .size()
        .rename("attempt_count")
        .reset_index()
    )
    player_drill = player_drill.merge(
        attempt_counts,
        on=group_keys,
        how="left",
        validate="one_to_one",
    )

    profile_rows = []
    for (draft_year, nfl_id), group in player_drill.groupby(
        ["draft_year", "nfl_id"], sort=True
    ):
        row = {
            "draft_year": draft_year,
            "nfl_id": nfl_id,
            "primary_drills_observed": group["drill_name"].nunique(),
            "primary_attempts_observed": group["attempt_count"].sum(),
        }

        for feature in PROFILE_FEATURES:
            values = pd.to_numeric(
                group[f"{feature}_median"], errors="coerce"
            ).dropna()

            row[f"combine_{feature}_median_across_drills"] = (
                values.median() if len(values) else np.nan
            )
            row[f"combine_{feature}_iqr_across_drills"] = (
                values.quantile(0.75) - values.quantile(0.25)
                if len(values) >= 2 else np.nan
            )

        coverage = pd.to_numeric(
            group["valid_pair_coverage_median"], errors="coerce"
        )
        row["median_turning_pair_coverage"] = coverage.median()
        row["drills_with_turning_coverage_below_70_pct"] = (
            coverage.lt(0.70).sum()
        )
        profile_rows.append(row)

    player_profile = pd.DataFrame(profile_rows)

    if player_profile["nfl_id"].duplicated().any():
        raise AssertionError("Combine player profile has duplicate nfl_id values.")
    if len(player_profile) != 107:
        raise AssertionError(
            f"Expected 107 Combine WR profiles; found {len(player_profile)}."
        )

    save_pair(player_drill, features_dir, "wr_combine_player_drill_profiles")
    save_pair(player_profile, features_dir, "wr_combine_player_profiles")

    print("\nCOMBINE PLAYER-DRILL PROFILES")
    print(f"Rows: {len(player_drill):,}")
    print(f"Players: {player_drill['nfl_id'].nunique():,}")
    print(f"Unique player-drill keys: {not player_drill.duplicated(group_keys).any()}")

    print("\nCOMBINE PLAYER PROFILES")
    print(f"Rows: {len(player_profile):,}")
    print(f"Players: {player_profile['nfl_id'].nunique():,}")
    print(f"Columns: {len(player_profile.columns):,}")
    print(f"Median primary drills/player: {player_profile['primary_drills_observed'].median():.1f}")
    print(f"Median primary attempts/player: {player_profile['primary_attempts_observed'].median():.1f}")

    return player_drill, player_profile


def build_game_player_season_profiles(
    features_dir: Path, combine_player_profile: pd.DataFrame
) -> pd.DataFrame:
    source = features_dir / "wr_game_movement_features.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Run game_pipeline.py first."
        )

    game = pd.read_parquet(source)

    required = {
        "nfl_id", "season", "position", "play_id", "game_id", "route_ran",
        *GAME_MOVEMENT_FEATURES,
    }
    missing = sorted(required - set(game.columns))
    if missing:
        raise ValueError(f"Game movement file is missing required columns: {missing}")

    combine_players = (
        combine_player_profile[["draft_year", "nfl_id"]].drop_duplicates()
    )
    game_cohort = game.merge(
        combine_players,
        on="nfl_id",
        how="inner",
        validate="many_to_one",
    )

    game_post = game_cohort.loc[
        (game_cohort["season"] >= game_cohort["draft_year"])
        & game_cohort["position"].isin(["WR", "SLOT_WR"])
    ].copy()

    for col in GAME_NUMERIC_CANDIDATES:
        if col in game_post.columns:
            game_post[col] = pd.to_numeric(game_post[col], errors="coerce")

    game_aggregation = {
        feature: ["median", "mean"] for feature in GAME_MOVEMENT_FEATURES
    }
    game_aggregation.update({
        "separation_at_pass_forward": ["median", "mean"],
        "receiving_yards": ["median", "mean"],
        "tracking_observations": ["median"],
        "maximum_sampling_gap_seconds": ["median", "max"],
        "gaps_over_0_15_seconds": ["sum"],
        "rows_with_missing_movement_values": ["sum"],
    })

    # Match Notebook 05 Cell 29: the secondary columns are expected in the
    # original game feature file. Fail clearly rather than silently changing
    # the aggregation if the local schema differs.
    secondary = [
        "separation_at_pass_forward",
        "receiving_yards",
        "tracking_observations",
        "maximum_sampling_gap_seconds",
        "gaps_over_0_15_seconds",
        "rows_with_missing_movement_values",
    ]
    missing_secondary = sorted(set(secondary) - set(game_post.columns))
    if missing_secondary:
        raise ValueError(
            "Game movement file lacks Cell 29 secondary aggregation columns: "
            f"{missing_secondary}"
        )

    player_season = game_post.groupby(
        ["draft_year", "season", "nfl_id"]
    ).agg(game_aggregation)
    player_season.columns = [
        f"game_{feature}_{stat}" for feature, stat in player_season.columns
    ]
    player_season = player_season.reset_index()

    exposure = (
        game_post.groupby(["draft_year", "season", "nfl_id"])
        .agg(
            receiver_plays=("play_id", "size"),
            games=("game_id", "nunique"),
            route_types_observed=("route_ran", "nunique"),
            plays_with_receiving_yards=("receiving_yards", "count"),
            plays_with_separation=("separation_at_pass_forward", "count"),
        )
        .reset_index()
    )

    player_season = player_season.merge(
        exposure,
        on=["draft_year", "season", "nfl_id"],
        how="left",
        validate="one_to_one",
    )
    player_season["receiving_yards_coverage_pct"] = (
        player_season["plays_with_receiving_yards"]
        / player_season["receiver_plays"] * 100
    )
    player_season["separation_coverage_pct"] = (
        player_season["plays_with_separation"]
        / player_season["receiver_plays"] * 100
    )

    keys = ["draft_year", "season", "nfl_id"]
    if player_season.duplicated(keys).any():
        raise AssertionError("Duplicate player-season keys found.")

    save_pair(player_season, features_dir, "wr_game_player_season_profiles")

    print("\nGAME PLAYER-SEASON PROFILES")
    print(f"Rows: {len(player_season):,}")
    print(f"Unique players: {player_season['nfl_id'].nunique():,}")
    print(f"Draft cohorts: {sorted(player_season['draft_year'].dropna().unique().tolist())}")
    print("\nSeason distribution:")
    print(
        player_season.groupby("season")
        .agg(
            player_seasons=("nfl_id", "size"),
            unique_players=("nfl_id", "nunique"),
            median_receiver_plays=("receiver_plays", "median"),
        )
        .reset_index()
        .to_string(index=False)
    )

    return player_season


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild WR-MovementIQ player/drill and player-season profiles."
    )
    parser.add_argument(
        "--features-dir",
        default="data/features",
        help="Project features directory (default: data/features).",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    print("=" * 72)
    print("WR-MovementIQ — NOTEBOOK 05 CELLS 28–29 RECONSTRUCTION")
    print("=" * 72)

    _, combine_profile = build_combine_profiles(features_dir)
    player_season = build_game_player_season_profiles(
        features_dir, combine_profile
    )

    print("\nOUTPUT FILES")
    for name in [
        "wr_combine_player_drill_profiles",
        "wr_combine_player_profiles",
        "wr_game_player_season_profiles",
    ]:
        print(f"  {features_dir / (name + '.parquet')}")
        print(f"  {features_dir / (name + '.csv')}")

    print("\nREBUILD COMPLETE")
    print(
        "Expected original reference: 146 player-seasons / 79 players. "
        "Compare these counts and the resulting rows with the Colab outputs "
        "before treating this as an exact reproduction."
    )


if __name__ == "__main__":
    main()
