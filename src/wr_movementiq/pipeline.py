from pathlib import Path
import pandas as pd
import yaml

from .features import build_combine_attempt_features, build_game_play_features
from .io import read_csv_required, write_table
from .validation import REQUIRED_COLUMNS, audit_frame, validate_columns


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def run_pipeline(config_path: Path) -> None:
    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    raw = root / cfg["paths"]["raw"]
    features_dir = root / cfg["paths"]["features"]
    audit_dir = root / cfg["paths"]["audit"]
    features_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    names = cfg["files"]
    datasets = {
        "combine_tracking": read_csv_required(raw / names["combine_tracking"]),
        "combine_results": read_csv_required(raw / names["combine_results"]),
        "player_play": read_csv_required(raw / names["player_play"]),
        "players": read_csv_required(raw / names["players"]),
        "games": read_csv_required(raw / names["games"]),
    }
    game_frames = [
        read_csv_required(raw / filename) for filename in names["game_tracking"]
    ]
    datasets["game_tracking"] = pd.concat(game_frames, ignore_index=True)

    audit_rows = []
    for name, frame in datasets.items():
        validate_columns(frame, REQUIRED_COLUMNS[name], name)
        audit_rows.append(audit_frame(frame, name))
    pd.DataFrame(audit_rows).to_csv(
        audit_dir / "input_data_audit.csv", index=False
    )

    wr_ids = set(
        datasets["combine_results"]
        .loc[
            datasets["combine_results"]["combine_position"]
            .astype(str).str.upper().eq("WR"),
            "nfl_id",
        ]
        .dropna()
    )

    combine_wr = datasets["combine_tracking"].loc[
        datasets["combine_tracking"]["nfl_id"].isin(wr_ids)
    ].copy()
    if "entity_type" in combine_wr.columns:
        combine_wr = combine_wr.loc[combine_wr["entity_type"].eq("PLAYER")]

    minimum_speed = float(cfg["features"]["minimum_speed_for_turning"])
    minimum_obs = int(cfg["features"]["minimum_observations_per_attempt"])
    combine_features = build_combine_attempt_features(
        combine_wr, minimum_speed, minimum_obs
    )

    game_wr = datasets["game_tracking"].loc[
        datasets["game_tracking"]["nfl_id"].isin(wr_ids)
    ].copy()
    game_features = build_game_play_features(game_wr)

    write_table(combine_features, features_dir / "combine_attempt_movement_features")
    write_table(game_features, features_dir / "game_player_play_movement_features")

    print(f"Input audit: {audit_dir / 'input_data_audit.csv'}")
    print(f"Combine attempt feature rows: {len(combine_features):,}")
    print(f"Game player-play feature rows: {len(game_features):,}")
    print("Pipeline complete. Review data coverage, event windows and units before modelling.")
