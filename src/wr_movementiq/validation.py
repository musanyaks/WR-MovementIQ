from collections.abc import Iterable
import pandas as pd

REQUIRED_COLUMNS = {
    "combine_tracking": {
        "draft_year", "nfl_id", "event_id", "drill_name", "attempt",
        "time", "x", "y", "s"
    },
    "combine_results": {"draft_year", "nfl_id", "combine_position"},
    "player_play": {"game_id", "play_id", "nfl_id"},
    "players": {"nfl_id", "nfl_position"},
    "games": {"game_id", "season"},
    "game_tracking": {"game_id", "play_id", "nfl_id", "time", "x", "y", "s"},
}


def validate_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def audit_frame(frame: pd.DataFrame, name: str) -> dict:
    return {
        "dataset": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "columns_with_missing": int(frame.isna().any().sum()),
        "column_names": ", ".join(map(str, frame.columns)),
    }
