import numpy as np
import pandas as pd

from wr_movementiq.features import (
    build_combine_attempt_features,
    build_game_play_features,
    circular_difference_degrees,
    coordinate_heading,
)


def test_circular_difference_wraparound():
    assert np.allclose(
        circular_difference_degrees(np.array([359.0]), np.array([1.0])),
        [2.0],
    )


def test_coordinate_heading_convention():
    assert np.allclose(
        coordinate_heading(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        [90.0, 0.0],
    )


def test_combine_attempt_features():
    frame = pd.DataFrame({
        "draft_year": [2025] * 5,
        "nfl_id": [100] * 5,
        "drill_type": ["SKILL"] * 5,
        "drill_name": ["TEST_ROUTE"] * 5,
        "attempt": [1] * 5,
        "event_id": [99] * 5,
        "time": pd.date_range("2025-01-01", periods=5, freq="100ms"),
        "x": [0, 1, 2, 3, 4],
        "y": [0, 0, 0, 0, 0],
        "s": [1, 2, 3, 2, 1],
    })
    result = build_combine_attempt_features(frame, minimum_speed=1.0)
    assert len(result) == 1
    assert np.isclose(result.iloc[0]["coordinate_path_length_raw"], 4.0)
    assert np.isclose(result.iloc[0]["path_efficiency"], 1.0)


def test_game_play_features():
    frame = pd.DataFrame({
        "game_id": [1, 1, 1],
        "play_id": [2, 2, 2],
        "nfl_id": [3, 3, 3],
        "time": pd.date_range("2025-01-01", periods=3, freq="100ms"),
        "x": [0, 1, 2],
        "y": [0, 0, 0],
        "s": [1, 2, 3],
    })
    result = build_game_play_features(frame)
    assert len(result) == 1
    assert result.iloc[0]["tracking_observations"] == 3
