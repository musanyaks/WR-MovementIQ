
import numpy as np
import pandas as pd

ATTEMPT_KEYS = [
    "draft_year", "nfl_id", "drill_type", "drill_name", "attempt", "event_id"
]
PLAY_KEYS = ["game_id", "play_id", "nfl_id"]


def circular_difference_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed angular difference from heading a to heading b."""
    return (b - a + 180.0) % 360.0 - 180.0


def coordinate_heading(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Clockwise heading from positive Y."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_combine_attempt_features(
    tracking: pd.DataFrame,
    minimum_speed: float = 1.0,
    minimum_observations: int = 5,
) -> pd.DataFrame:

    required = ATTEMPT_KEYS + ["time", "x", "y", "s"]
    missing = sorted(set(required) - set(tracking.columns))

    if missing:
        raise ValueError(f"Combine tracking missing required columns: {missing}")

    df = _coerce_numeric(tracking, ["x", "y", "s"])

    if "entity_type" in df.columns:
        df = df.loc[df["entity_type"].eq("PLAYER")].copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.dropna(
        subset=ATTEMPT_KEYS + ["time", "x", "y", "s"]
    )

    df = df.sort_values(ATTEMPT_KEYS + ["time"])

    records = []

    for key, group in df.groupby(
        ATTEMPT_KEYS, dropna=False, sort=False
    ):

        group = group.reset_index(drop=True)

        if len(group) < minimum_observations:
            continue

        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        speed = group["s"].to_numpy(dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)

        step_distance = np.hypot(dx, dy)

        path_length = float(step_distance.sum())

        net_displacement = float(
            np.hypot(x[-1] - x[0], y[-1] - y[0])
        )

        # One heading per coordinate transition.
        headings = coordinate_heading(dx, dy)

        # Heading changes compare two consecutive movement segments.
        heading_change = np.abs(
            circular_difference_degrees(
                headings[:-1],
                headings[1:]
            )
        )

        # Three observations surround each heading transition.
        valid_heading_transitions = (
            (speed[:-2] >= minimum_speed)
            & (speed[1:-1] >= minimum_speed)
            & (speed[2:] >= minimum_speed)
        )

        valid_turns = heading_change[valid_heading_transitions]

        duration = float(
            (
                group["time"].iloc[-1]
                - group["time"].iloc[0]
            ).total_seconds()
        )

        record = dict(zip(ATTEMPT_KEYS, key))

        record.update({
            "tracking_observations": int(len(group)),
            "duration_seconds": duration,

            "mean_speed_raw": float(speed.mean()),
            "peak_speed_raw": float(speed.max()),
            "p95_speed_raw": float(np.quantile(speed, 0.95)),

            "coordinate_path_length_raw": path_length,
            "net_displacement_raw": net_displacement,

            "path_efficiency": (
                net_displacement / path_length
                if path_length > 0 else np.nan
            ),

            "valid_turning_pairs": int(
                valid_heading_transitions.sum()
            ),

            "turning_pair_coverage": (
                float(valid_heading_transitions.mean())
                if len(valid_heading_transitions) else np.nan
            ),

            "coordinate_total_turn_deg_speed_filtered": float(
                valid_turns.sum()
            ),

            "coordinate_mean_turn_deg_speed_filtered": (
                float(valid_turns.mean())
                if len(valid_turns) else np.nan
            ),

            "turning_rate_deg_per_second": (
                float(valid_turns.sum() / duration)
                if duration > 0 and len(valid_turns) else np.nan
            ),
        })

        records.append(record)

    return pd.DataFrame.from_records(records)


def build_game_play_features(tracking: pd.DataFrame) -> pd.DataFrame:

    required = PLAY_KEYS + ["time", "x", "y", "s"]

    missing = sorted(set(required) - set(tracking.columns))

    if missing:
        raise ValueError(
            f"Game tracking missing required columns: {missing}"
        )

    df = _coerce_numeric(tracking, ["x", "y", "s"])

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.dropna(
        subset=PLAY_KEYS + ["time", "x", "y", "s"]
    )

    df = df.sort_values(PLAY_KEYS + ["time"])

    records = []

    for key, group in df.groupby(
        PLAY_KEYS, dropna=False, sort=False
    ):

        group = group.reset_index(drop=True)

        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        speed = group["s"].to_numpy(dtype=float)

        path_length = float(
            np.hypot(np.diff(x), np.diff(y)).sum()
        )

        net_displacement = float(
            np.hypot(x[-1] - x[0], y[-1] - y[0])
        )

        record = dict(zip(PLAY_KEYS, key))

        record.update({
            "tracking_observations": int(len(group)),

            "mean_speed_raw": float(speed.mean()),
            "peak_speed_raw": float(speed.max()),
            "p95_speed_raw": float(np.quantile(speed, 0.95)),

            "coordinate_path_length_raw": path_length,
            "net_displacement_raw": net_displacement,

            "path_efficiency": (
                net_displacement / path_length
                if path_length > 0 else np.nan
            ),
        })

        records.append(record)

    return pd.DataFrame.from_records(records)
