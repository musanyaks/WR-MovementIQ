import argparse
from pathlib import Path

from wr_movementiq.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the WR-MovementIQ feature engineering pipeline."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/project.yaml")
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
