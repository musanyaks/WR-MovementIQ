# Data setup

Obtain the official NFL Big Data Bowl 2027 data from the competition's authorized Kaggle data page. This repository intentionally does not redistribute competition data.

Put these files in `data/raw/`:

- `combine_tracking.csv`
- `combine_results.csv`
- `player_play.csv`
- `players.csv`
- `games.csv`
- `game_tracking_2023.csv`
- `game_tracking_2024.csv`
- `game_tracking_2025.csv`

Filenames are configurable in `config/project.yaml`. Do not silently rename source columns. If the official schema changes, update the validation schema and document the change.

Run from the repository root:

```bash
pip install -e ".[dev]"
python run_pipeline.py --config config/project.yaml
pytest
```

The pipeline writes CSV and Parquet feature tables to `data/features/` and an input audit to `reports/data_audit/`.
