# WR-MovementIQ
## NFL Big Data Bowl 2027 — Wide Receiver Movement Intelligence

A reproducible Python project for auditing NFL Combine receiver-drill tracking and regular-season receiver movement. The project is designed around transparent feature engineering, player linkage, player-grouped validation, and cautious interpretation.

**Research question:** To what extent do movement characteristics measured during NFL Combine receiver-specific drills explain variation in wide receivers' in-game movement characteristics during regular-season NFL plays?

> **Research status:** This is an executable pipeline foundation, not a claim that every historical notebook result has already been reproduced from a clean environment. The competition source data and the user's executed Colab notebooks are not bundled. The pipeline produces auditable movement features; the previously reported nested CV, cohort holdout, bootstrap and ablation experiments must be reconciled and added as analysis modules before describing this as a full reproduction of all research results.

## Quick start

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Place the authorized CSV files in `data/raw/` using the names in `docs/DATA_SETUP.md`, then run:

```bash
python run_pipeline.py --config config/project.yaml
pytest
```

## Outputs

- `reports/data_audit/input_data_audit.csv`
- `data/features/combine_attempt_movement_features.csv` and `.parquet`
- `data/features/game_player_play_movement_features.csv` and `.parquet`

## Important methodological safeguards

- The source acceleration field `a` is not assumed to be signed acceleration or braking.
- The pipeline keeps units marked `_raw` unless source units have been verified.
- Turning features use a configurable minimum-speed filter and report pair coverage.
- The game feature function summarizes all available observations per player-play. It does not claim to recreate a ball-snap-to-pass-forward window.
- Validation for predictive modelling must group by `nfl_id`; multiple seasons from one player must not leak across train/test partitions.
- This is observational research. Predictive association is not causation.
- Raw competition data is not included. Obtain it through the official Kaggle competition and follow its rules.

## Project structure

```text
WR-MovementIQ/
├── config/project.yaml
├── data/{raw,processed,features}/
├── docs/
├── reports/{data_audit,figures,model_results}/
├── src/wr_movementiq/
├── tests/
├── run_pipeline.py
├── pyproject.toml
└── README.md
```

See `docs/METHODOLOGY.md` and `docs/RESEARCH_STATUS.md`.
