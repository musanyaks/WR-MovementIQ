
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = (
    NOTEBOOK_DIR /
    "08C_receiver_level_signal_and_model_discrimination.ipynb"
)

nb = nbf.v4.new_notebook()

cells = []

def markdown(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))


markdown("""# NOTEBOOK 08C
## Receiver-Level Signal and Model Discrimination

Objectives:
1. Evaluate performance within each outer fold.
2. Compare models against the training-mean baseline.
3. Aggregate predictions at receiver level.
4. Evaluate receiver-level discrimination.
5. Perform receiver-cluster bootstrap.
6. Investigate error concentration.
7. Examine draft-cohort performance.

This notebook does not refit models or modify the frozen dataset.
""")

code("""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path.cwd()

if ROOT.name == "notebooks":
    ROOT = ROOT.parent

OOF_PATH = ROOT / "reports/nested_ridge_modeling/oof_predictions.csv"
DATASET_PATH = ROOT / "data/features/wr_analysis_dataset_core_common.parquet"

OUTPUT_DIR = ROOT / "reports/receiver_level_signal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 2027

print("Project root:", ROOT)
print("OOF exists:", OOF_PATH.exists())
print("Dataset exists:", DATASET_PATH.exists())
""")

code("""
if not OOF_PATH.exists():
    raise FileNotFoundError(OOF_PATH)

oof = pd.read_csv(OOF_PATH)

required = [
    "outcome", "model", "nfl_id", "season",
    "draft_year", "outer_fold", "actual", "predicted"
]

missing = set(required) - set(oof.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

for col in [
    "actual", "predicted", "season",
    "draft_year", "outer_fold"
]:
    oof[col] = pd.to_numeric(oof[col], errors="coerce")

oof["nfl_id"] = oof["nfl_id"].astype(str)

if oof[required].isna().any().any():
    raise ValueError("Missing values in required OOF fields.")

print("Rows:", len(oof))
print("Receivers:", oof.nfl_id.nunique())
print("Outcomes:", sorted(oof.outcome.unique()))
print("Models:", sorted(oof.model.unique()))
print("Folds:", sorted(oof.outer_fold.unique()))
""")

code("""
EXPECTED_MODELS = {
    "BASELINE", "TRADITIONAL",
    "TRACKING", "COMBINED"
}

if set(oof.model.unique()) != EXPECTED_MODELS:
    raise ValueError("Unexpected model set.")

if oof.duplicated(
    ["outcome", "model", "nfl_id", "season", "draft_year"]
).any():
    raise ValueError("Duplicate OOF keys detected.")

coverage = (
    oof.groupby(["outcome", "model"])
    .agg(
        observations=("nfl_id", "size"),
        receivers=("nfl_id", "nunique")
    )
    .reset_index()
)

display(coverage)

for outcome in oof.outcome.unique():

    samples = {}

    for model in EXPECTED_MODELS:

        part = oof[
            (oof.outcome == outcome) &
            (oof.model == model)
        ]

        samples[model] = set(
            zip(part.nfl_id, part.season, part.draft_year)
        )

    reference = samples["BASELINE"]

    for model in EXPECTED_MODELS:

        if samples[model] != reference:
            raise ValueError(
                f"Sample mismatch: {outcome}, {model}"
            )

print("PASS: identical player-season samples.")
""")

code("""
def safe_corr(x, y, method="pearson"):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    if method == "pearson":
        return float(pearsonr(x, y).statistic)

    return float(spearmanr(x, y).statistic)
""")

markdown("## 1. Outer-fold performance")

code("""
records = []

for (outcome, model, fold), part in oof.groupby(
    ["outcome", "model", "outer_fold"]
):

    actual = part.actual.to_numpy(float)
    predicted = part.predicted.to_numpy(float)

    actual_sd = np.std(actual, ddof=1)
    prediction_sd = np.std(predicted, ddof=1)

    slope = (
        np.cov(actual, predicted, ddof=1)[0, 1]
        / np.var(actual, ddof=1)
        if actual_sd > 1e-12 else np.nan
    )

    records.append({
        "outcome": outcome,
        "model": model,
        "outer_fold": int(fold),
        "observations": len(part),
        "receivers": part.nfl_id.nunique(),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
        "pearson_r": safe_corr(actual, predicted),
        "spearman_rho": safe_corr(actual, predicted, "spearman"),
        "prediction_sd_ratio": (
            prediction_sd / actual_sd
            if actual_sd > 1e-12 else np.nan
        ),
        "calibration_slope": slope
    })

fold_metrics = pd.DataFrame(records)

baseline = (
    fold_metrics[fold_metrics.model == "BASELINE"]
    [["outcome", "outer_fold", "mae", "rmse"]]
    .rename(columns={
        "mae": "baseline_mae",
        "rmse": "baseline_rmse"
    })
)

fold_metrics = fold_metrics.merge(
    baseline,
    on=["outcome", "outer_fold"],
    how="left"
)

fold_metrics["mae_difference_vs_baseline"] = (
    fold_metrics.mae - fold_metrics.baseline_mae
)

fold_metrics["rmse_difference_vs_baseline"] = (
    fold_metrics.rmse - fold_metrics.baseline_rmse
)

fold_metrics.to_csv(
    OUTPUT_DIR / "fold_level_model_metrics.csv",
    index=False
)

display(fold_metrics.sort_values(
    ["outcome", "model", "outer_fold"]
))
""")

markdown("## 2. Receiver-level aggregation")

code("""
receiver_predictions = (
    oof.groupby(["outcome", "model", "nfl_id"], as_index=False)
    .agg(
        actual_receiver_mean=("actual", "mean"),
        predicted_receiver_mean=("predicted", "mean"),
        seasons=("season", "nunique"),
        player_seasons=("season", "size"),
        draft_year=("draft_year", "min")
    )
)

receiver_predictions.to_csv(
    OUTPUT_DIR / "receiver_level_predictions.csv",
    index=False
)

records = []

for (outcome, model), part in receiver_predictions.groupby(
    ["outcome", "model"]
):

    actual = part.actual_receiver_mean.to_numpy(float)
    predicted = part.predicted_receiver_mean.to_numpy(float)

    actual_sd = np.std(actual, ddof=1)
    prediction_sd = np.std(predicted, ddof=1)

    records.append({
        "outcome": outcome,
        "model": model,
        "receivers": len(part),
        "pearson_r": safe_corr(actual, predicted),
        "spearman_rho": safe_corr(actual, predicted, "spearman"),
        "receiver_mean_mae": mean_absolute_error(actual, predicted),
        "receiver_mean_rmse": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "receiver_mean_r2": r2_score(actual, predicted),
        "prediction_sd_ratio": (
            prediction_sd / actual_sd
            if actual_sd > 1e-12 else np.nan
        )
    })

receiver_metrics = pd.DataFrame(records)

receiver_metrics.to_csv(
    OUTPUT_DIR / "receiver_level_discrimination.csv",
    index=False
)

display(receiver_metrics.sort_values(["outcome", "model"]))
""")

markdown("## 3. Receiver-level errors")

code("""
errors = oof.copy()

errors["absolute_error"] = (
    errors.actual - errors.predicted
).abs()

receiver_error = (
    errors.groupby(
        ["outcome", "model", "nfl_id"],
        as_index=False
    )
    .agg(receiver_mae=("absolute_error", "mean"))
)

display(receiver_error.head())
""")

markdown("## 4. Receiver-cluster bootstrap")

code("""
def bootstrap_receiver_mae(
    outcome, model, rng, n_boot=BOOTSTRAP_REPLICATES
):

    part = receiver_error[
        (receiver_error.outcome == outcome) &
        (receiver_error.model == model)
    ]

    values = part.receiver_mae.to_numpy(float)
    n = len(values)

    if n < 5:
        return np.nan, np.nan, np.nan, n

    boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = values[idx].mean()

    return (
        values.mean(),
        np.quantile(boot, 0.025),
        np.quantile(boot, 0.975),
        n
    )


def bootstrap_paired_difference(
    outcome, model, rng, n_boot=BOOTSTRAP_REPLICATES
):

    m = receiver_error[
        (receiver_error.outcome == outcome) &
        (receiver_error.model == model)
    ][["nfl_id", "receiver_mae"]].rename(
        columns={"receiver_mae": "model_mae"}
    )

    b = receiver_error[
        (receiver_error.outcome == outcome) &
        (receiver_error.model == "BASELINE")
    ][["nfl_id", "receiver_mae"]].rename(
        columns={"receiver_mae": "baseline_mae"}
    )

    paired = m.merge(
        b, on="nfl_id", validate="one_to_one"
    )

    differences = (
        paired.model_mae - paired.baseline_mae
    ).to_numpy(float)

    n = len(differences)

    if n < 5:
        return np.nan, np.nan, np.nan, n

    boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = differences[idx].mean()

    return (
        differences.mean(),
        np.quantile(boot, 0.025),
        np.quantile(boot, 0.975),
        n
    )


rng = np.random.default_rng(RANDOM_SEED)

records = []

for outcome in sorted(oof.outcome.unique()):

    for model in sorted(EXPECTED_MODELS):

        estimate, low, high, n = bootstrap_receiver_mae(
            outcome, model, rng
        )

        diff, diff_low, diff_high, n_pair = (
            bootstrap_paired_difference(
                outcome, model, rng
            )
        )

        records.append({
            "outcome": outcome,
            "model": model,
            "receivers": n,
            "receiver_equal_weight_mae": estimate,
            "mae_ci_lower": low,
            "mae_ci_upper": high,
            "mae_difference_vs_baseline": diff,
            "difference_ci_lower": diff_low,
            "difference_ci_upper": diff_high,
            "paired_receivers": n_pair
        })

bootstrap_results = pd.DataFrame(records)

bootstrap_results.to_csv(
    OUTPUT_DIR / "receiver_cluster_bootstrap.csv",
    index=False
)

display(bootstrap_results.sort_values(["outcome", "model"]))
""")

markdown("## 5. Error concentration")

code("""
records = []

for (outcome, model), part in receiver_error.groupby(
    ["outcome", "model"]
):

    ranked = part.sort_values(
        "receiver_mae",
        ascending=False
    )

    total = ranked.receiver_mae.sum()

    records.append({
        "outcome": outcome,
        "model": model,
        "receivers": len(ranked),
        "top5_receiver_error_share": (
            ranked.head(5).receiver_mae.sum() / total
            if total > 0 else np.nan
        ),
        "top10_receiver_error_share": (
            ranked.head(10).receiver_mae.sum() / total
            if total > 0 else np.nan
        ),
        "worst_receiver_mae": ranked.receiver_mae.max(),
        "median_receiver_mae": ranked.receiver_mae.median()
    })

error_concentration = pd.DataFrame(records)

error_concentration.to_csv(
    OUTPUT_DIR / "receiver_error_concentration.csv",
    index=False
)

display(error_concentration)
""")

markdown("## 6. Draft-cohort descriptive metrics")

code("""
cohort_predictions = (
    oof.groupby(
        ["outcome", "model", "nfl_id", "draft_year"],
        as_index=False
    )
    .agg(
        actual=("actual", "mean"),
        predicted=("predicted", "mean"),
        player_seasons=("season", "size")
    )
)

records = []

for (outcome, model, cohort), part in cohort_predictions.groupby(
    ["outcome", "model", "draft_year"]
):

    actual = part.actual.to_numpy(float)
    predicted = part.predicted.to_numpy(float)

    records.append({
        "outcome": outcome,
        "model": model,
        "draft_year": int(cohort),
        "receivers": part.nfl_id.nunique(),
        "pearson_r": safe_corr(actual, predicted),
        "spearman_rho": safe_corr(actual, predicted, "spearman"),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted))
    })

cohort_metrics = pd.DataFrame(records)

cohort_metrics.to_csv(
    OUTPUT_DIR / "draft_cohort_receiver_metrics.csv",
    index=False
)

display(cohort_metrics.sort_values(
    ["outcome", "model", "draft_year"]
))
""")

markdown("## 7. Model versus baseline decision table")

code("""
comparison = bootstrap_results[
    bootstrap_results.model != "BASELINE"
].copy()

comparison["bootstrap_conclusion"] = np.select(
    [
        comparison.difference_ci_upper < 0,
        comparison.difference_ci_lower > 0
    ],
    [
        "IMPROVEMENT",
        "WORSE_THAN_BASELINE"
    ],
    default="UNCERTAIN"
)

comparison.to_csv(
    OUTPUT_DIR / "model_vs_baseline_decisions.csv",
    index=False
)

display(comparison[[
    "outcome",
    "model",
    "mae_difference_vs_baseline",
    "difference_ci_lower",
    "difference_ci_upper",
    "bootstrap_conclusion"
]])
""")

markdown("## 8. Research integrity manifest")

code("""
def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


manifest = {
    "notebook": "08C_receiver_level_signal_and_model_discrimination",
    "oof_file": str(OOF_PATH.relative_to(ROOT)),
    "dataset_file": str(DATASET_PATH.relative_to(ROOT)),
    "dataset_sha256": (
        sha256_file(DATASET_PATH)
        if DATASET_PATH.exists() else None
    ),
    "oof_rows": int(len(oof)),
    "receivers": int(oof.nfl_id.nunique()),
    "outcomes": sorted(oof.outcome.unique().tolist()),
    "models": sorted(oof.model.unique().tolist()),
    "outer_folds": sorted(
        oof.outer_fold.astype(int).unique().tolist()
    ),
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": RANDOM_SEED,
    "bootstrap_unit": "nfl_id",
    "models_refitted": False,
    "dataset_modified": False
}

with open(
    OUTPUT_DIR / "receiver_level_signal_manifest.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=4)

print(json.dumps(manifest, indent=4))
""")

markdown("## 9. Final output audit")

code("""
expected_outputs = [
    "fold_level_model_metrics.csv",
    "receiver_level_predictions.csv",
    "receiver_level_discrimination.csv",
    "receiver_cluster_bootstrap.csv",
    "receiver_error_concentration.csv",
    "draft_cohort_receiver_metrics.csv",
    "model_vs_baseline_decisions.csv",
    "receiver_level_signal_manifest.json"
]

missing = []

for filename in expected_outputs:

    path = OUTPUT_DIR / filename

    if path.exists():
        print(f"[PASS] {filename}")
    else:
        print(f"[FAIL] {filename}")
        missing.append(filename)

if missing:
    raise RuntimeError(f"Missing outputs: {missing}")

print("\\nNOTEBOOK 08C COMPLETE")
print("Rows:", len(oof))
print("Receivers:", oof.nfl_id.nunique())
print("Outputs:", OUTPUT_DIR)
""")

nb.cells = cells

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10"
    }
}

nbf.validate(nb)

nbf.write(nb, OUTPUT)

print("=" * 70)
print("NOTEBOOK 08C CREATED SUCCESSFULLY")
print("=" * 70)
print("Path:", OUTPUT)
print("Cells:", len(nb.cells))
print("Valid Jupyter Notebook JSON: YES")
