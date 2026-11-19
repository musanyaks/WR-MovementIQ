
from pathlib import Path
import nbformat as nbf

# ============================================================
# WR-MovementIQ — NOTEBOOK 08D GENERATOR
# ============================================================

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = NOTEBOOK_DIR / "08D_fold_stratified_receiver_validation.ipynb"

nb = nbf.v4.new_notebook()

cells = []

# ------------------------------------------------------------
# CELL 1 — TITLE
# ------------------------------------------------------------

cells.append(nbf.v4.new_markdown_cell("""
# Notebook 08D — Fold-Stratified Receiver Validation

## WR-MovementIQ | NFL Big Data Bowl 2027

### Research objectives

Evaluate whether Combine-derived movement features contain
reproducible predictive information about NFL regular-season
wide receiver movement performance.

### Validation design

- Preserve Notebook 07 outer GroupKFold assignments.
- Use frozen Notebook 08 out-of-fold predictions.
- Aggregate predictions at receiver level within each fold.
- Calculate fold-specific Pearson and Spearman correlations.
- Compare paired receiver-level MAE against baseline.
- Perform receiver-cluster bootstrap within each fold.
- Use 5,000 bootstrap replications.

Negative MAE differences favour the candidate model.

No models are refitted in this notebook.
"""))

# ------------------------------------------------------------
# CELL 2 — IMPORTS AND CONFIGURATION
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

SEED = 2027
N_BOOT = 5000

ROOT = Path.cwd()

OOF_PATH = ROOT / "reports/nested_ridge_modeling/oof_predictions.csv"
FOLD_PATH = ROOT / "reports/modeling_design/outer_grouped_fold_assignments.csv"
DATASET_PATH = ROOT / "data/features/wr_analysis_dataset_core_common.parquet"

OUT_DIR = ROOT / "reports/fold_stratified_receiver_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [OOF_PATH, FOLD_PATH, DATASET_PATH]:
    if not p.exists():
        raise FileNotFoundError(p)

print("Project:", ROOT)
print("Output:", OUT_DIR)
"""))

# ------------------------------------------------------------
# CELL 3 — LOAD AND AUDIT
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
oof = pd.read_csv(OOF_PATH)
folds = pd.read_csv(FOLD_PATH)
dataset = pd.read_parquet(DATASET_PATH)

required_oof = {
    "outcome", "model", "nfl_id", "season",
    "draft_year", "outer_fold", "actual", "predicted"
}

assert required_oof.issubset(oof.columns)
assert {"nfl_id", "outer_fold"}.issubset(folds.columns)

oof["nfl_id"] = oof["nfl_id"].astype(str)
folds["nfl_id"] = folds["nfl_id"].astype(str)
dataset["nfl_id"] = dataset["nfl_id"].astype(str)

oof["outer_fold"] = oof["outer_fold"].astype(int)
folds["outer_fold"] = folds["outer_fold"].astype(int)

assert folds["nfl_id"].is_unique
assert folds["outer_fold"].nunique() == 5
assert oof["nfl_id"].nunique() == 54

expected_models = {
    "BASELINE", "TRADITIONAL", "TRACKING", "COMBINED"
}
assert set(oof["model"].unique()) == expected_models

# Verify original Notebook 07 fold assignments.
check = oof.merge(
    folds,
    on="nfl_id",
    how="left",
    suffixes=("", "_assigned"),
    validate="many_to_one"
)

assert check["outer_fold_assigned"].notna().all()
assert (check["outer_fold"] == check["outer_fold_assigned"]).all()

# Verify complete player-season coverage.
keys = ["nfl_id", "season", "draft_year"]
expected_keys = set(map(tuple, dataset[keys].drop_duplicates().to_numpy()))

for (outcome, model), group in oof.groupby(["outcome", "model"]):
    actual_keys = set(map(tuple, group[keys].drop_duplicates().to_numpy()))
    assert actual_keys == expected_keys, (outcome, model)

print("AUDIT PASS")
print("OOF rows:", len(oof))
print("Receivers:", oof.nfl_id.nunique())
print("Outer folds:", folds.outer_fold.nunique())
print(folds.groupby("outer_fold").nfl_id.nunique())
"""))

# ------------------------------------------------------------
# CELL 4 — RECEIVER-LEVEL AGGREGATION
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
receiver = (
    oof.groupby(
        ["outcome", "model", "outer_fold", "nfl_id"],
        as_index=False
    )
    .agg(
        actual=("actual", "mean"),
        predicted=("predicted", "mean"),
        player_seasons=("season", "nunique")
    )
)

receiver["absolute_error"] = (
    receiver["actual"] - receiver["predicted"]
).abs()

receiver["squared_error"] = (
    receiver["actual"] - receiver["predicted"]
) ** 2

assert (
    receiver.groupby(["outcome", "model", "nfl_id"])
    .outer_fold.nunique().max()
    == 1
)

receiver.to_csv(
    OUT_DIR / "receiver_level_within_fold_predictions.csv",
    index=False
)

display(receiver.head())
"""))

# ------------------------------------------------------------
# CELL 5 — METRIC FUNCTIONS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            return float(pearsonr(x, y).statistic)
        return float(spearmanr(x, y).statistic)


def sd_ratio(actual, predicted):
    a = np.std(actual, ddof=1)
    p = np.std(predicted, ddof=1)
    return p / a if a > 0 else np.nan
"""))

# ------------------------------------------------------------
# CELL 6 — FOLD METRICS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
rows = []

for (outcome, model, fold), g in receiver.groupby(
    ["outcome", "model", "outer_fold"]
):
    actual = g.actual.to_numpy()
    predicted = g.predicted.to_numpy()

    rows.append({
        "outcome": outcome,
        "model": model,
        "outer_fold": int(fold),
        "receivers": g.nfl_id.nunique(),
        "player_seasons": int(g.player_seasons.sum()),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
        "pearson_r": safe_corr(actual, predicted, "pearson"),
        "spearman_rho": safe_corr(actual, predicted, "spearman"),
        "prediction_sd_ratio": sd_ratio(actual, predicted),
        "mean_actual": actual.mean(),
        "mean_predicted": predicted.mean(),
        "mean_residual": (predicted - actual).mean()
    })

fold_metrics = pd.DataFrame(rows)

fold_metrics.to_csv(
    OUT_DIR / "fold_stratified_receiver_metrics.csv",
    index=False
)

display(fold_metrics.round(4))
"""))

# ------------------------------------------------------------
# CELL 7 — CORRELATION SUMMARY
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
corr_summary = (
    fold_metrics.groupby(["outcome", "model"])
    .agg(
        folds_with_pearson=("pearson_r", "count"),
        pearson_mean=("pearson_r", "mean"),
        pearson_median=("pearson_r", "median"),
        pearson_min=("pearson_r", "min"),
        pearson_max=("pearson_r", "max"),
        folds_positive_pearson=("pearson_r", lambda x: int((x > 0).sum())),
        spearman_mean=("spearman_rho", "mean"),
        spearman_median=("spearman_rho", "median"),
        spearman_min=("spearman_rho", "min"),
        spearman_max=("spearman_rho", "max"),
        folds_positive_spearman=("spearman_rho", lambda x: int((x > 0).sum()))
    )
    .reset_index()
)

corr_summary.to_csv(
    OUT_DIR / "within_fold_correlation_summary.csv",
    index=False
)

display(corr_summary.round(4))
"""))

# ------------------------------------------------------------
# CELL 8 — PAIRED RECEIVER MAE
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
mae_wide = receiver.pivot_table(
    index=["outcome", "outer_fold", "nfl_id"],
    columns="model",
    values="absolute_error"
).reset_index()

candidate_models = ["TRADITIONAL", "TRACKING", "COMBINED"]

paired_rows = []

for outcome, group in mae_wide.groupby("outcome"):
    for model in candidate_models:
        for fold, fg in group.groupby("outer_fold"):

            difference = fg[model] - fg["BASELINE"]

            paired_rows.append({
                "outcome": outcome,
                "model": model,
                "outer_fold": int(fold),
                "receivers": len(fg),
                "candidate_mae": fg[model].mean(),
                "baseline_mae": fg["BASELINE"].mean(),
                "mae_difference": difference.mean(),
                "receivers_better": int((difference < 0).sum()),
                "receivers_worse": int((difference > 0).sum())
            })

fold_differences = pd.DataFrame(paired_rows)

fold_differences.to_csv(
    OUT_DIR / "paired_receiver_mae_differences_by_fold.csv",
    index=False
)

equal_fold_summary = (
    fold_differences.groupby(["outcome", "model"])
    .agg(
        folds=("outer_fold", "nunique"),
        mean_fold_mae_difference=("mae_difference", "mean"),
        median_fold_mae_difference=("mae_difference", "median"),
        folds_model_better=("mae_difference", lambda x: int((x < 0).sum())),
        folds_model_worse=("mae_difference", lambda x: int((x > 0).sum())),
        minimum_fold_difference=("mae_difference", "min"),
        maximum_fold_difference=("mae_difference", "max")
    )
    .reset_index()
)

equal_fold_summary.to_csv(
    OUT_DIR / "equal_fold_mae_difference_summary.csv",
    index=False
)

display(fold_differences.round(5))
display(equal_fold_summary.round(5))
"""))

# ------------------------------------------------------------
# CELL 9 — STRATIFIED RECEIVER BOOTSTRAP
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
rng = np.random.default_rng(SEED)
bootstrap_rows = []

for outcome, outcome_data in mae_wide.groupby("outcome"):

    fold_arrays = {}

    for fold, fg in outcome_data.groupby("outer_fold"):
        fold_arrays[int(fold)] = {
            model: fg[model].to_numpy(float)
            for model in ["BASELINE"] + candidate_models
        }
        fold_arrays[int(fold)]["n"] = len(fg)

    for model in candidate_models:

        boot = np.empty(N_BOOT)

        for b in range(N_BOOT):

            fold_diffs = []

            for fold in sorted(fold_arrays):
                arr = fold_arrays[fold]
                idx = rng.integers(0, arr["n"], size=arr["n"])

                diff = (
                    arr[model][idx] - arr["BASELINE"][idx]
                )

                fold_diffs.append(diff.mean())

            boot[b] = np.mean(fold_diffs)

        estimate = np.mean([
            np.mean(
                fold_arrays[f][model] - fold_arrays[f]["BASELINE"]
            )
            for f in sorted(fold_arrays)
        ])

        lo, hi = np.quantile(boot, [0.025, 0.975])

        if hi < 0:
            decision = "CANDIDATE_LOWER_MAE"
        elif lo > 0:
            decision = "CANDIDATE_HIGHER_MAE"
        else:
            decision = "UNCERTAIN_CI_INCLUDES_ZERO"

        bootstrap_rows.append({
            "outcome": outcome,
            "model": model,
            "bootstrap_replicates": N_BOOT,
            "point_estimate": estimate,
            "ci_2_5": lo,
            "ci_97_5": hi,
            "probability_difference_below_zero": np.mean(boot < 0),
            "interpretation": decision
        })

bootstrap_results = pd.DataFrame(bootstrap_rows)

bootstrap_results.to_csv(
    OUT_DIR / "fold_stratified_receiver_bootstrap.csv",
    index=False
)

display(bootstrap_results.round(5))
"""))

# ------------------------------------------------------------
# CELL 10 — SENSITIVITY ANALYSIS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
pooled_rows = []

for outcome, group in mae_wide.groupby("outcome"):
    for model in candidate_models:
        difference = group[model] - group["BASELINE"]

        pooled_rows.append({
            "outcome": outcome,
            "model": model,
            "receivers": len(group),
            "receiver_weighted_mae_difference": difference.mean(),
            "receivers_better": int((difference < 0).sum()),
            "receivers_worse": int((difference > 0).sum())
        })

pooled_sensitivity = pd.DataFrame(pooled_rows)

pooled_sensitivity.to_csv(
    OUT_DIR / "receiver_weighted_mae_sensitivity.csv",
    index=False
)

PRIMARY_OUTCOMES = [
    "game_p95_speed_raw_median",
    "game_peak_speed_raw_median"
]

primary_decisions = (
    bootstrap_results[
        bootstrap_results.outcome.isin(PRIMARY_OUTCOMES)
    ]
    .merge(
        equal_fold_summary,
        on=["outcome", "model"],
        how="left"
    )
    .merge(
        pooled_sensitivity,
        on=["outcome", "model"],
        how="left"
    )
)

primary_decisions.to_csv(
    OUT_DIR / "primary_speed_outcome_decisions.csv",
    index=False
)

display(primary_decisions.round(5))
"""))

# ------------------------------------------------------------
# CELL 11 — COMBINED MODEL DIAGNOSTICS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
combined = fold_metrics[
    fold_metrics.model == "COMBINED"
].copy()

combined = combined.merge(
    fold_differences[
        fold_differences.model == "COMBINED"
    ][["outcome", "outer_fold", "mae_difference"]],
    on=["outcome", "outer_fold"],
    how="left"
)

combined.rename(
    columns={"mae_difference": "mae_difference_vs_baseline"},
    inplace=True
)

def stability_label(row):
    if pd.isna(row.pearson_r) or pd.isna(row.spearman_rho):
        return "CORRELATION_UNDEFINED"

    if row.pearson_r > 0 and row.spearman_rho > 0:
        if row.prediction_sd_ratio >= 0.20:
            return "POSITIVE_RANK_SIGNAL_WITH_VARIATION"
        return "POSITIVE_SIGNAL_BUT_COMPRESSED"

    return "WEAK_OR_INCONSISTENT_SIGNAL"

combined["diagnostic_label"] = combined.apply(
    stability_label,
    axis=1
)

combined.to_csv(
    OUT_DIR / "combined_model_fold_stability.csv",
    index=False
)

display(combined.round(4))
"""))

# ------------------------------------------------------------
# CELL 12 — VISUALISATIONS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
for outcome in PRIMARY_OUTCOMES:

    plot_data = fold_differences[
        fold_differences.outcome == outcome
    ]

    pivot = plot_data.pivot(
        index="outer_fold",
        columns="model",
        values="mae_difference"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.axhline(0, linestyle="--")

    ax.set_title(
        f"Fold-Level MAE Difference vs Baseline: {outcome}"
    )

    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Candidate MAE - Baseline MAE")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{outcome}_fold_mae_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


for outcome in PRIMARY_OUTCOMES:

    plot_data = fold_metrics[
        (fold_metrics.outcome == outcome)
        & (fold_metrics.model != "BASELINE")
    ]

    pivot = plot_data.pivot(
        index="outer_fold",
        columns="model",
        values="spearman_rho"
    )

    ax = pivot.plot(
        marker="o",
        figsize=(10, 6)
    )

    ax.axhline(0, linestyle="--")

    ax.set_title(
        f"Within-Fold Spearman Correlation: {outcome}"
    )

    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Spearman correlation")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{outcome}_within_fold_spearman.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
"""))

# ------------------------------------------------------------
# CELL 13 — RESEARCH INTERPRETATION
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
print("=" * 85)
print("NOTEBOOK 08D — RESEARCH INTERPRETATION")
print("=" * 85)

for outcome in PRIMARY_OUTCOMES:

    print("\\\\nOUTCOME:", outcome)

    subset = primary_decisions[
        primary_decisions.outcome == outcome
    ]

    for _, row in subset.iterrows():

        print("\\\\nMODEL:", row.model)

        print(
            "Equal-fold MAE difference:",
            round(row.point_estimate, 5)
        )

        print(
            "95% CI:",
            round(row.ci_2_5, 5),
            "to",
            round(row.ci_97_5, 5)
        )

        print(
            "Folds improved:",
            row.folds_model_better,
            "/ 5"
        )

        print(
            "Decision:",
            row.interpretation
        )

print("\\\\nInterpretation cautions:")
print("- Five folds contain only 10–11 receivers each.")
print("- Bootstrap intervals condition on the existing fold design.")
print("- Correlation does not establish accurate calibration.")
print("- Multiple outcome/model comparisons require caution.")
print("- No causal or external-generalisation claim is justified.")
"""))

# ------------------------------------------------------------
# CELL 14 — MANIFEST AND FINAL AUDIT
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


manifest = {
    "notebook": "08D_fold_stratified_receiver_validation.ipynb",
    "seed": SEED,
    "bootstrap_replicates": N_BOOT,
    "oof_rows": int(len(oof)),
    "receivers": int(oof.nfl_id.nunique()),
    "outer_folds": sorted(map(int, folds.outer_fold.unique())),
    "outcomes": sorted(oof.outcome.unique().tolist()),
    "models": sorted(oof.model.unique().tolist()),
    "models_refit": False,
    "fold_assignment_validation": "PASS",
    "primary_estimand": (
        "Equal-weight mean of five fold-specific "
        "receiver-level paired MAE differences."
    ),
    "bootstrap_method": (
        "Receiver resampling within original outer folds."
    ),
    "core_dataset_sha256": sha256(DATASET_PATH),
    "oof_sha256": sha256(OOF_PATH),
    "fold_assignments_sha256": sha256(FOLD_PATH)
}

with open(
    OUT_DIR / "fold_stratified_receiver_validation_manifest.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=4)

expected_outputs = [
    "receiver_level_within_fold_predictions.csv",
    "fold_stratified_receiver_metrics.csv",
    "within_fold_correlation_summary.csv",
    "paired_receiver_mae_differences_by_fold.csv",
    "equal_fold_mae_difference_summary.csv",
    "fold_stratified_receiver_bootstrap.csv",
    "receiver_weighted_mae_sensitivity.csv",
    "primary_speed_outcome_decisions.csv",
    "combined_model_fold_stability.csv",
    "fold_stratified_receiver_validation_manifest.json"
]

missing = [
    name for name in expected_outputs
    if not (OUT_DIR / name).exists()
]

assert not missing, f"Missing outputs: {missing}"

print("=" * 80)
print("NOTEBOOK 08D COMPLETE")
print("=" * 80)
print("OOF rows:", len(oof))
print("Receivers:", oof.nfl_id.nunique())
print("Folds:", folds.outer_fold.nunique())
print("Bootstrap:", N_BOOT)
print("Missing outputs:", missing)
print("Output:", OUT_DIR)
"""))

# ------------------------------------------------------------
# CREATE NOTEBOOK
# ------------------------------------------------------------

nb.cells = cells

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

nbf.validate(nb)

nbf.write(nb, OUTPUT)

print("=" * 75)
print("NOTEBOOK 08D GENERATED SUCCESSFULLY")
print("=" * 75)
print("Notebook:", OUTPUT)
print("Cells:", len(nb.cells))
print("Size:", OUTPUT.stat().st_size, "bytes")
