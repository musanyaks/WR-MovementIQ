
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

notebook = nbf.v4.new_notebook()

notebook.cells = [
    nbf.v4.new_markdown_cell("""
# Notebook 10 — Robustness and Sensitivity Analysis

## Research objective
Test whether the predictive findings from the nested Ridge experiment are sensitive to the representation of Combine tracking features.

## Validation design
- Frozen core dataset: 93 player-seasons, 54 receivers.
- Original five outer GroupKFold assignments from Notebook 07.
- Receiver-grouped inner cross-validation.
- Training-only StandardScaler.
- Ridge alpha tuned within each outer training fold.
- No imputation.
- Receiver-level bootstrap stratified by outer fold.

All comparisons use identical observations within each outcome. The baseline is a training-fold median predictor.
"""),

    nbf.v4.new_code_cell(r'''
from pathlib import Path
import json
import warnings
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Resolve project root from the notebook execution directory.
CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
]

ROOT = None

for candidate in CANDIDATES:
    if (
        (candidate / "data").is_dir()
        and (candidate / "reports").is_dir()
        and (candidate / "notebooks").is_dir()
    ):
        ROOT = candidate.resolve()
        break

if ROOT is None:
    raise FileNotFoundError(
        "WR-MovementIQ project root could not be located. "
        f"Current directory: {Path.cwd()}"
    )

print("Project root:", ROOT)
DATA_PATH = ROOT / "data/features/wr_analysis_dataset_core_common.parquet"
FOLD_PATH = ROOT / "reports/modeling_design/outer_grouped_fold_assignments.csv"
OUT_DIR = ROOT / "reports/robustness_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2027
BOOTSTRAP_REPS = 5000
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

OUTCOMES = [
    "game_mean_speed_raw_median",
    "game_peak_speed_raw_median",
    "game_p95_speed_raw_median",
    "game_mean_acceleration_raw_median",
    "game_peak_acceleration_raw_median",
    "game_path_efficiency_median",
]

TRADITIONAL = ["forty", "vertical", "broad_jump"]

TRACKING = [
    "combine_mean_speed_median_across_drills",
    "combine_peak_speed_median_across_drills",
    "combine_p95_speed_median_across_drills",
    "combine_mean_acceleration_median_across_drills",
    "combine_peak_acceleration_median_across_drills",
    "combine_path_efficiency_median_across_drills",
]

SPEED_TRACKING = [
    "combine_mean_speed_median_across_drills",
    "combine_peak_speed_median_across_drills",
    "combine_p95_speed_median_across_drills",
]

SPECS = {
    "BASELINE": [],
    "TRADITIONAL": TRADITIONAL,
    "TRACKING_FULL": TRACKING,
    "COMBINED_FULL": TRADITIONAL + TRACKING,
    "COMBINED_NO_PEAK_SPEED": [
        x for x in TRADITIONAL + TRACKING
        if x != "combine_peak_speed_median_across_drills"
    ],
    "COMBINED_NO_P95_SPEED": [
        x for x in TRADITIONAL + TRACKING
        if x != "combine_p95_speed_median_across_drills"
    ],
    "COMBINED_PEAK_SPEED_ONLY": TRADITIONAL + [
        "combine_peak_speed_median_across_drills"
    ],
    "COMBINED_P95_SPEED_ONLY": TRADITIONAL + [
        "combine_p95_speed_median_across_drills"
    ],
    "COMBINED_NO_ACCELERATION": [
        x for x in TRADITIONAL + TRACKING
        if "acceleration" not in x
    ],
    "TRACKING_SPEED_ONLY": SPEED_TRACKING,
}

assert DATA_PATH.exists(), DATA_PATH
assert FOLD_PATH.exists(), FOLD_PATH

df = pd.read_parquet(DATA_PATH)
folds = pd.read_csv(FOLD_PATH)

print("Dataset shape:", df.shape)
print("Fold assignment shape:", folds.shape)

required = ["nfl_id", *OUTCOMES, *TRADITIONAL, *TRACKING]
missing = sorted(set(required) - set(df.columns))
assert not missing, f"Missing dataset columns: {missing}"

assert "nfl_id" in folds.columns, "Fold file missing nfl_id"
assert "outer_fold" in folds.columns, "Fold file missing outer_fold"

df["nfl_id"] = df["nfl_id"].astype(str)
folds["nfl_id"] = folds["nfl_id"].astype(str)

fold_map = folds[["nfl_id", "outer_fold"]].drop_duplicates()

assert fold_map["nfl_id"].is_unique, (
    "A receiver appears in multiple outer folds."
)

df = df.merge(
    fold_map,
    on="nfl_id",
    how="inner",
    validate="many_to_one"
)

assert df["outer_fold"].notna().all()
assert df.duplicated(["draft_year", "nfl_id", "season"]).sum() == 0

for col in OUTCOMES + TRADITIONAL + TRACKING:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Analysis shape:", df.shape)
print("Receivers:", df["nfl_id"].nunique())
print("Outer folds:", sorted(df["outer_fold"].unique()))

expected_receivers = 54
expected_rows = 93

assert df["nfl_id"].nunique() == expected_receivers
assert len(df) == expected_rows

dataset_hash = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

print("Core dataset SHA256:", dataset_hash)
'''),

    nbf.v4.new_markdown_cell("""
## 1. Model specifications and sample validation

Each model is evaluated on the same complete observations for a given outcome. This avoids attributing changes in sample composition to changes in predictors.
"""),

    nbf.v4.new_code_cell(r'''
# Check complete-case sample sizes for every outcome and model.
sample_audit = []

for outcome in OUTCOMES:
    for model, features in SPECS.items():
        cols = [outcome] + features
        subset = df.dropna(subset=cols)

        sample_audit.append({
            "outcome": outcome,
            "model": model,
            "rows": len(subset),
            "receivers": subset["nfl_id"].nunique(),
            "folds": subset["outer_fold"].nunique()
        })

sample_audit = pd.DataFrame(sample_audit)
display(sample_audit)

assert sample_audit["rows"].eq(len(df)).all(), (
    "At least one model specification has a different sample. "
    "Investigate missingness before proceeding."
)
'''),

    nbf.v4.new_markdown_cell("""
## 2. Nested grouped validation

The outer folds are fixed. Within each outer training sample, alpha is selected using GroupKFold by receiver. Scaling is fitted independently inside each inner training fold.
"""),

    nbf.v4.new_code_cell(r'''
prediction_records = []
tuning_records = []
fold_metric_records = []

outer_folds = sorted(df["outer_fold"].unique())

for outcome in OUTCOMES:
    print("\nOUTCOME:", outcome)

    for model_name, features in SPECS.items():

        for outer_fold in outer_folds:

            train = df[df["outer_fold"] != outer_fold].copy()
            test = df[df["outer_fold"] == outer_fold].copy()

            if features:
                X_train = train[features].to_numpy(dtype=float)
                X_test = test[features].to_numpy(dtype=float)
            else:
                X_train = np.zeros((len(train), 1))
                X_test = np.zeros((len(test), 1))

            y_train = train[outcome].to_numpy(dtype=float)
            y_test = test[outcome].to_numpy(dtype=float)

            groups = train["nfl_id"].to_numpy()

            assert train["nfl_id"].nunique() >= 2
            assert set(train["nfl_id"]).isdisjoint(set(test["nfl_id"]))

            if model_name == "BASELINE":
                model = DummyRegressor(strategy="median")
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                selected_alpha = np.nan
                inner_best_mae = np.nan

            else:
                n_splits = min(4, len(np.unique(groups)))
                inner_cv = GroupKFold(n_splits=n_splits)

                alpha_scores = []

                for alpha in ALPHAS:
                    inner_errors = []

                    for inner_train_idx, inner_valid_idx in inner_cv.split(
                        X_train, y_train, groups
                    ):
                        X_it = X_train[inner_train_idx]
                        X_iv = X_train[inner_valid_idx]
                        y_it = y_train[inner_train_idx]
                        y_iv = y_train[inner_valid_idx]

                        pipeline = Pipeline([
                            ("scale", StandardScaler()),
                            ("ridge", Ridge(alpha=alpha))
                        ])

                        pipeline.fit(X_it, y_it)
                        pred = pipeline.predict(X_iv)

                        inner_errors.append(
                            mean_absolute_error(y_iv, pred)
                        )

                    score = float(np.mean(inner_errors))
                    alpha_scores.append((alpha, score))

                selected_alpha, inner_best_mae = min(
                    alpha_scores, key=lambda item: item[1]
                )

                for alpha, score in alpha_scores:
                    tuning_records.append({
                        "outcome": outcome,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "alpha": alpha,
                        "inner_mean_mae": score
                    })

                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=selected_alpha))
                ])

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            for i, (_, row) in enumerate(test.iterrows()):
                prediction_records.append({
                    "outcome": outcome,
                    "model": model_name,
                    "nfl_id": row["nfl_id"],
                    "season": row["season"],
                    "draft_year": row["draft_year"],
                    "outer_fold": outer_fold,
                    "actual": y_test[i],
                    "predicted": predictions[i],
                    "absolute_error": abs(y_test[i] - predictions[i]),
                    "best_alpha": selected_alpha
                })

            fold_metric_records.append({
                "outcome": outcome,
                "model": model_name,
                "outer_fold": outer_fold,
                "n_train": len(train),
                "n_test": len(test),
                "train_receivers": train["nfl_id"].nunique(),
                "test_receivers": test["nfl_id"].nunique(),
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions)
                    if len(y_test) > 1 else np.nan,
                "best_alpha": selected_alpha,
                "inner_best_mae": inner_best_mae
            })

predictions_df = pd.DataFrame(prediction_records)
tuning_df = pd.DataFrame(tuning_records)
fold_metrics = pd.DataFrame(fold_metric_records)

print("Prediction rows:", len(predictions_df))
print("Expected prediction rows:", len(df) * len(OUTCOMES) * len(SPECS))

assert len(predictions_df) == len(df) * len(OUTCOMES) * len(SPECS)
assert not predictions_df.duplicated(
    ["outcome", "model", "nfl_id", "season"]
).any()

predictions_df.to_csv(
    OUT_DIR / "sensitivity_oof_predictions.csv", index=False
)
tuning_df.to_csv(
    OUT_DIR / "sensitivity_inner_tuning.csv", index=False
)
fold_metrics.to_csv(
    OUT_DIR / "sensitivity_fold_metrics.csv", index=False
)
'''),

    nbf.v4.new_markdown_cell("""
## 3. Pooled out-of-fold performance

These metrics summarize predictions from the fixed outer folds. They are descriptive and should be interpreted alongside fold-level results and receiver-level uncertainty.
"""),

    nbf.v4.new_code_cell(r'''
pooled_records = []

for (outcome, model), group in predictions_df.groupby(
    ["outcome", "model"]
):
    actual = group["actual"].to_numpy()
    predicted = group["predicted"].to_numpy()

    pooled_records.append({
        "outcome": outcome,
        "model": model,
        "observations": len(group),
        "receivers": group["nfl_id"].nunique(),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
        "pearson": (
            np.corrcoef(actual, predicted)[0, 1]
            if np.std(predicted) > 0 and np.std(actual) > 0
            else np.nan
        )
    })

pooled_metrics = pd.DataFrame(pooled_records)
pooled_metrics.to_csv(
    OUT_DIR / "sensitivity_pooled_metrics.csv", index=False
)

display(
    pooled_metrics.sort_values(["outcome", "mae"])
)
'''),

    nbf.v4.new_markdown_cell("""
## 4. Receiver-level, fold-stratified bootstrap

Absolute errors are first averaged within receiver and model. Bootstrap resampling then occurs within each original outer fold. The reported estimand is the equal-weight average of the five fold-specific receiver-level MAE differences.

A negative difference means the sensitivity model has lower MAE than the median baseline.
"""),

    nbf.v4.new_code_cell(r'''
receiver_errors = (
    predictions_df
    .groupby(["outcome", "model", "outer_fold", "nfl_id"], as_index=False)
    ["absolute_error"].mean()
)

bootstrap_records = []
rng = np.random.default_rng(SEED)

for outcome in OUTCOMES:

    outcome_data = receiver_errors[
        receiver_errors["outcome"] == outcome
    ]

    wide = outcome_data.pivot_table(
        index=["outer_fold", "nfl_id"],
        columns="model",
        values="absolute_error"
    ).reset_index()

    assert "BASELINE" in wide.columns

    for model_name in SPECS:
        if model_name == "BASELINE":
            continue

        if model_name not in wide.columns:
            continue

        differences_by_fold = {}

        for fold in outer_folds:
            fold_data = wide[wide["outer_fold"] == fold].dropna(
                subset=["BASELINE", model_name]
            )

            differences_by_fold[fold] = (
                fold_data[model_name].to_numpy()
                - fold_data["BASELINE"].to_numpy()
            )

        fold_means = {
            fold: float(np.mean(values))
            for fold, values in differences_by_fold.items()
        }

        observed = float(np.mean(list(fold_means.values())))

        boot = np.empty(BOOTSTRAP_REPS)

        for b in range(BOOTSTRAP_REPS):
            sampled_fold_means = []

            for fold in outer_folds:
                values = differences_by_fold[fold]
                sampled = rng.choice(
                    values, size=len(values), replace=True
                )
                sampled_fold_means.append(np.mean(sampled))

            boot[b] = np.mean(sampled_fold_means)

        lower, upper = np.quantile(boot, [0.025, 0.975])

        bootstrap_records.append({
            "outcome": outcome,
            "model": model_name,
            "comparison": "MODEL_MINUS_BASELINE",
            "mean_mae_difference": observed,
            "ci_lower": lower,
            "ci_upper": upper,
            "probability_below_zero": float(np.mean(boot < 0)),
            "folds": len(outer_folds),
            "receivers": wide["nfl_id"].nunique(),
            "bootstrap_reps": BOOTSTRAP_REPS
        })

bootstrap_df = pd.DataFrame(bootstrap_records)
bootstrap_df.to_csv(
    OUT_DIR / "sensitivity_receiver_bootstrap.csv", index=False
)

display(bootstrap_df)
'''),

    nbf.v4.new_markdown_cell("""
## 5. Direct comparison of peak-speed feature representations

This comparison isolates the highly correlated peak-speed and P95-speed predictors. It evaluates their separate and joint use in the combined specification.
"""),

    nbf.v4.new_code_cell(r'''
speed_models = [
    "COMBINED_FULL",
    "COMBINED_NO_PEAK_SPEED",
    "COMBINED_NO_P95_SPEED",
    "COMBINED_PEAK_SPEED_ONLY",
    "COMBINED_P95_SPEED_ONLY",
]

speed_comparison = pooled_metrics[
    pooled_metrics["model"].isin(speed_models)
    & pooled_metrics["outcome"].isin([
        "game_mean_speed_raw_median",
        "game_p95_speed_raw_median",
        "game_peak_speed_raw_median"
    ])
].copy()

speed_comparison.to_csv(
    OUT_DIR / "speed_feature_sensitivity.csv", index=False
)

display(
    speed_comparison.sort_values(["outcome", "mae"])
)
'''),

    nbf.v4.new_markdown_cell("""
## 6. Fold-level consistency

A model that improves pooled MAE because of one unusually favorable fold should not be described as consistently superior.
"""),

    nbf.v4.new_code_cell(r'''
fold_comparison = fold_metrics.merge(
    fold_metrics[fold_metrics["model"] == "BASELINE"][
        ["outcome", "outer_fold", "mae"]
    ].rename(columns={"mae": "baseline_mae"}),
    on=["outcome", "outer_fold"],
    how="left",
    validate="many_to_one"
)

fold_comparison["mae_difference_vs_baseline"] = (
    fold_comparison["mae"] - fold_comparison["baseline_mae"]
)

fold_comparison.to_csv(
    OUT_DIR / "sensitivity_fold_comparison.csv", index=False
)

fold_stability = (
    fold_comparison[fold_comparison["model"] != "BASELINE"]
    .groupby(["outcome", "model"])
    .agg(
        mean_mae_difference=("mae_difference_vs_baseline", "mean"),
        median_mae_difference=("mae_difference_vs_baseline", "median"),
        folds_better=("mae_difference_vs_baseline", lambda x: int((x < 0).sum())),
        folds_worse=("mae_difference_vs_baseline", lambda x: int((x > 0).sum())),
        minimum_difference=("mae_difference_vs_baseline", "min"),
        maximum_difference=("mae_difference_vs_baseline", "max")
    )
    .reset_index()
)

fold_stability.to_csv(
    OUT_DIR / "sensitivity_fold_stability.csv", index=False
)

display(fold_stability)
'''),

    nbf.v4.new_markdown_cell("""
## 7. Ridge alpha stability

Repeated selection of the largest alpha indicates that the model prefers strong shrinkage within the tested grid. It is not evidence that the largest alpha is the true optimum outside the grid.
"""),

    nbf.v4.new_code_cell(r'''
alpha_summary = (
    tuning_df
    .groupby(["outcome", "model"])
    .agg(
        folds_tuned=("outer_fold", "nunique"),
        median_selected_alpha=("alpha", "median"),
        minimum_selected_alpha=("alpha", "min"),
        maximum_selected_alpha=("alpha", "max")
    )
    .reset_index()
)

selected_alphas = (
    fold_metrics[fold_metrics["model"] != "BASELINE"]
    [["outcome", "model", "outer_fold", "best_alpha"]]
)

boundary = (
    selected_alphas
    .assign(at_upper_boundary=lambda x: x["best_alpha"] == max(ALPHAS))
    .groupby(["outcome", "model"])
    .agg(
        selected_folds=("outer_fold", "nunique"),
        upper_boundary_folds=("at_upper_boundary", "sum")
    )
    .reset_index()
)

alpha_summary = alpha_summary.merge(
    boundary, on=["outcome", "model"], how="left"
)

alpha_summary["upper_boundary_pct"] = (
    100 * alpha_summary["upper_boundary_folds"]
    / alpha_summary["selected_folds"]
)

alpha_summary.to_csv(
    OUT_DIR / "sensitivity_alpha_summary.csv", index=False
)

display(alpha_summary)
'''),

    nbf.v4.new_markdown_cell("""
## 8. Automated integrity checks and manifest

This notebook does not overwrite Notebook 08 or its original predictions. It creates an independent sensitivity-analysis output directory.
"""),

    nbf.v4.new_code_cell(r'''
# Confirm all models have the same number of observations for each outcome.
coverage = (
    predictions_df
    .groupby(["outcome", "model"])
    .agg(
        rows=("nfl_id", "size"),
        receivers=("nfl_id", "nunique"),
        folds=("outer_fold", "nunique")
    )
    .reset_index()
)

assert coverage["rows"].eq(len(df)).all()
assert coverage["receivers"].eq(df["nfl_id"].nunique()).all()
assert coverage["folds"].eq(len(outer_folds)).all()

# Verify no receiver crosses outer folds.
receiver_fold_counts = (
    predictions_df.groupby("nfl_id")["outer_fold"].nunique()
)
assert receiver_fold_counts.max() == 1

expected_files = [
    "sensitivity_oof_predictions.csv",
    "sensitivity_inner_tuning.csv",
    "sensitivity_fold_metrics.csv",
    "sensitivity_pooled_metrics.csv",
    "sensitivity_receiver_bootstrap.csv",
    "speed_feature_sensitivity.csv",
    "sensitivity_fold_comparison.csv",
    "sensitivity_fold_stability.csv",
    "sensitivity_alpha_summary.csv",
]

manifest = {
    "notebook": "10_robustness_sensitivity_analysis.ipynb",
    "dataset": str(DATA_PATH.relative_to(ROOT)),
    "dataset_sha256": dataset_hash,
    "rows": int(len(df)),
    "receivers": int(df["nfl_id"].nunique()),
    "outer_folds": [int(x) for x in outer_folds],
    "outcomes": OUTCOMES,
    "model_specifications": SPECS,
    "alpha_grid": ALPHAS,
    "bootstrap_reps": BOOTSTRAP_REPS,
    "bootstrap_unit": "receiver, stratified by outer fold",
    "primary_estimand": "equal-fold receiver-level MAE difference vs median baseline",
    "missing_outputs": []
}

for filename in expected_files:
    if not (OUT_DIR / filename).exists():
        manifest["missing_outputs"].append(filename)

with open(OUT_DIR / "sensitivity_analysis_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

assert manifest["missing_outputs"] == []

print("=" * 75)
print("NOTEBOOK 10 COMPLETE")
print("=" * 75)
print("Output:", OUT_DIR)
print("Rows:", len(df))
print("Receivers:", df["nfl_id"].nunique())
print("Models:", len(SPECS))
print("Outcomes:", len(OUTCOMES))
print("OOF prediction rows:", len(predictions_df))
print("Missing outputs:", manifest["missing_outputs"])
print("Dataset SHA256:", dataset_hash)
''')
]

notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python"}
}

output_path = NOTEBOOK_DIR / "10_robustness_sensitivity_analysis.ipynb"

nbf.write(notebook, output_path)

print("=" * 70)
print("NOTEBOOK 10 GENERATED")
print("=" * 70)
print(output_path)
print("Cells:", len(notebook.cells))
