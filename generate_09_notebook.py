
from pathlib import Path
import nbformat as nbf

# ============================================================
# WR-MovementIQ — NOTEBOOK 09 GENERATOR
# Predictor Interpretation and Feature Stability
# ============================================================

ROOT = Path(r"D:\PROJECTS\WR-MovementIQ")
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = NOTEBOOK_DIR / "09_predictor_interpretation_and_feature_stability.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# ------------------------------------------------------------
# CELL 1 — TITLE
# ------------------------------------------------------------

cells.append(nbf.v4.new_markdown_cell("""
# Notebook 09 — Predictor Interpretation and Feature Stability

**Project:** WR-MovementIQ  
**Research:** Beyond the 40-Yard Dash

## Objectives

1. Examine traditional Combine predictor relationships.
2. Examine tracking-derived Combine predictor relationships.
3. Investigate multicollinearity.
4. Estimate fold-specific standardized Ridge coefficients.
5. Evaluate coefficient sign stability.
6. Investigate associations with NFL P95 and peak speed.
7. Identify predictors requiring further robustness analysis.

## Research safeguards

- Preserve the frozen core analysis dataset.
- Preserve Notebook 07 outer folds.
- Use selected alpha values from Notebook 08.
- Fit scaling using training data only.
- Do not interpret coefficients as causal effects.
- Do not modify existing OOF predictions.
"""))

# ------------------------------------------------------------
# CELL 2 — IMPORTS AND PATHS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

# Resolve project root
HERE = Path.cwd().resolve()

if (HERE / "data").exists() and (HERE / "reports").exists():
    ROOT = HERE
elif (HERE.parent / "data").exists() and (HERE.parent / "reports").exists():
    ROOT = HERE.parent
else:
    raise FileNotFoundError(
        f"Cannot locate project root from {HERE}"
    )

DATASET_PATH = (
    ROOT / "data/features/wr_analysis_dataset_core_common.parquet"
)

FOLD_PATH = (
    ROOT / "reports/modeling_design/outer_grouped_fold_assignments.csv"
)

OOF_PATH = (
    ROOT / "reports/nested_ridge_modeling/oof_predictions.csv"
)

TUNING_PATH = (
    ROOT / "reports/nested_ridge_modeling/inner_tuning_results.csv"
)

OUT_DIR = ROOT / "reports/predictor_interpretation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", ROOT)
print("Dataset:", DATASET_PATH)
print("Output:", OUT_DIR)
"""))

# ------------------------------------------------------------
# CELL 3 — LOAD AND VALIDATE DATA
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
for path in [DATASET_PATH, FOLD_PATH, OOF_PATH, TUNING_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

data = pd.read_parquet(DATASET_PATH)
folds_raw = pd.read_csv(FOLD_PATH)
oof = pd.read_csv(OOF_PATH)
tuning = pd.read_csv(TUNING_PATH)

data["nfl_id"] = pd.to_numeric(
    data["nfl_id"], errors="raise"
).astype("int64")

folds_raw["nfl_id"] = pd.to_numeric(
    folds_raw["nfl_id"], errors="raise"
).astype("int64")

folds_raw["outer_fold"] = pd.to_numeric(
    folds_raw["outer_fold"], errors="raise"
).astype(int)

folds = (
    folds_raw[["nfl_id", "outer_fold"]]
    .drop_duplicates()
    .copy()
)

assert folds.groupby("nfl_id")["outer_fold"].nunique().eq(1).all()
assert folds["nfl_id"].is_unique

assert len(data) == 93
assert data["nfl_id"].nunique() == 54
assert folds["nfl_id"].nunique() == 54
assert folds["outer_fold"].nunique() == 5

print("Dataset rows:", len(data))
print("Receivers:", data.nfl_id.nunique())
print("Outer folds:", sorted(folds.outer_fold.unique()))
print("OOF rows:", len(oof))
print("Tuning rows:", len(tuning))

display(folds.groupby("outer_fold").nfl_id.nunique())
"""))

# ------------------------------------------------------------
# CELL 4 — MODEL SPECIFICATIONS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
TRADITIONAL = [
    "forty",
    "vertical",
    "broad_jump"
]

TRACKING = [
    "combine_mean_speed_median_across_drills",
    "combine_peak_speed_median_across_drills",
    "combine_p95_speed_median_across_drills",
    "combine_mean_acceleration_median_across_drills",
    "combine_peak_acceleration_median_across_drills",
    "combine_path_efficiency_median_across_drills"
]

FEATURE_SETS = {
    "TRADITIONAL": TRADITIONAL,
    "TRACKING": TRACKING,
    "COMBINED": TRADITIONAL + TRACKING
}

OUTCOMES = [
    "game_mean_speed_raw_median",
    "game_p95_speed_raw_median",
    "game_peak_speed_raw_median",
    "game_mean_acceleration_raw_median",
    "game_peak_acceleration_raw_median",
    "game_path_efficiency_median"
]

PRIMARY_SPEED = [
    "game_p95_speed_raw_median",
    "game_peak_speed_raw_median"
]

required = (
    TRADITIONAL + TRACKING + OUTCOMES
)

missing = [
    c for c in required
    if c not in data.columns
]

assert not missing, f"Missing columns: {missing}"

print("Traditional predictors:", len(TRADITIONAL))
print("Tracking predictors:", len(TRACKING))
print("Combined predictors:", len(TRADITIONAL + TRACKING))
"""))

# ------------------------------------------------------------
# CELL 5 — PREDICTOR DISTRIBUTIONS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
rows = []

for feature in TRADITIONAL + TRACKING:

    x = pd.to_numeric(data[feature], errors="coerce")

    rows.append({
        "predictor": feature,
        "group": (
            "TRADITIONAL"
            if feature in TRADITIONAL
            else "TRACKING"
        ),
        "observations": x.notna().sum(),
        "missing_pct": x.isna().mean() * 100,
        "mean": x.mean(),
        "std": x.std(),
        "median": x.median(),
        "minimum": x.min(),
        "maximum": x.max()
    })

predictor_summary = pd.DataFrame(rows)

predictor_summary.to_csv(
    OUT_DIR / "predictor_distribution_summary.csv",
    index=False
)

display(predictor_summary)
"""))

# ------------------------------------------------------------
# CELL 6 — MULTICOLLINEARITY
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
predictors = TRADITIONAL + TRACKING

correlation = data[predictors].corr(
    method="spearman"
)

correlation.to_csv(
    OUT_DIR / "predictor_spearman_correlation_matrix.csv"
)

pairs = []

for i, first in enumerate(predictors):

    for second in predictors[i + 1:]:

        rho = correlation.loc[first, second]

        pairs.append({
            "predictor_a": first,
            "predictor_b": second,
            "spearman_rho": rho,
            "absolute_correlation": abs(rho)
        })

correlation_pairs = (
    pd.DataFrame(pairs)
    .sort_values("absolute_correlation", ascending=False)
)

correlation_pairs.to_csv(
    OUT_DIR / "predictor_correlation_pairs.csv",
    index=False
)

display(correlation_pairs)

fig, ax = plt.subplots(figsize=(12, 10))

image = ax.imshow(
    correlation.values,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(range(len(predictors)))
ax.set_yticks(range(len(predictors)))

ax.set_xticklabels(predictors, rotation=90)
ax.set_yticklabels(predictors)

fig.colorbar(image, ax=ax)

ax.set_title("Combine Predictor Spearman Correlations")

fig.tight_layout()

fig.savefig(
    OUT_DIR / "predictor_correlation_heatmap.png",
    dpi=180
)

plt.show()
"""))

# ------------------------------------------------------------
# CELL 7 — UNIVARIATE OUTCOME ASSOCIATIONS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
association_rows = []

for outcome in OUTCOMES:

    for predictor in predictors:

        pair = data[
            [outcome, predictor]
        ].apply(
            pd.to_numeric,
            errors="coerce"
        ).dropna()

        rho = (
            pair[outcome].corr(
                pair[predictor],
                method="spearman"
            )
            if len(pair) >= 5
            else np.nan
        )

        association_rows.append({
            "outcome": outcome,
            "predictor": predictor,
            "group": (
                "TRADITIONAL"
                if predictor in TRADITIONAL
                else "TRACKING"
            ),
            "n": len(pair),
            "spearman_rho": rho
        })

associations = pd.DataFrame(association_rows)

associations.to_csv(
    OUT_DIR / "univariate_spearman_associations.csv",
    index=False
)

display(
    associations[
        associations.outcome.isin(PRIMARY_SPEED)
    ].sort_values(
        ["outcome", "spearman_rho"],
        ascending=[True, False]
    )
)
"""))

# ------------------------------------------------------------
# CELL 8 — INSPECT TUNING SCHEMA
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
print("TUNING FILE COLUMNS")
print(tuning.columns.tolist())

display(tuning.head(15))

# Identify columns without assuming a fixed schema
lower_cols = {
    c.lower(): c
    for c in tuning.columns
}

def find_column(candidates):

    for candidate in candidates:

        if candidate in lower_cols:
            return lower_cols[candidate]

    return None

OUTCOME_COL = find_column([
    "outcome",
    "target"
])

MODEL_COL = find_column([
    "model",
    "specification",
    "model_name"
])

FOLD_COL = find_column([
    "outer_fold",
    "fold"
])

ALPHA_COL = find_column([
    "best_alpha",
    "selected_alpha",
    "alpha"
])

SCORE_COL = find_column([
    "inner_best_mae",
    "mean_validation_mae",
    "validation_mae",
    "mean_mae"
])

print({
    "outcome": OUTCOME_COL,
    "model": MODEL_COL,
    "fold": FOLD_COL,
    "alpha": ALPHA_COL,
    "score": SCORE_COL
})

if not all([
    OUTCOME_COL,
    MODEL_COL,
    FOLD_COL,
    ALPHA_COL
]):
    raise ValueError(
        "Cannot identify the selected-alpha schema. "
        "Inspect the tuning columns printed above before continuing."
    )
"""))

# ------------------------------------------------------------
# CELL 9 — SELECTED ALPHA TABLE
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
t = tuning.copy()

t[FOLD_COL] = pd.to_numeric(
    t[FOLD_COL], errors="coerce"
)

t[ALPHA_COL] = pd.to_numeric(
    t[ALPHA_COL], errors="coerce"
)

t = t.dropna(
    subset=[
        OUTCOME_COL,
        MODEL_COL,
        FOLD_COL,
        ALPHA_COL
    ]
)

t[FOLD_COL] = t[FOLD_COL].astype(int)

KEYS = [
    OUTCOME_COL,
    MODEL_COL,
    FOLD_COL
]

if SCORE_COL:

    t[SCORE_COL] = pd.to_numeric(
        t[SCORE_COL],
        errors="coerce"
    )

    selected = (
        t.sort_values(SCORE_COL)
        .drop_duplicates(KEYS, keep="first")
    )

else:

    selected = t.drop_duplicates(
        KEYS,
        keep="last"
    )

alpha_lookup = {}

for _, row in selected.iterrows():

    key = (
        str(row[OUTCOME_COL]),
        str(row[MODEL_COL]).upper(),
        int(row[FOLD_COL])
    )

    alpha_lookup[key] = float(
        row[ALPHA_COL]
    )

print("Selected alpha combinations:", len(alpha_lookup))

print("Model labels:")
print(sorted(set(k[1] for k in alpha_lookup)))

print("Outcome labels:")
print(sorted(set(k[0] for k in alpha_lookup)))
"""))

# ------------------------------------------------------------
# CELL 10 — FOLD-SPECIFIC RIDGE COEFFICIENTS
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
coefficient_rows = []

for outcome in OUTCOMES:

    for model_name, features in FEATURE_SETS.items():

        for fold_id in sorted(folds.outer_fold.unique()):

            key = (
                outcome,
                model_name,
                int(fold_id)
            )

            if key not in alpha_lookup:
                continue

            alpha = alpha_lookup[key]

            train_receivers = set(
                folds.loc[
                    folds.outer_fold != fold_id,
                    "nfl_id"
                ]
            )

            train = data[
                data.nfl_id.isin(train_receivers)
            ].copy()

            X = train[features].apply(
                pd.to_numeric,
                errors="coerce"
            )

            y = pd.to_numeric(
                train[outcome],
                errors="coerce"
            )

            valid = (
                y.notna()
                & X.notna().all(axis=1)
            )

            X = X.loc[valid]
            y = y.loc[valid]

            if len(X) < 10:
                continue

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha))
            ])

            pipeline.fit(X, y)

            model = pipeline.named_steps["ridge"]

            for feature, coefficient in zip(
                features,
                model.coef_
            ):

                coefficient_rows.append({
                    "outcome": outcome,
                    "model": model_name,
                    "outer_fold": int(fold_id),
                    "predictor": feature,
                    "standardized_coefficient": float(coefficient),
                    "selected_alpha": alpha,
                    "training_rows": len(X),
                    "training_receivers": train.loc[
                        X.index,
                        "nfl_id"
                    ].nunique()
                })

coefficients = pd.DataFrame(coefficient_rows)

if coefficients.empty:
    raise RuntimeError(
        "No coefficients generated. Check tuning model labels "
        "and selected alpha records."
    )

coefficients.to_csv(
    OUT_DIR / "fold_specific_standardized_coefficients.csv",
    index=False
)

display(coefficients.head(20))

print("Coefficient records:", len(coefficients))
"""))

# ------------------------------------------------------------
# CELL 11 — COEFFICIENT STABILITY
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
stability = (
    coefficients
    .groupby([
        "outcome",
        "model",
        "predictor"
    ])
    .agg(
        folds=("outer_fold", "nunique"),
        mean_coefficient=(
            "standardized_coefficient",
            "mean"
        ),
        median_coefficient=(
            "standardized_coefficient",
            "median"
        ),
        coefficient_sd=(
            "standardized_coefficient",
            "std"
        ),
        minimum=(
            "standardized_coefficient",
            "min"
        ),
        maximum=(
            "standardized_coefficient",
            "max"
        ),
        positive_folds=(
            "standardized_coefficient",
            lambda x: int((x > 0).sum())
        ),
        negative_folds=(
            "standardized_coefficient",
            lambda x: int((x < 0).sum())
        )
    )
    .reset_index()
)

stability["sign_consistency"] = (
    stability[
        ["positive_folds", "negative_folds"]
    ].max(axis=1)
    / stability["folds"]
)

stability.to_csv(
    OUT_DIR / "coefficient_stability_summary.csv",
    index=False
)

display(
    stability[
        stability.outcome.isin(PRIMARY_SPEED)
    ].sort_values(
        [
            "outcome",
            "model",
            "sign_consistency"
        ],
        ascending=[True, True, False]
    )
)
"""))

# ------------------------------------------------------------
# CELL 12 — COEFFICIENT VISUALISATION
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
for outcome in PRIMARY_SPEED:

    subset = coefficients[
        (coefficients.outcome == outcome)
        & (coefficients.model == "COMBINED")
    ]

    if subset.empty:
        print("No combined coefficients:", outcome)
        continue

    pivot = subset.pivot(
        index="predictor",
        columns="outer_fold",
        values="standardized_coefficient"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 7)
    )

    ax.axhline(0, linewidth=1)

    ax.set_title(
        f"Combined Ridge Coefficients Across Folds: {outcome}"
    )

    ax.set_ylabel("Standardized coefficient")
    ax.set_xlabel("")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{outcome}_coefficient_stability.png",
        dpi=180
    )

    plt.show()
"""))

# ------------------------------------------------------------
# CELL 13 — EXISTING OOF PERFORMANCE
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
oof["actual"] = pd.to_numeric(
    oof["actual"],
    errors="coerce"
)

oof["predicted"] = pd.to_numeric(
    oof["predicted"],
    errors="coerce"
)

oof = oof.dropna(
    subset=["actual", "predicted"]
)

oof_rows = []

for (outcome, model), group in oof.groupby(
    ["outcome", "model"]
):

    actual = group["actual"].to_numpy()
    predicted = group["predicted"].to_numpy()

    oof_rows.append({
        "outcome": outcome,
        "model": model,
        "predictions": len(group),
        "receivers": group.nfl_id.nunique(),
        "mae": np.mean(np.abs(actual - predicted)),
        "rmse": np.sqrt(
            np.mean((actual - predicted) ** 2)
        ),
        "pearson": (
            np.corrcoef(actual, predicted)[0, 1]
            if np.std(actual) > 0 and np.std(predicted) > 0
            else np.nan
        )
    })

oof_summary = pd.DataFrame(oof_rows)

oof_summary.to_csv(
    OUT_DIR / "existing_oof_performance_snapshot.csv",
    index=False
)

display(oof_summary)
"""))

# ------------------------------------------------------------
# CELL 14 — RESEARCH INTERPRETATION
# ------------------------------------------------------------

cells.append(nbf.v4.new_markdown_cell("""
## Research interpretation rules

Interpret the outputs cautiously.

### 1. Predictor associations

Univariate Spearman correlations describe monotonic relationships. They do not control for other Combine measurements.

### 2. Ridge coefficients

Coefficients are standardized within each training fold. Positive and negative signs describe conditional associations under the fitted specification.

### 3. Coefficient stability

A coefficient with the same sign across five folds is more stable than one whose sign changes repeatedly. Five folds are still a small number of replications.

### 4. Multicollinearity

Highly correlated predictors may share explanatory signal. Ridge regularisation distributes that signal across predictors, so individual coefficient magnitude should not be treated as definitive feature importance.

### 5. Predictive performance

The existing OOF performance results remain the primary evaluation evidence. This notebook does not replace the nested cross-validation analysis.

### 6. Scientific language

Use:
- "associated with"
- "predictive signal"
- "conditional relationship"
- "out-of-sample performance"

Avoid:
- "causes"
- "determines"
- "proves"
- "improves performance" unless supported by the relevant uncertainty analysis.
"""))

# ------------------------------------------------------------
# CELL 15 — EXPORT MANIFEST
# ------------------------------------------------------------

cells.append(nbf.v4.new_code_cell("""
manifest = {
    "notebook": "09_predictor_interpretation_and_feature_stability.ipynb",
    "dataset": str(DATASET_PATH.relative_to(ROOT)),
    "dataset_rows": int(len(data)),
    "receivers": int(data.nfl_id.nunique()),
    "outer_folds": sorted(
        int(x) for x in folds.outer_fold.unique()
    ),
    "outcomes": OUTCOMES,
    "traditional_predictors": TRADITIONAL,
    "tracking_predictors": TRACKING,
    "coefficient_method": (
        "Fold-specific standardized Ridge coefficients using "
        "selected alpha from existing inner tuning. "
        "Scaling fitted on outer training data only."
    ),
    "causal_claims": False,
    "outputs": sorted(
        p.name for p in OUT_DIR.iterdir()
    )
}

with open(
    OUT_DIR / "predictor_interpretation_manifest.json",
    "w",
    encoding="utf-8"
) as f:

    import json
    json.dump(manifest, f, indent=2)

expected = [
    "predictor_distribution_summary.csv",
    "predictor_spearman_correlation_matrix.csv",
    "predictor_correlation_pairs.csv",
    "univariate_spearman_associations.csv",
    "fold_specific_standardized_coefficients.csv",
    "coefficient_stability_summary.csv",
    "existing_oof_performance_snapshot.csv",
    "predictor_interpretation_manifest.json"
]

missing = [
    name for name in expected
    if not (OUT_DIR / name).exists()
]

print("=" * 75)
print("NOTEBOOK 09 COMPLETE")
print("=" * 75)
print("Output:", OUT_DIR)
print("Coefficient rows:", len(coefficients))
print("Missing outputs:", missing)

assert not missing
"""))

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

nbf.write(nb, OUTPUT)

print("=" * 70)
print("NOTEBOOK 09 GENERATED")
print("=" * 70)
print("File:", OUTPUT)
print("Cells:", len(nb.cells))
