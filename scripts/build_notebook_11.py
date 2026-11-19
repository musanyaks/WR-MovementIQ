
from pathlib import Path
import json

# ============================================================
# 0. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NOTEBOOK = NOTEBOOK_DIR / "11_final_results_synthesis.ipynb"


# ============================================================
# 1. NOTEBOOK CELL HELPERS
# ============================================================

def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = []


# ============================================================
# 2. TITLE AND RESEARCH DESCRIPTION
# ============================================================

cells.append(markdown("""# Notebook 11 — Final Results Synthesis

## WR-MovementIQ

**Beyond the 40-Yard Dash: Linking NFL Combine Receiver Movement Mechanics to Regular-Season Performance Using Player Tracking Data**

### Purpose

This notebook consolidates independently audited outputs from the preceding analysis notebooks.

It does not train models, tune hyperparameters, change player samples, or generate new predictions.

It produces:

- Main nested-CV model comparison
- Leave-one-draft-cohort-out results
- Receiver-level bootstrap uncertainty
- Feature ablation and sensitivity summaries
- Eligibility-threshold comparisons
- Predictor association and coefficient stability summaries
- Publication-quality figures
- Final reproducibility manifest

**Important:** Results are observational and predictive. They do not establish causal relationships.
"""))


# ============================================================
# 3. ENVIRONMENT AND PROJECT ROOT
# ============================================================

cells.append(code("""from pathlib import Path
import json
import hashlib
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolve project root independently of notebook execution directory.
CURRENT_DIR = Path.cwd().resolve()

PROJECT_ROOT_CANDIDATES = [
    CURRENT_DIR,
    CURRENT_DIR.parent,
]

ROOT = next(
    (
        candidate
        for candidate in PROJECT_ROOT_CANDIDATES
        if (
            candidate
            / "reports"
            / "nested_ridge_modeling"
            / "nested_cv_metrics.csv"
        ).is_file()
    ),
    None,
)

if ROOT is None:
    raise FileNotFoundError(
        "Cannot locate WR-MovementIQ project root. "
        f"Current directory: {CURRENT_DIR}"
    )

print("Resolved project root:", ROOT)

OUT = ROOT / "reports" / "final_results_synthesis"
FIG = OUT / "figures"
TABLES = OUT / "tables"

for folder in [OUT, FIG, TABLES]:
    folder.mkdir(parents=True, exist_ok=True)

print("Project root:", ROOT)
print("Output directory:", OUT)
print("Python:", sys.version)
print("Platform:", platform.platform())
"""))


# ============================================================
# 4. LOAD AND VALIDATE SOURCE ARTIFACTS
# ============================================================

cells.append(markdown("""## 1. Load and validate authoritative source artifacts

All source artifacts are loaded from their existing project locations.

Notebook 10 historical outputs are not used as authoritative results because their saved predictions and tuning artifacts failed independent reconstruction.

Notebook 10B all-threshold results are used because their complete independent reconstruction passed.
"""))


cells.append(code("""SOURCE_FILES = {
    "nested_cv": "reports/nested_ridge_modeling/nested_cv_metrics.csv",
    "loco": "reports/nested_ridge_modeling/loco_metrics.csv",
    "primary_decisions": (
        "reports/fold_stratified_receiver_validation/"
        "primary_speed_outcome_decisions.csv"
    ),
    "receiver_bootstrap": (
        "reports/fold_stratified_receiver_validation/"
        "fold_stratified_receiver_bootstrap.csv"
    ),
    "speed_ablation": (
        "reports/robustness_sensitivity/"
        "speed_feature_sensitivity.csv"
    ),
    "fold_stability": (
        "reports/robustness_sensitivity/"
        "sensitivity_fold_stability.csv"
    ),
    "eligibility": (
        "reports/sample_eligibility_sensitivity/"
        "eligibility_threshold_comparison.csv"
    ),
    "associations": (
        "reports/predictor_interpretation/"
        "univariate_spearman_associations.csv"
    ),
    "coefficients": (
        "reports/predictor_interpretation/"
        "coefficient_stability_summary.csv"
    ),
    "all_threshold_audit": (
        "reports/robustness_sensitivity/"
        "notebook10B_all_threshold_reconstruction_audit.csv"
    ),
}

DATA = {}
FILE_AUDIT = []

for name, relative_path in SOURCE_FILES.items():

    path = ROOT / relative_path

    if not path.is_file():
        raise FileNotFoundError(
            f"Required source artifact missing: {path}"
        )

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    df = pd.read_csv(path)

    DATA[name] = df

    FILE_AUDIT.append({
        "artifact": name,
        "path": relative_path,
        "rows": len(df),
        "columns": len(df.columns),
        "sha256": digest,
        "status": "PRESENT",
    })

    print(
        f"{name:22s} "
        f"rows={len(df):6d} "
        f"columns={len(df.columns):3d}"
    )

file_audit = pd.DataFrame(FILE_AUDIT)

file_audit.to_csv(
    TABLES / "source_artifact_inventory.csv",
    index=False,
)

nested = DATA["nested_cv"]
loco = DATA["loco"]
decisions = DATA["primary_decisions"]
bootstrap = DATA["receiver_bootstrap"]
ablation = DATA["speed_ablation"]
stability = DATA["fold_stability"]
eligibility = DATA["eligibility"]
associations = DATA["associations"]
coefficients = DATA["coefficients"]
threshold_audit = DATA["all_threshold_audit"]

EXPECTED_ROWS = {
    "nested_cv": 24,
    "loco": 72,
    "primary_decisions": 6,
    "receiver_bootstrap": 18,
    "speed_ablation": 15,
    "fold_stability": 54,
    "eligibility": 300,
    "associations": 54,
    "coefficients": 108,
    "all_threshold_audit": 15,
}

for name, expected in EXPECTED_ROWS.items():

    actual = len(DATA[name])

    if actual != expected:
        raise ValueError(
            f"{name}: expected {expected} rows, found {actual}"
        )

required_audit_columns = {"pass"}

if not required_audit_columns.issubset(threshold_audit.columns):
    raise ValueError(
        "All-threshold audit is missing the required 'pass' column."
    )

audit_pass = (
    threshold_audit["pass"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

if not audit_pass.all():
    raise AssertionError(
        "At least one Notebook 10B reconstruction audit failed."
    )

print("\\nSource schema and row-count checks passed.")
print("All-threshold independent reconstruction: PASS")
"""))


# ============================================================
# 5. MAIN NESTED CROSS-VALIDATION RESULTS
# ============================================================

cells.append(markdown("""## 2. Main nested cross-validation performance

The main comparison includes:

- BASELINE
- TRADITIONAL
- TRACKING
- COMBINED

The metrics are descriptive out-of-fold performance summaries across player-seasons.
"""))


cells.append(code("""MODEL_ORDER = [
    "BASELINE",
    "TRADITIONAL",
    "TRACKING",
    "COMBINED",
]

nested_summary = nested.copy()

nested_summary["model"] = pd.Categorical(
    nested_summary["model"],
    categories=MODEL_ORDER,
    ordered=True,
)

for col in ["mae", "rmse", "r2"]:
    nested_summary[col] = pd.to_numeric(
        nested_summary[col],
        errors="coerce",
    )

nested_summary = nested_summary.sort_values(
    ["outcome", "model"]
)

nested_summary.to_csv(
    TABLES / "main_nested_cv_performance.csv",
    index=False,
)

display(nested_summary)

for outcome in nested_summary["outcome"].unique():

    subset = nested_summary[
        nested_summary["outcome"] == outcome
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        subset["model"].astype(str),
        subset["mae"],
    )

    ax.set_title(f"Nested CV MAE — {outcome}")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()

    fig.savefig(
        FIG / f"nested_mae_{outcome}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

print("\\nDescriptive mean metrics across outcomes:")
display(
    nested_summary.groupby(
        "model",
        observed=True,
    )[["mae", "rmse", "r2"]]
    .mean()
    .sort_values("mae")
)
"""))


# ============================================================
# 6. LEAVE-ONE-COHORT-OUT RESULTS
# ============================================================

cells.append(markdown("""## 3. Leave-one-draft-cohort-out validation

These results evaluate how models perform when an entire draft cohort is held out.

Cohort-specific estimates should be interpreted cautiously because sample sizes differ.
"""))


cells.append(code("""loco_summary = loco.copy()

for col in [
    "n",
    "mae",
    "rmse",
    "r2",
    "held_out_draft_year",
]:
    loco_summary[col] = pd.to_numeric(
        loco_summary[col],
        errors="coerce",
    )

loco_summary = loco_summary.sort_values(
    ["outcome", "held_out_draft_year", "model"]
)

loco_summary.to_csv(
    TABLES / "loco_performance.csv",
    index=False,
)

display(loco_summary)

speed_outcomes = [
    "game_mean_speed_raw_median",
    "game_p95_speed_raw_median",
    "game_peak_speed_raw_median",
]

for outcome in speed_outcomes:

    subset = loco_summary[
        loco_summary["outcome"] == outcome
    ].copy()

    pivot = subset.pivot(
        index="held_out_draft_year",
        columns="model",
        values="mae",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))

    pivot.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(f"LOCO MAE — {outcome}")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlabel("Held-out draft cohort")
    ax.legend(title="Model")

    fig.tight_layout()

    fig.savefig(
        FIG / f"loco_{outcome}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)
"""))


# ============================================================
# 7. RECEIVER-LEVEL BOOTSTRAP
# ============================================================

cells.append(markdown("""## 4. Receiver-level bootstrap uncertainty

The bootstrap compares receiver-level model errors against the baseline.

Negative MAE differences favour the evaluated model.

Confidence intervals containing zero do not establish a reliable improvement.
"""))


cells.append(code("""bootstrap_summary = bootstrap.copy()

numeric_cols = [
    "point_estimate",
    "ci_2_5",
    "ci_97_5",
    "probability_difference_below_zero",
]

for col in numeric_cols:

    if col in bootstrap_summary.columns:
        bootstrap_summary[col] = pd.to_numeric(
            bootstrap_summary[col],
            errors="coerce",
        )

bootstrap_summary.to_csv(
    TABLES / "receiver_level_bootstrap_summary.csv",
    index=False,
)

display(bootstrap_summary)

primary = decisions.copy()

for col in [
    "point_estimate",
    "ci_2_5",
    "ci_97_5",
    "probability_difference_below_zero",
]:
    if col in primary.columns:
        primary[col] = pd.to_numeric(
            primary[col],
            errors="coerce",
        )

primary.to_csv(
    TABLES / "primary_speed_outcome_decisions.csv",
    index=False,
)

required_plot_columns = {
    "outcome",
    "model",
    "point_estimate",
    "ci_2_5",
    "ci_97_5",
}

if not required_plot_columns.issubset(primary.columns):
    raise ValueError(
        "Primary decision table does not contain the required "
        "confidence interval plotting columns. "
        f"Available columns: {list(primary.columns)}"
    )

plot_data = primary.copy()

plot_data["label"] = (
    plot_data["outcome"].astype(str)
    .str.replace("game_", "", regex=False)
    + " — "
    + plot_data["model"].astype(str)
)

plot_data = plot_data.sort_values(
    ["outcome", "model"]
)

y = np.arange(len(plot_data))

x = plot_data["point_estimate"].to_numpy(dtype=float)

lower_error = (
    x - plot_data["ci_2_5"].to_numpy(dtype=float)
)

upper_error = (
    plot_data["ci_97_5"].to_numpy(dtype=float) - x
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(
    x,
    y,
    xerr=[lower_error, upper_error],
    fmt="o",
    capsize=4,
)

ax.axvline(
    0,
    linestyle="--",
)

ax.set_yticks(y)
ax.set_yticklabels(plot_data["label"])

ax.set_xlabel(
    "MAE difference versus baseline "
    "(negative favours evaluated model)"
)

ax.set_title(
    "Receiver-level bootstrap: primary speed outcomes"
)

fig.tight_layout()

fig.savefig(
    FIG / "primary_speed_bootstrap_intervals.png",
    dpi=220,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)
"""))


# ============================================================
# 8. FEATURE ABLATION AND FOLD STABILITY
# ============================================================

cells.append(markdown("""## 5. Feature ablation and fold stability

These analyses examine whether individual tracking-feature groups contribute consistently.

Point estimates should be considered alongside fold-to-fold variability.
"""))


cells.append(code("""ablation_summary = ablation.copy()
stability_summary = stability.copy()

ablation_summary.to_csv(
    TABLES / "speed_feature_ablation.csv",
    index=False,
)

stability_summary.to_csv(
    TABLES / "sensitivity_fold_stability.csv",
    index=False,
)

display(ablation_summary)
display(stability_summary)

for outcome in speed_outcomes:

    subset = ablation_summary[
        ablation_summary["outcome"] == outcome
    ].copy()

    subset["mae"] = pd.to_numeric(
        subset["mae"],
        errors="coerce",
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        subset["model"].astype(str),
        subset["mae"],
    )

    ax.set_title(
        f"Speed feature ablation — {outcome}"
    )

    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    fig.savefig(
        FIG / f"ablation_{outcome}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)
"""))


# ============================================================
# 9. SAMPLE ELIGIBILITY SENSITIVITY
# ============================================================

cells.append(markdown("""## 6. Sample eligibility sensitivity

This section compares model results across minimum receiver-play eligibility thresholds.

The threshold analysis uses the independently reconstructed Notebook 10B outputs.
"""))


cells.append(code("""eligibility_summary = eligibility.copy()

for col in [
    "threshold",
    "player_seasons",
    "unique_receivers",
    "mae",
    "rmse",
    "r2",
]:
    if col in eligibility_summary.columns:
        eligibility_summary[col] = pd.to_numeric(
            eligibility_summary[col],
            errors="coerce",
        )

eligibility_summary = eligibility_summary.sort_values(
    ["outcome", "model", "threshold"]
)

eligibility_summary.to_csv(
    TABLES / "eligibility_threshold_comparison.csv",
    index=False,
)

display(eligibility_summary.head(30))

required_sample_columns = {
    "threshold",
    "player_seasons",
    "unique_receivers",
}

if not required_sample_columns.issubset(
    eligibility_summary.columns
):
    raise ValueError(
        "Eligibility table lacks sample-size columns. "
        f"Available columns: {list(eligibility_summary.columns)}"
    )

sample_sizes = (
    eligibility_summary[
        ["threshold", "player_seasons", "unique_receivers"]
    ]
    .drop_duplicates()
    .sort_values("threshold")
)

display(sample_sizes)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(
    sample_sizes["threshold"],
    sample_sizes["player_seasons"],
    marker="o",
    label="Player-seasons",
)

ax.plot(
    sample_sizes["threshold"],
    sample_sizes["unique_receivers"],
    marker="o",
    label="Unique receivers",
)

ax.set_title(
    "Sample size by minimum receiver-play eligibility threshold"
)

ax.set_xlabel("Minimum receiver plays")
ax.set_ylabel("Sample size")
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG / "eligibility_sample_sizes.png",
    dpi=220,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

threshold_audit.to_csv(
    TABLES / "notebook10B_all_threshold_reconstruction_audit.csv",
    index=False,
)

print("Eligibility sensitivity and independent reconstruction audit exported.")
"""))


# ============================================================
# 10. PREDICTOR ASSOCIATIONS AND COEFFICIENT STABILITY
# ============================================================

cells.append(markdown("""## 7. Predictor associations and coefficient stability

Univariate Spearman associations describe marginal relationships.

They must not be interpreted as independent predictor effects or causal effects.
"""))


cells.append(code("""association_summary = associations.copy()

if "spearman_rho" not in association_summary.columns:
    raise ValueError(
        "Missing spearman_rho in association source. "
        f"Columns: {list(association_summary.columns)}"
    )

association_summary["spearman_rho"] = pd.to_numeric(
    association_summary["spearman_rho"],
    errors="coerce",
)

association_summary = association_summary.sort_values(
    ["outcome", "spearman_rho"],
    key=lambda s: (
        s.abs()
        if s.name == "spearman_rho"
        else s
    ),
    ascending=[True, False],
)

association_summary.to_csv(
    TABLES / "univariate_spearman_associations.csv",
    index=False,
)

coefficient_summary = coefficients.copy()

coefficient_summary.to_csv(
    TABLES / "coefficient_stability_summary.csv",
    index=False,
)

display(association_summary)
display(coefficient_summary)

required_heatmap_columns = {
    "predictor",
    "outcome",
    "spearman_rho",
}

if not required_heatmap_columns.issubset(
    association_summary.columns
):
    raise ValueError(
        "Association table is missing heatmap columns. "
        f"Columns: {list(association_summary.columns)}"
    )

heat = association_summary.pivot(
    index="predictor",
    columns="outcome",
    values="spearman_rho",
)

fig, ax = plt.subplots(
    figsize=(11, max(5, 0.45 * len(heat)))
)

im = ax.imshow(
    heat.to_numpy(dtype=float),
    aspect="auto",
    vmin=-1,
    vmax=1,
)

ax.set_xticks(
    np.arange(len(heat.columns))
)

ax.set_xticklabels(
    heat.columns,
    rotation=35,
    ha="right",
)

ax.set_yticks(
    np.arange(len(heat.index))
)

ax.set_yticklabels(
    heat.index
)

ax.set_title(
    "Univariate Spearman associations"
)

fig.colorbar(
    im,
    ax=ax,
    label="Spearman rho",
)

fig.tight_layout()

fig.savefig(
    FIG / "predictor_spearman_heatmap.png",
    dpi=220,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

if "coefficient_sd" in coefficient_summary.columns:

    coefficient_summary["coefficient_sd"] = pd.to_numeric(
        coefficient_summary["coefficient_sd"],
        errors="coerce",
    )

    display(
        coefficient_summary.groupby(
            ["outcome", "model"]
        )["coefficient_sd"]
        .mean()
        .rename("mean_coefficient_sd")
        .reset_index()
    )
"""))


# ============================================================
# 11. RESEARCH INTERPRETATION
# ============================================================

cells.append(markdown("""## 8. Research interpretation

The following interpretation is deliberately conservative.

It reflects the observed results and their limitations rather than treating favourable point estimates as established improvements.
"""))


cells.append(code("""interpretation = {
    "research_question": (
        "To what extent do movement characteristics measured during "
        "NFL Combine receiver-specific drills explain variation in "
        "wide receivers' in-game movement characteristics?"
    ),

    "main_findings": [
        (
            "Traditional and tracking-derived Combine measures show "
            "some associations with in-game movement outcomes."
        ),
        (
            "Tracking-derived features do not demonstrate consistent, "
            "convincing incremental improvement in absolute predictive accuracy."
        ),
        (
            "Some combined models have favourable point estimates for peak "
            "and 95th-percentile game speed, but receiver-level uncertainty "
            "intervals include zero."
        ),
        (
            "Predictor coefficient magnitudes and signs vary across folds, "
            "limiting claims about stable independent feature effects."
        ),
        (
            "Eligibility sensitivity is independently reconstructed across "
            "all five thresholds."
        ),
    ],

    "limitations": [
        (
            "The primary modeling sample contains 93 player-seasons "
            "from 54 receivers."
        ),
        (
            "The sample is small relative to the complexity of NFL "
            "receiver performance."
        ),
        (
            "Complete-case requirements reduce the available sample."
        ),
        (
            "Draft cohorts have unequal sample sizes."
        ),
        (
            "Combine speed predictors are strongly correlated."
        ),
        (
            "Regularized coefficients are sensitive to the training "
            "fold and penalty."
        ),
        (
            "Observational associations do not establish causal effects."
        ),
        (
            "LOCO estimates are cohort-specific and should not be "
            "interpreted as definitive evidence of future-season "
            "generalization."
        ),
    ],

    "interpretation_guardrails": [
        (
            "Do not describe an uncertain point estimate as a proven gain."
        ),
        (
            "Do not interpret negative R-squared as a negative correlation."
        ),
        (
            "Do not describe univariate Spearman associations as "
            "independent effects."
        ),
        (
            "Do not infer causal effects from predictive associations."
        ),
        (
            "Do not claim robust generalization from a small held-out cohort."
        ),
    ],
}

interpretation_path = OUT / "research_interpretation.json"

interpretation_path.write_text(
    json.dumps(
        interpretation,
        indent=2,
    ),
    encoding="utf-8",
)

for heading, items in interpretation.items():

    print(f"\\n{heading.upper()}")

    if isinstance(items, list):

        for item in items:
            print("-", item)

    else:
        print(items)
"""))


# ============================================================
# 12. REPRODUCIBILITY MANIFEST
# ============================================================

cells.append(markdown("""## 9. Reproducibility manifest and final audit

This final section records the source artifact hashes, audit status, execution environment and generated outputs.

The manifest provides an auditable record of the synthesis.
"""))


cells.append(code("""manifest = {
    "project": "WR-MovementIQ",

    "notebook": "11_final_results_synthesis.ipynb",

    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "python_version": sys.version,

    "platform": platform.platform(),

    "analysis_type": (
        "Results synthesis; no model refitting."
    ),

    "primary_dataset_sha256": (
        "02fbd802522973f7c4d100da57cd680f7bbfde6346a0431a7aa1fe5e25807c5c"
    ),

    "notebook10_status": (
        "Historical outputs retained for provenance but not authoritative "
        "because saved tuning and prediction artifacts failed independent "
        "reconstruction."
    ),

    "notebook10B_status": (
        "PASS: all-threshold independent reconstruction."
    ),

    "threshold_audit_checks": int(
        len(threshold_audit)
    ),

    "threshold_audit_passes": int(
        audit_pass.sum()
    ),

    "threshold_audit_failures": int(
        (~audit_pass).sum()
    ),

    "source_artifacts": FILE_AUDIT,

    "outputs": [],
}

assert manifest["threshold_audit_failures"] == 0

assert manifest["threshold_audit_checks"] == 15

manifest_path = OUT / "final_results_manifest.json"

# Capture all generated artifacts before writing the manifest.
for path in sorted(OUT.rglob("*")):

    if (
        path.is_file()
        and path.name != "final_results_manifest.json"
    ):

        manifest["outputs"].append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        })

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)

file_audit.to_csv(
    OUT / "source_artifact_inventory.csv",
    index=False,
)

print("\\n" + "=" * 80)
print("NOTEBOOK 11 SYNTHESIS COMPLETE")
print("=" * 80)

print("Source artifacts:", len(FILE_AUDIT))
print("Nested CV rows:", len(nested))
print("LOCO rows:", len(loco))
print("Bootstrap rows:", len(bootstrap))
print("Eligibility comparison rows:", len(eligibility))

print(
    "Independent threshold audit:",
    "PASS" if audit_pass.all() else "FAIL",
)

print(
    "Figures:",
    len(list(FIG.glob("*.png"))),
)

print(
    "Tables:",
    len(list(TABLES.glob("*.csv"))),
)

print("Manifest:", manifest_path)
"""))


# ============================================================
# 13. NOTEBOOK METADATA AND CELL IDS
# ============================================================

for index, cell in enumerate(cells):
    cell["id"] = f"notebook11-cell-{index:02d}"


notebook = {
    "cells": cells,

    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },

        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },

    "nbformat": 4,
    "nbformat_minor": 5,
}


# ============================================================
# 14. WRITE NOTEBOOK
# ============================================================

OUTPUT_NOTEBOOK.write_text(
    json.dumps(
        notebook,
        indent=2,
    ),
    encoding="utf-8",
)

print("=" * 70)
print("NOTEBOOK 11 GENERATED SUCCESSFULLY")
print("=" * 70)

print("Notebook:", OUTPUT_NOTEBOOK)
print("Cells:", len(cells))
print("Project root:", ROOT)

print(
    "Purpose: Final research results synthesis. "
    "No model training or refitting."
)
