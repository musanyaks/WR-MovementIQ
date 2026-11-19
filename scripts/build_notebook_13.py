
from pathlib import Path
import json

# ============================================================
# WR-MovementIQ
# NOTEBOOK 13 — COMPETITION METHODOLOGY AND RESEARCH REPORT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = NOTEBOOK_DIR / "13_competition_research_report.ipynb"


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

cells.append(markdown("""# Notebook 13 — Competition Methodology and Research Report

## WR-MovementIQ

### Beyond the 40-Yard Dash: Linking NFL Combine Receiver Movement Mechanics to Regular-Season Performance Using Player Tracking Data

**Purpose:** Consolidate the validated research pipeline into a reproducible competition research report.

### Research integrity

This notebook:
- Uses audited outputs from Notebooks 01–12.
- Does not retrain models.
- Does not change the frozen analysis sample.
- Does not alter model predictions or validation assignments.
- Does not treat associations as causal effects.
- Preserves uncertainty and sample-size limitations.

### Deliverables

1. Complete methodology and results report in Markdown.
2. Main results tables.
3. Validation and uncertainty tables.
4. Embedded publication figures.
5. Source provenance inventory.
6. Reproducibility manifest.
"""))


cells.append(code(r'''
# ============================================================
# 1. ENVIRONMENT AND PROJECT ROOT
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = Path.cwd().resolve()

candidates = [
    CURRENT_DIR,
    CURRENT_DIR.parent,
]

ROOT = next(
    (
        p for p in candidates
        if (
            p / "reports" / "final_results_synthesis"
            / "tables" / "main_nested_cv_performance.csv"
        ).is_file()
    ),
    None,
)

if ROOT is None:
    raise FileNotFoundError(
        "Cannot locate WR-MovementIQ project root. "
        f"Current directory: {CURRENT_DIR}"
    )

FINAL_TABLES = (
    ROOT / "reports" / "final_results_synthesis" / "tables"
)

VISUAL_DIR = (
    ROOT / "reports" / "publication_visualizations"
)

VISUAL_TABLES = VISUAL_DIR / "tables"

OUT = ROOT / "reports" / "competition_research_report"
TABLE_DIR = OUT / "tables"

for folder in [OUT, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("Project root:", ROOT)
print("Report output:", OUT)
print("Python:", sys.version)
print("Platform:", platform.platform())
'''))


cells.append(markdown("""## 2. Load authoritative research results

The final synthesis tables from Notebook 11 are the primary source for model results. Notebook 12 supplies the publication figure catalogue and visual assets.
"""))


cells.append(code(r'''
# ============================================================
# 2. LOAD VERIFIED SOURCE ARTIFACTS
# ============================================================

FILES = {
    "nested": FINAL_TABLES / "main_nested_cv_performance.csv",
    "loco": FINAL_TABLES / "loco_performance.csv",
    "decisions": FINAL_TABLES / "primary_speed_outcome_decisions.csv",
    "bootstrap": FINAL_TABLES / "receiver_level_bootstrap_summary.csv",
    "ablation": FINAL_TABLES / "speed_feature_ablation.csv",
    "stability": FINAL_TABLES / "sensitivity_fold_stability.csv",
    "eligibility": FINAL_TABLES / "eligibility_threshold_comparison.csv",
    "associations": FINAL_TABLES / "univariate_spearman_associations.csv",
    "coefficients": FINAL_TABLES / "coefficient_stability_summary.csv",
    "figure_catalogue": VISUAL_TABLES / "publication_figure_catalogue.csv",
}

DATA = {}

for name, path in FILES.items():

    if not path.is_file():
        raise FileNotFoundError(path)

    DATA[name] = pd.read_csv(path)

nested = DATA["nested"]
loco = DATA["loco"]
decisions = DATA["decisions"]
bootstrap = DATA["bootstrap"]
ablation = DATA["ablation"]
stability = DATA["stability"]
eligibility = DATA["eligibility"]
associations = DATA["associations"]
coefficients = DATA["coefficients"]
figure_catalogue = DATA["figure_catalogue"]

EXPECTED_ROWS = {
    "nested": 24,
    "loco": 72,
    "decisions": 6,
    "bootstrap": 18,
    "ablation": 15,
    "stability": 54,
    "eligibility": 300,
    "associations": 54,
    "coefficients": 108,
    "figure_catalogue": 12,
}

for name, expected in EXPECTED_ROWS.items():

    actual = len(DATA[name])

    if actual != expected:
        raise AssertionError(
            f"{name}: expected {expected} rows, found {actual}"
        )

print("All source tables loaded.")
print("All expected row-count checks passed.")

for name, df in DATA.items():
    print(f"{name:20s} {df.shape}")
'''))


cells.append(markdown("""## 3. Research design and sample definition

The primary modeling sample contains 93 player-season observations from 54 receivers. The analysis uses complete cases for the core traditional Combine predictors and the specified game movement outcomes.

The primary sample is frozen. No imputation is introduced in this report.
"""))


cells.append(code(r'''
# ============================================================
# 3. REPORT HELPERS AND SAMPLE AUDIT
# ============================================================

def markdown_table(df, columns=None, digits=4):
    """
    Convert a DataFrame to a Markdown table without requiring
    the optional tabulate package.
    """
    view = df.copy()

    if columns is not None:
        view = view[columns]

    def fmt(value):

        if pd.isna(value):
            return "—"

        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"

        return str(value)

    headers = list(view.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in view.iterrows():
        lines.append(
            "| " + " | ".join(fmt(v) for v in row.tolist()) + " |"
        )

    return "\n".join(lines)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome_label(value):
    labels = {
        "game_mean_speed_raw_median": "Mean game speed",
        "game_p95_speed_raw_median": "P95 game speed",
        "game_peak_speed_raw_median": "Peak game speed",
        "game_mean_acceleration_raw_median": "Mean game acceleration",
        "game_peak_acceleration_raw_median": "Peak game acceleration",
        "game_path_efficiency_median": "Game path efficiency",
    }

    return labels.get(
        value,
        str(value).replace("_", " ").title(),
    )


# Primary sample facts established by Notebook 06/07.
PRIMARY_PLAYER_SEASONS = 93
PRIMARY_RECEIVERS = 54
OUTER_FOLDS = 5

assert len(nested) == 24
assert nested["n"].nunique() == 1
assert int(nested["n"].iloc[0]) == PRIMARY_PLAYER_SEASONS

print("Primary player-seasons:", PRIMARY_PLAYER_SEASONS)
print("Primary receivers:", PRIMARY_RECEIVERS)
print("Outer validation folds:", OUTER_FOLDS)

# Sample-size sensitivity table.
sample_sizes = (
    eligibility[
        ["threshold", "player_seasons", "unique_receivers"]
    ]
    .drop_duplicates()
    .sort_values("threshold")
)

sample_sizes.to_csv(
    TABLE_DIR / "sample_eligibility_summary.csv",
    index=False,
)

display(sample_sizes)
'''))


cells.append(markdown("""## 4. Model specifications and validation

Four principal model specifications are compared:

- **Baseline:** intercept-only reference.
- **Traditional:** 40-yard dash, vertical jump and broad jump.
- **Tracking:** Combine tracking-derived movement predictors.
- **Combined:** traditional and tracking-derived predictors together.

The primary analysis uses receiver-grouped five-fold outer cross-validation. Hyperparameters are selected within the training data using receiver-grouped inner cross-validation.

A separate leave-one-draft-cohort-out (LOCO) analysis evaluates performance when a draft cohort is withheld.
"""))


cells.append(code(r'''
# ============================================================
# 4. MODEL SPECIFICATIONS AND VALIDATION TABLES
# ============================================================

model_specs = pd.DataFrame([
    {
        "model": "BASELINE",
        "predictors": "Intercept-only reference",
    },
    {
        "model": "TRADITIONAL",
        "predictors": "40-yard dash, vertical jump, broad jump",
    },
    {
        "model": "TRACKING",
        "predictors": (
            "Combine-derived mean speed, peak speed, "
            "P95 speed, mean acceleration, peak acceleration "
            "and path efficiency"
        ),
    },
    {
        "model": "COMBINED",
        "predictors": (
            "Traditional Combine measurements and "
            "tracking-derived movement predictors"
        ),
    },
])

model_specs.to_csv(
    TABLE_DIR / "model_specifications.csv",
    index=False,
)

display(model_specs)

# Export primary nested-CV results.
nested_report = nested.copy()
nested_report["outcome_label"] = nested_report["outcome"].map(
    outcome_label
)

nested_report.to_csv(
    TABLE_DIR / "nested_cv_results.csv",
    index=False,
)

# Export LOCO results.
loco_report = loco.copy()
loco_report["outcome_label"] = loco_report["outcome"].map(
    outcome_label
)

loco_report.to_csv(
    TABLE_DIR / "loco_results.csv",
    index=False,
)

print("Nested CV rows:", len(nested_report))
print("LOCO rows:", len(loco_report))
'''))


cells.append(markdown("""## 5. Primary predictive performance

The principal evaluation metrics are mean absolute error (MAE), root mean squared error (RMSE), coefficient of determination (R²), and Spearman rank correlation where available.

MAE and RMSE are reported on the source outcome scale. Values from different outcomes must not be compared as though they share a common measurement unit.

Negative R² means the model performed worse than predicting the test-set mean for that evaluation; it does not mean the model has a negative correlation.
"""))


cells.append(code(r'''
# ============================================================
# 5. PRIMARY RESULTS TABLES
# ============================================================

nested_display = nested_report[
    [
        "outcome_label",
        "model",
        "n",
        "mae",
        "rmse",
        "r2",
        "spearman",
    ]
].copy()

nested_display = nested_display.sort_values(
    ["outcome_label", "model"]
)

nested_display.to_csv(
    TABLE_DIR / "primary_model_performance.csv",
    index=False,
)

display(nested_display)
'''))


cells.append(markdown("""## 6. Primary speed-outcome uncertainty

The receiver-level bootstrap compares model MAE against the baseline. The reported point estimate and confidence interval are differences in MAE.

- Negative values favour the evaluated model.
- Positive values favour the baseline.
- A confidence interval crossing zero indicates that the available evidence does not establish a clear difference.

The bootstrap uses 5,000 replicates and fold-stratified receiver resampling.
"""))


cells.append(code(r'''
# ============================================================
# 6. BOOTSTRAP UNCERTAINTY RESULTS
# ============================================================

speed_decisions = decisions.copy()

speed_decisions["outcome_label"] = (
    speed_decisions["outcome"].map(outcome_label)
)

decision_columns = [
    "outcome_label",
    "model",
    "point_estimate",
    "ci_2_5",
    "ci_97_5",
    "probability_difference_below_zero",
    "folds_model_better",
    "folds_model_worse",
    "receivers",
]

decision_display = speed_decisions[
    decision_columns
].copy()

decision_display.to_csv(
    TABLE_DIR / "primary_speed_bootstrap_decisions.csv",
    index=False,
)

display(decision_display)

# Focused primary speed outcomes.
primary_speed = speed_decisions[
    speed_decisions["outcome"].isin([
        "game_p95_speed_raw_median",
        "game_peak_speed_raw_median",
    ])
].copy()

display(
    primary_speed[
        [
            "outcome_label",
            "model",
            "point_estimate",
            "ci_2_5",
            "ci_97_5",
            "probability_difference_below_zero",
        ]
    ]
)
'''))


cells.append(markdown("""## 7. Draft-cohort generalization

LOCO results are reported separately for the 2023, 2024 and 2025 held-out cohorts. The number of test player-seasons differs by cohort.

These results should be interpreted as cohort-specific evidence, not as proof of generalization to all future NFL receivers.
"""))


cells.append(code(r'''
# ============================================================
# 7. LOCO COHORT SAMPLE SIZES AND PERFORMANCE
# ============================================================

loco_sizes = (
    loco[
        ["held_out_draft_year", "n"]
    ]
    .drop_duplicates()
    .sort_values("held_out_draft_year")
)

loco_sizes.to_csv(
    TABLE_DIR / "loco_cohort_sample_sizes.csv",
    index=False,
)

display(loco_sizes)

loco_speed = loco[
    loco["outcome"].isin([
        "game_mean_speed_raw_median",
        "game_p95_speed_raw_median",
        "game_peak_speed_raw_median",
    ])
].copy()

loco_speed["outcome_label"] = loco_speed["outcome"].map(
    outcome_label
)

loco_speed.to_csv(
    TABLE_DIR / "loco_speed_performance.csv",
    index=False,
)

display(
    loco_speed.sort_values(
        ["outcome", "held_out_draft_year", "model"]
    )
)
'''))


cells.append(markdown("""## 8. Feature associations and ablation

Univariate Spearman correlations describe marginal relationships between Combine predictors and game outcomes. They are not adjusted effects.

Feature ablation compares alternative tracking-feature specifications. These comparisons are descriptive and do not establish that an individual feature causes better prediction.
"""))


cells.append(code(r'''
# ============================================================
# 8. FEATURE ASSOCIATIONS AND ABLATION
# ============================================================

association_report = associations.copy()

association_report["outcome_label"] = (
    association_report["outcome"].map(outcome_label)
)

association_report["absolute_rho"] = (
    association_report["spearman_rho"].abs()
)

top_associations = association_report.sort_values(
    "absolute_rho",
    ascending=False,
).head(15)

top_associations.to_csv(
    TABLE_DIR / "strongest_univariate_associations.csv",
    index=False,
)

display(
    top_associations[
        [
            "outcome_label",
            "predictor",
            "group",
            "n",
            "spearman_rho",
        ]
    ]
)

ablation_report = ablation.copy()

ablation_report["outcome_label"] = (
    ablation_report["outcome"].map(outcome_label)
)

ablation_report.to_csv(
    TABLE_DIR / "tracking_feature_ablation.csv",
    index=False,
)

display(ablation_report)
'''))


cells.append(markdown("""## 9. Model and coefficient stability

Fold-level performance variation and coefficient variability are included as diagnostics.

Ridge coefficients are conditional on predictor scaling, model specification and the training fold. Their variation across folds is a warning against interpreting coefficient averages as stable independent effects.
"""))


cells.append(code(r'''
# ============================================================
# 9. STABILITY DIAGNOSTICS
# ============================================================

stability_report = stability.copy()

stability_report.to_csv(
    TABLE_DIR / "fold_stability.csv",
    index=False,
)

coefficient_report = coefficients.copy()

coefficient_report.to_csv(
    TABLE_DIR / "coefficient_stability.csv",
    index=False,
)

coefficient_report["coefficient_sd"] = pd.to_numeric(
    coefficient_report["coefficient_sd"],
    errors="coerce",
)

coefficient_variability = (
    coefficient_report.groupby(
        ["outcome", "model"],
        as_index=False,
    )
    .agg(
        mean_coefficient_sd=("coefficient_sd", "mean"),
        median_coefficient_sd=("coefficient_sd", "median"),
    )
)

coefficient_variability["outcome_label"] = (
    coefficient_variability["outcome"].map(outcome_label)
)

coefficient_variability.to_csv(
    TABLE_DIR / "coefficient_variability.csv",
    index=False,
)

display(stability_report.head(15))
display(coefficient_variability)
'''))


cells.append(markdown("""## 10. Publication figures

The figures below are generated and validated in Notebook 12. This report references those files rather than recreating or modifying them.
"""))


cells.append(code(r'''
# ============================================================
# 10. FIGURE INVENTORY AND VALIDATION
# ============================================================

figure_records = []

for _, row in figure_catalogue.iterrows():

    png = ROOT / row["png_path"]
    pdf = ROOT / row["pdf_path"]

    if not png.is_file():
        raise FileNotFoundError(png)

    if not pdf.is_file():
        raise FileNotFoundError(pdf)

    if png.stat().st_size < 5000:
        raise AssertionError(f"PNG too small: {png}")

    if pdf.stat().st_size < 1000:
        raise AssertionError(f"PDF too small: {pdf}")

    figure_records.append({
        "figure_id": row["figure_id"],
        "title": row["title"],
        "caption": row["caption"],
        "png_path": row["png_path"],
        "pdf_path": row["pdf_path"],
        "png_sha256": sha256(png),
        "pdf_sha256": sha256(pdf),
    })

print("Validated publication figures:", len(figure_records))

figure_audit = pd.DataFrame(figure_records)

figure_audit.to_csv(
    TABLE_DIR / "figure_inventory.csv",
    index=False,
)

display(
    figure_audit[["figure_id", "title"]]
)
'''))


cells.append(markdown("""## 11. Assemble the competition research report

The following report is generated from the validated results. The narrative is deliberately conservative where confidence intervals, cohort sizes or coefficient stability limit interpretation.
"""))


cells.append(code(r'''
# ============================================================
# 11. BUILD THE COMPLETE RESEARCH REPORT
# ============================================================

def relative_figure_path(filename):
    return f"../publication_visualizations/png/{filename}"


figure_links = []

for _, row in figure_catalogue.iterrows():

    figure_links.append(
        f"### {row['title']}\n\n"
        f"![{row['title']}]"
        f"({relative_figure_path(Path(row['png_path']).name)})\n\n"
        f"*{row['caption']}*\n"
    )

figures_markdown = "\n\n".join(figure_links)


nested_md = markdown_table(
    nested_display,
    digits=4,
)

decision_md = markdown_table(
    decision_display,
    digits=4,
)

loco_md = markdown_table(
    loco_speed[
        [
            "outcome_label",
            "held_out_draft_year",
            "model",
            "n",
            "mae",
            "rmse",
            "r2",
        ]
    ].sort_values(
        ["outcome_label", "held_out_draft_year", "model"]
    ),
    digits=4,
)

sample_md = markdown_table(
    sample_sizes,
    digits=0,
)

association_md = markdown_table(
    top_associations[
        [
            "outcome_label",
            "predictor",
            "group",
            "n",
            "spearman_rho",
        ]
    ],
    digits=4,
)


report = f"""# WR-MovementIQ: Competition Research Report

## Beyond the 40-Yard Dash: Linking NFL Combine Receiver Movement Mechanics to Regular-Season Performance Using Player Tracking Data

**Project:** WR-MovementIQ  
**Competition:** NFL Big Data Bowl 2027  
**Report status:** Evidence-based research draft  
**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

---

# Abstract

This study investigates the extent to which movement characteristics measured during NFL Combine receiver-specific drills are associated with regular-season in-game movement performance among wide receivers.

The research combines traditional Combine measurements with tracking-derived movement characteristics from Combine drills and NFL regular-season player tracking. Game movement outcomes include mean speed, peak speed, 95th-percentile speed, acceleration summaries and path efficiency.

The primary modeling dataset contains 93 player-season observations from 54 receivers. Four model specifications are compared: an intercept-only baseline, traditional Combine predictors, tracking-derived Combine predictors and a combined predictor set. Predictive evaluation uses receiver-grouped nested cross-validation, with a separate leave-one-draft-cohort-out analysis. Receiver-level bootstrap procedures quantify uncertainty in selected comparisons.

The results show that traditional Combine measurements, particularly the 40-yard dash, have meaningful marginal associations with in-game speed. Tracking-derived Combine movement features also show modest associations with selected in-game speed outcomes. However, the evidence does not establish a consistent, statistically convincing improvement in absolute predictive accuracy from adding tracking-derived features. Some combined models have favourable point estimates for high-speed outcomes, but the receiver-level bootstrap confidence intervals include zero.

The findings support further investigation of receiver movement mechanics while highlighting sample-size constraints, predictor collinearity, cohort differences and the distinction between association and prediction.

**Keywords:** NFL, wide receivers, player tracking, NFL Combine, movement mechanics, Ridge regression, nested cross-validation, receiver analytics.

---

# 1. Introduction

The NFL Combine provides standardized measurements of athletic performance. Traditional tests such as the 40-yard dash, vertical jump and broad jump are widely used to characterize athletic attributes. These tests, however, capture only selected dimensions of movement performance.

Receiver-specific Combine drills provide a richer opportunity to examine movement mechanics. Tracking data can describe speed, acceleration, directional changes and path characteristics during structured drills. NFL player tracking data also provide observations of movement during regular-season play.

This project investigates whether Combine movement characteristics correspond to measurable in-game movement performance.

The study does not assume that a Combine drill reproduces the demands of an NFL play. Instead, it tests whether structured athletic measurements contain information associated with selected game movement summaries.

## 1.1 Research problem

Traditional Combine metrics are convenient and standardized, but they may not fully characterize the multidimensional movement demands of NFL receiving.

The central empirical problem is whether tracking-derived Combine features provide useful information beyond traditional Combine measurements.

## 1.2 Research question

To what extent do movement characteristics measured during NFL Combine receiver-specific drills explain variation in wide receivers' in-game movement characteristics during regular-season NFL plays?

## 1.3 Research objectives

1. Construct receiver-level and player-season movement features from Combine and NFL tracking data.
2. Quantify associations between Combine movement characteristics and in-game movement outcomes.
3. Compare traditional, tracking-derived and combined predictor specifications.
4. Evaluate performance using receiver-grouped nested cross-validation.
5. Examine cohort-specific performance using leave-one-draft-cohort-out validation.
6. Assess uncertainty, feature sensitivity and coefficient stability.

## 1.4 Research hypotheses

- **H1:** Traditional Combine measurements are associated with selected in-game movement outcomes.
- **H2:** Tracking-derived Combine movement features are associated with selected in-game movement outcomes.
- **H3:** Combining traditional and tracking-derived features may improve prediction relative to traditional measurements alone.
- **H4:** The magnitude and consistency of predictive performance vary across held-out draft cohorts.

These are empirical hypotheses. The analysis does not presume that any will be supported.

---

# 2. Data and provenance

## 2.1 Data sources

The project uses the following data files supplied within the competition research environment:

- `combine_results.csv`
- `combine_tracking.csv`
- `game_tracking_2023.csv`
- `game_tracking_2024.csv`
- `game_tracking_2025.csv`
- `player_play.csv`
- `players.csv`
- `games.csv`
- `player_career_successes.csv`

The files serve different roles in the pipeline, including Combine measurements, Combine drill tracking, NFL game tracking, player/play metadata, player identity and draft information.

The analysis uses player identifiers to connect Combine and game records. Data provenance and intermediate audit outputs are maintained in the project.

## 2.2 Population

The analytical population is focused on wide receivers, including the WR and SLOT_WR categories where specified by the pipeline.

The project does not assume that results for wide receivers generalize to other positions.

## 2.3 Research unit

The primary modeling unit is the player-season.

Multiple player-season observations can belong to the same receiver. Therefore, validation groups observations by receiver to reduce information leakage between training and test folds.

## 2.4 Eligibility

The core sample applies the project-defined receiver-play eligibility requirement and complete-case predictor/outcome requirements.

The primary frozen sample contains:

- **93 player-season observations**
- **54 unique receivers**
- **5 outer validation folds**

No imputation was introduced to force incomplete records into the core modeling sample.

The threshold sensitivity analysis is retained as a separate robustness assessment.

### Eligibility sample sensitivity

{sample_md}

---

# 3. Feature engineering

## 3.1 Traditional Combine predictors

The traditional predictor specification uses:

- 40-yard dash time (`forty`)
- Vertical jump (`vertical`)
- Broad jump (`broad_jump`)

These measurements provide standardized athletic benchmarks.

## 3.2 Tracking-derived Combine predictors

The tracking specification uses summaries of Combine drill movement:

- Mean speed
- Peak speed
- 95th-percentile speed
- Mean acceleration
- Peak acceleration
- Path efficiency

The project constructs attempt-level features before aggregating information across drills and player profiles.

Tracking-derived variables are treated according to their validated data definitions. Acceleration magnitude is not automatically interpreted as signed braking or deceleration.

## 3.3 Game movement outcomes

The primary outcomes are:

- Mean game speed
- Peak game speed
- 95th-percentile game speed
- Mean game acceleration
- Peak game acceleration
- Game path efficiency

The outcomes summarize in-game tracking observations at the player-season level.

They are movement descriptors, not direct measures of receiving skill, route-running quality or overall player value.

---

# 4. Modeling methodology

## 4.1 Model specifications

Four specifications are compared.

| Model | Predictors |
|---|---|
| Baseline | Intercept-only reference |
| Traditional | 40-yard dash, vertical jump, broad jump |
| Tracking | Combine tracking-derived movement features |
| Combined | Traditional and tracking-derived predictors |

## 4.2 Ridge regression

Ridge regression is used to estimate predictive relationships while regularizing coefficients.

Predictors are standardized within a modeling pipeline. Hyperparameter selection is conducted using training data only.

The alpha search grid is:

`[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]`

The selected regularization strength is determined by inner validation MAE.

## 4.3 Nested cross-validation

The primary evaluation uses five outer GroupKFold folds grouped by receiver identifier.

Inner validation is also grouped by receiver. Hyperparameter selection occurs within the outer training data.

This design is intended to prevent observations from the same receiver from appearing in both the training and test portions of a fold.

The reported out-of-fold metrics are predictive estimates under this validation design. They are not in-sample fit statistics.

## 4.4 Leave-one-draft-cohort-out validation

The LOCO analysis withholds one draft cohort at a time.

The available held-out cohorts are 2023, 2024 and 2025. Their test sample sizes differ, so cohort-specific estimates have different levels of precision.

## 4.5 Evaluation metrics

- **MAE:** Average absolute prediction error.
- **RMSE:** Square-root average squared prediction error, more sensitive to large errors.
- **R²:** Performance relative to the test-set mean reference.
- **Spearman correlation:** Rank association between predicted and observed values where reported.

The receiver-level bootstrap provides uncertainty estimates for paired model comparisons. The bootstrap estimand is not interchangeable with the pooled player-season OOF metric.

---

# 5. Results

## 5.1 Nested cross-validation performance

The following table reports the primary nested-CV results.

{nested_md}

The baseline is an important reference. A more complex specification should not be considered useful merely because it produces a fitted relationship. It must demonstrate improved out-of-fold predictive performance.

Results differ across movement outcomes. The combined model has favourable point estimates for some high-speed outcomes, but does not consistently improve performance across all outcomes.

## 5.2 Primary speed-outcome uncertainty

The receiver-level bootstrap compares each evaluated model against the baseline. Negative MAE differences favour the evaluated model.

{decision_md}

For the 95th-percentile and peak-speed outcomes, the combined specification has favourable point estimates in the reported receiver-level comparisons. However, the confidence intervals include zero.

Consequently, the results do not establish a reliable improvement in absolute predictive accuracy.

## 5.3 Draft-cohort generalization

{loco_md}

The LOCO results vary across cohorts and outcomes. The 2025 held-out cohort contains fewer test player-seasons than the earlier cohorts. Its estimates should therefore be interpreted cautiously.

The available evidence does not justify a claim of robust generalization to unseen future draft cohorts.

## 5.4 Univariate predictor associations

The strongest marginal associations include:

{association_md}

The 40-yard dash shows a negative association with selected in-game speed outcomes: faster dash times tend to correspond to higher game speed summaries.

Combine tracking-derived speed features also show positive marginal associations with selected in-game speed outcomes.

These are univariate relationships. They do not establish independent predictor effects after adjustment for other variables.

## 5.5 Feature ablation

Feature ablation compares full and reduced tracking-derived predictor specifications.

The results are retained in `tables/tracking_feature_ablation.csv`.

The ablation analysis does not support a simplistic conclusion that one feature is universally essential. Predictor redundancy and correlations between speed summaries complicate attribution.

In particular, Combine peak speed and Combine P95 speed are strongly correlated. Their information may overlap.

## 5.6 Stability and interpretation

Coefficient stability varies across outer folds. Strong regularization is selected for some outcomes and model specifications.

These findings indicate that the fitted coefficients should not be interpreted as stable independent effects.

The study therefore emphasizes out-of-fold prediction and uncertainty rather than assigning causal or mechanistic meaning to individual Ridge coefficients.

---

# 6. Discussion

## 6.1 Traditional athletic benchmarks retain value

The 40-yard dash has a meaningful marginal relationship with selected in-game speed summaries. This supports the continued relevance of standardized speed testing as one component of receiver athletic evaluation.

However, a speed test does not capture every aspect of receiver movement.

## 6.2 Tracking-derived Combine features contain some signal

Tracking-derived speed summaries show marginal associations with game speed outcomes. This suggests that structured drill tracking can provide additional descriptive information about receiver movement.

The results are not sufficient to establish that tracking-derived features consistently improve absolute prediction beyond traditional measurements.

## 6.3 Combined models require cautious interpretation

Some combined models have better point estimates for high-speed outcomes. Nevertheless, receiver-level bootstrap intervals include zero, and fold-level results are not uniformly favourable.

A lower point estimate alone is not enough to claim an established predictive gain.

## 6.4 Prediction is not causation

The study is observational. Combine movement measurements are not randomly assigned, and player movement is influenced by many factors that are not fully represented in this analysis.

The findings must not be interpreted as evidence that a particular Combine drill characteristic causes better NFL performance.

---

# 7. Limitations

1. **Sample size:** The core analysis contains 93 player-seasons from 54 receivers.
2. **Complete-case selection:** Requiring complete predictors and outcomes reduces the available sample.
3. **Cohort imbalance:** The LOCO test cohorts have unequal sample sizes.
4. **Predictor collinearity:** Some tracking-derived speed features are strongly correlated.
5. **Coefficient instability:** Coefficient estimates vary across validation folds.
6. **Outcome scope:** The outcomes are movement summaries, not comprehensive measures of receiver effectiveness.
7. **Observational design:** The analysis does not identify causal effects.
8. **External validity:** Findings from the available cohorts should not automatically be generalized to all NFL receivers or future seasons.
9. **Measurement interpretation:** Tracking variables must be interpreted using their documented definitions and validated transformations.

---

# 8. Scientific contribution

The contribution of WR-MovementIQ is a reproducible framework for connecting structured Combine drill tracking with regular-season player movement.

The project moves beyond a single athletic test by constructing multidimensional movement summaries and evaluating them against in-game tracking outcomes.

It also emphasizes receiver-grouped validation, cohort holdout analysis, bootstrap uncertainty and independent reconstruction audits.

The evidence suggests that tracking-derived Combine features may contain useful movement information. However, their incremental predictive value beyond traditional measurements remains uncertain in the available sample.

---

# 9. Conclusion

This study examined whether NFL Combine receiver movement characteristics are associated with regular-season in-game movement performance.

The results show that traditional Combine measurements, especially the 40-yard dash, have meaningful marginal associations with game speed. Tracking-derived Combine movement features also show modest associations with selected game speed outcomes.

However, the evidence does not establish consistent incremental predictive improvement from tracking-derived features. Combined models show some favourable point estimates for P95 and peak game speed, but receiver-level bootstrap confidence intervals include zero.

The most defensible conclusion is that Combine drill tracking provides a promising additional description of receiver movement, but its incremental predictive value requires further validation using larger samples and additional cohorts.

---

# 10. Reproducibility and audit trail

The report is generated from the validated Notebook 11 synthesis tables and Notebook 12 publication figures.

The underlying modeling and validation outputs are retained in their original project directories.

No model fitting or prediction generation is performed in this report notebook.

## Key project outputs

- `reports/final_results_synthesis/`
- `reports/nested_ridge_modeling/`
- `reports/fold_stratified_receiver_validation/`
- `reports/sample_eligibility_sensitivity/`
- `reports/robustness_sensitivity/`
- `reports/predictor_interpretation/`
- `reports/publication_visualizations/`

## Publication figures

{figures_markdown}

---

# 11. Further research

Future work should investigate:

1. Larger receiver cohorts and additional NFL seasons.
2. More detailed route-specific movement outcomes.
3. Separation of acceleration magnitude and directional braking where measurement definitions permit.
4. Temporal and spatial route-phase features.
5. External validation on future cohorts.
6. Comparison with alternative regularized and nonlinear models under the same leakage-safe validation design.
7. Whether tracking-derived features improve decisions in a defined scouting or player-development task.

These are proposed extensions, not findings established by the current analysis.

---

# 12. Research integrity statement

This report distinguishes marginal association, out-of-fold predictive performance and uncertainty.

It does not claim causal effects. It does not claim that a favourable point estimate establishes a reliable model improvement. It preserves the frozen analysis sample and validated model outputs.

**End of report.**
"""

REPORT_PATH = OUT / "WR-MovementIQ_Competition_Research_Report.md"

REPORT_PATH.write_text(
    report,
    encoding="utf-8",
)

print("Research report created:")
print(REPORT_PATH)
print("Report size:", REPORT_PATH.stat().st_size, "bytes")
print("Report characters:", len(report))
'''))


cells.append(markdown("""## 12. Export final research tables and source provenance

All report tables and source files are inventoried so that the report can be traced back to the validated artifacts.
"""))


cells.append(code(r'''
# ============================================================
# 12. SOURCE INVENTORY AND MANIFEST
# ============================================================

source_inventory = []

for name, path in FILES.items():

    source_inventory.append({
        "artifact": name,
        "filename": path.name,
        "relative_path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": len(DATA[name]),
        "columns": len(DATA[name].columns),
    })

source_inventory = pd.DataFrame(source_inventory)

source_inventory.to_csv(
    TABLE_DIR / "report_source_inventory.csv",
    index=False,
)

# Check referenced publication images.
for record in figure_records:

    for field in ["png_path", "pdf_path"]:

        path = ROOT / record[field]

        if not path.is_file():
            raise FileNotFoundError(path)

# Report checks.
report_text = REPORT_PATH.read_text(encoding="utf-8")

required_sections = [
    "# 1. Introduction",
    "# 2. Data and provenance",
    "# 3. Feature engineering",
    "# 4. Modeling methodology",
    "# 5. Results",
    "# 6. Discussion",
    "# 7. Limitations",
    "# 8. Scientific contribution",
    "# 9. Conclusion",
    "# 10. Reproducibility and audit trail",
]

missing_sections = [
    section
    for section in required_sections
    if section not in report_text
]

if missing_sections:
    raise AssertionError(
        f"Missing report sections: {missing_sections}"
    )

if len(figure_records) != 12:
    raise AssertionError(
        f"Expected 12 figures, found {len(figure_records)}"
    )

manifest = {
    "project": "WR-MovementIQ",
    "notebook": "13_competition_research_report.ipynb",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version,
    "platform": platform.platform(),
    "primary_player_seasons": PRIMARY_PLAYER_SEASONS,
    "primary_receivers": PRIMARY_RECEIVERS,
    "outer_folds": OUTER_FOLDS,
    "source_artifact_count": len(source_inventory),
    "figure_count": len(figure_records),
    "report_file": str(REPORT_PATH.relative_to(ROOT)),
    "report_bytes": REPORT_PATH.stat().st_size,
    "report_sha256": sha256(REPORT_PATH),
    "source_inventory": source_inventory.to_dict(
        orient="records"
    ),
    "model_training_performed": False,
    "predictions_modified": False,
    "primary_sample_modified": False,
    "report_section_audit": "PASS",
    "figure_reference_audit": "PASS",
    "status": "PASS",
}

manifest_path = OUT / "competition_report_manifest.json"

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("Source inventory:", len(source_inventory))
print("Figure references:", len(figure_records))
print("Report sections:", len(required_sections))
print("Report bytes:", REPORT_PATH.stat().st_size)
print("Manifest:", manifest_path)
'''))


cells.append(markdown("""## 13. Final audit

A successful run must create the report, all source inventory records and the reproducibility manifest without errors.
"""))


cells.append(code(r'''
# ============================================================
# 13. FINAL AUDIT
# ============================================================

assert REPORT_PATH.is_file()
assert REPORT_PATH.stat().st_size > 10000

assert manifest_path.is_file()
assert len(source_inventory) == 10
assert len(figure_records) == 12

loaded_manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

assert loaded_manifest["status"] == "PASS"
assert loaded_manifest["model_training_performed"] is False
assert loaded_manifest["predictions_modified"] is False
assert loaded_manifest["primary_sample_modified"] is False

print("=" * 75)
print("NOTEBOOK 13 — COMPETITION RESEARCH REPORT COMPLETE")
print("=" * 75)

print("Report:", REPORT_PATH)
print("Report size:", REPORT_PATH.stat().st_size)
print("Source artifacts:", len(source_inventory))
print("Figures referenced:", len(figure_records))
print("Primary sample:", PRIMARY_PLAYER_SEASONS)
print("Receivers:", PRIMARY_RECEIVERS)
print("Outer folds:", OUTER_FOLDS)
print("Model retraining:", "NOT PERFORMED")
print("Sample modifications:", "NONE")
print("Audit:", "PASS")
print("Manifest:", manifest_path)
'''))


for index, cell in enumerate(cells):
    cell["id"] = f"notebook13-cell-{index:02d}"


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

OUTPUT.write_text(
    json.dumps(notebook, indent=2),
    encoding="utf-8",
)

print("=" * 70)
print("NOTEBOOK 13 GENERATED")
print("=" * 70)
print("Notebook:", OUTPUT)
print("Cells:", len(cells))
print("Project root:", ROOT)
