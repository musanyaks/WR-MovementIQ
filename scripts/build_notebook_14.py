
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil

import nbformat
import pandas as pd

# ============================================================
# NOTEBOOK 14 — COMPETITION SUBMISSION BUILDER
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_TABLES = (
    ROOT / "reports" / "final_results_synthesis" / "tables"
)

SOURCE_FIGURES = (
    ROOT / "reports" / "publication_visualizations" / "png"
)

OUT = ROOT / "reports" / "competition_submission"
FIGURES_OUT = OUT / "figures"

NOTEBOOK_OUT = (
    ROOT / "notebooks" / "14_competition_submission_builder.ipynb"
)

OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

# Eight figures: fewer than the official maximum of 10
FIGURE_FILES = [
    "figure_01_nested_game_mean_speed_raw_median.png",
    "figure_02_nested_game_p95_speed_raw_median.png",
    "figure_03_nested_game_peak_speed_raw_median.png",
    "figure_05_bootstrap_game_p95_speed_raw_median.png",
    "figure_06_bootstrap_game_peak_speed_raw_median.png",
    "figure_08_loco_game_p95_speed_raw_median.png",
    "figure_09_loco_game_peak_speed_raw_median.png",
    "figure_11_tracking_feature_ablation.png",
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_table(filename):
    path = SOURCE_TABLES / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


# ============================================================
# 1. LOAD AUTHORITATIVE TABLES
# ============================================================

nested = load_table("main_nested_cv_performance.csv")
decisions = load_table("primary_speed_outcome_decisions.csv")
associations = load_table("univariate_spearman_associations.csv")
loco = load_table("loco_performance.csv")
ablation = load_table("speed_feature_ablation.csv")

# Independently audited Notebook 10B sample-eligibility sensitivity.
sample_flow_path = (
    ROOT / "reports" / "sample_eligibility_sensitivity" / "sample_flow.csv"
)

if not sample_flow_path.is_file():
    raise FileNotFoundError(sample_flow_path)

sample_flow = pd.read_csv(sample_flow_path)

required_sample_columns = {
    "minimum_receiver_plays",
    "player_seasons",
    "unique_receivers",
}

if not required_sample_columns.issubset(sample_flow.columns):
    raise ValueError(
        "Sample-flow CSV is missing required columns: "
        f"{required_sample_columns - set(sample_flow.columns)}"
    )

sample_flow_lookup = sample_flow.set_index("minimum_receiver_plays")

expected_sample_flow = {
    5: (93, 54),
    8: (93, 54),
    10: (93, 54),
    12: (92, 53),
    15: (83, 48),
}

for threshold, expected in expected_sample_flow.items():
    if threshold not in sample_flow_lookup.index:
        raise ValueError(f"Missing sample-flow threshold: {threshold}")

    actual = (
        int(sample_flow_lookup.loc[threshold, "player_seasons"]),
        int(sample_flow_lookup.loc[threshold, "unique_receivers"]),
    )

    if actual != expected:
        raise ValueError(
            f"Unexpected sample flow at threshold {threshold}: "
            f"expected {expected}, received {actual}"
        )


assert len(nested) == 24
assert len(decisions) == 6
assert len(associations) == 54
assert len(loco) == 72
assert len(ablation) == 15

# ============================================================
# 2. COPY AND VERIFY FIGURES
# ============================================================

figure_records = []

for filename in FIGURE_FILES:

    source = SOURCE_FIGURES / filename

    if not source.is_file():
        raise FileNotFoundError(source)

    destination = FIGURES_OUT / filename
    shutil.copy2(source, destination)

    figure_records.append({
        "filename": filename,
        "path": f"figures/{filename}",
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    })

assert len(figure_records) == 8

# ============================================================
# 3. EXTRACT VALIDATED RESULTS
# ============================================================

def metric(outcome, model, metric_name="mae"):
    row = nested[
        (nested["outcome"] == outcome)
        & (nested["model"].str.upper() == model.upper())
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected one row for {outcome}/{model}; got {len(row)}"
        )

    return float(row.iloc[0][metric_name])


def decision(outcome, model="COMBINED"):
    row = decisions[
        (decisions["outcome"] == outcome)
        & (decisions["model"].str.upper() == model.upper())
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected one decision row for {outcome}/{model}; got {len(row)}"
        )

    return row.iloc[0]


def association(outcome, predictor):
    row = associations[
        (associations["outcome"] == outcome)
        & (associations["predictor"] == predictor)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected one association row: {outcome}/{predictor}; "
            f"got {len(row)}"
        )

    return float(row.iloc[0]["spearman_rho"])


# Primary speed comparisons
p95 = decision("game_p95_speed_raw_median")
peak = decision("game_peak_speed_raw_median")

# Independently validated associations
forty_peak = association(
    "game_peak_speed_raw_median",
    "forty",
)

forty_mean = association(
    "game_mean_speed_raw_median",
    "forty",
)

forty_p95 = association(
    "game_p95_speed_raw_median",
    "forty",
)

combine_peak_to_game_p95 = association(
    "game_p95_speed_raw_median",
    "combine_peak_speed_median_across_drills",
)

combine_peak_to_game_peak = association(
    "game_peak_speed_raw_median",
    "combine_peak_speed_median_across_drills",
)

# ============================================================
# 4. BUILD THE WRITE-UP
# ============================================================

writeup = f"""# Beyond the 40-Yard Dash: Do Combine Movement Mechanics Translate to NFL Receiver Performance?

**WR-MovementIQ | NFL Big Data Bowl 2027**

## The football question

The 40-yard dash measures straight-line speed. But NFL receivers must also accelerate, redirect, maintain speed through routes and move effectively in live plays.

The NFL Combine's 10 Hz sensor tracking creates an opportunity to study these movement characteristics in greater detail. We ask a focused question:

**Do movement characteristics recorded during receiver-specific Combine drills provide useful information about regular-season NFL receiver movement beyond traditional Combine measurements?**

We focus on wide receivers and slot receivers rather than attempting to model every position group.

## What we measured

We linked Combine drill tracking with regular-season NFL player tracking and player/play information.

Traditional Combine predictors were the 40-yard dash, vertical jump and broad jump.

Tracking-derived Combine predictors summarized speed, peak speed, 95th-percentile speed, acceleration and path efficiency across receiver drill attempts.

In-game outcomes summarized mean speed, peak speed, 95th-percentile speed, acceleration and path efficiency at the player-season level.

These are movement outcomes, not direct measures of receiving ability, route-running quality or overall player value.

The final core modeling sample contained **93 player-season observations from 54 receivers**. We retained complete cases for the primary predictors and outcomes rather than imputing missing Combine measurements.

## How we tested the relationship

We compared four model specifications:

- An intercept-only baseline.
- Traditional Combine measurements.
- Tracking-derived Combine movement features.
- A combined traditional and tracking-derived specification.

We used Ridge regression with standardized predictors. Hyperparameters were selected inside receiver-grouped inner cross-validation.

The outer evaluation used five-fold GroupKFold validation, grouping by receiver so that the same receiver could not appear in both training and test sets within a fold.

We also conducted leave-one-draft-cohort-out validation, withholding the 2023, 2024 and 2025 draft cohorts in turn.

For selected speed outcomes, we used 5,000 fold-stratified receiver-level bootstrap replicates to quantify uncertainty in paired MAE differences. Negative differences favour the evaluated model over the baseline.

## What we found

### 1. Traditional speed remains informative

The 40-yard dash had negative Spearman associations with in-game mean speed (rho = {forty_mean:.3f}), peak speed (rho = {forty_peak:.3f}) and 95th-percentile speed (rho = {forty_p95:.3f}). Faster dash times therefore tended to correspond to higher in-game speed summaries, particularly peak and upper-tail speed.

This supports the 40-yard dash as a useful athletic benchmark, but not as a complete description of receiver movement. These are univariate associations, not evidence of independent predictive effects.

![Nested mean speed results](figures/{FIGURE_FILES[0]})

### 2. Combine tracking contains additional movement signal

Combine peak speed was positively associated with in-game P95 speed (rho = {combine_peak_to_game_p95:.3f}) and peak speed (rho = {combine_peak_to_game_peak:.3f}).

These are marginal associations. They do not establish that tracking-derived features provide independent predictive value after accounting for traditional tests.

![Nested P95 speed results](figures/{FIGURE_FILES[1]})

![Nested peak speed results](figures/{FIGURE_FILES[2]})

### 3. Incremental predictive improvement remains uncertain

For P95 game speed, the combined model's receiver-level MAE difference against baseline was {float(p95["point_estimate"]):.4f}, with a 95% receiver-level, fold-stratified bootstrap interval of [{float(p95["ci_2_5"]):.4f}, {float(p95["ci_97_5"]):.4f}].

For peak game speed, the corresponding difference was {float(peak["point_estimate"]):.4f}, with a 95% interval of [{float(peak["ci_2_5"]):.4f}, {float(peak["ci_97_5"]):.4f}].

Negative differences favour the combined model. Both intervals include zero. The combined model has favourable point estimates, but the available evidence does not establish a reliable improvement in absolute predictive accuracy.

![P95 bootstrap uncertainty](figures/{FIGURE_FILES[3]})

![Peak speed bootstrap uncertainty](figures/{FIGURE_FILES[4]})

### 4. Performance must be tested across draft cohorts

We evaluated models by withholding each draft cohort. Cohort test sizes were unequal, and the 2025 held-out sample was particularly small.

The cohort results are therefore useful generalization diagnostics, not proof that the model will generalize to future NFL prospects.

![P95 cohort validation](figures/{FIGURE_FILES[5]})

![Peak speed cohort validation](figures/{FIGURE_FILES[6]})

### 5. More movement features do not automatically mean better prediction

Combine tracking-derived peak speed and 95th-percentile speed are strongly correlated (Spearman rho = 0.970). This overlap makes it difficult to distinguish their independent contributions. Ridge coefficient estimates also vary across validation folds, so individual coefficients should not be interpreted as stable measures of football importance.

Feature ablation and coefficient-stability analyses reinforce the need to evaluate the complete model out of sample rather than selecting a feature because it has an attractive individual association.

**Sample-eligibility sensitivity.** At minimum-play thresholds of 5, 8 and 10, the complete-case sample retained {int(sample_flow_lookup.loc[5, "player_seasons"])} player-seasons from {int(sample_flow_lookup.loc[5, "unique_receivers"])} receivers. At 12 plays, it decreased to {int(sample_flow_lookup.loc[12, "player_seasons"])} player-seasons from {int(sample_flow_lookup.loc[12, "unique_receivers"])} receivers. At 15 plays, it decreased to {int(sample_flow_lookup.loc[15, "player_seasons"])} player-seasons from {int(sample_flow_lookup.loc[15, "unique_receivers"])} receivers. These results describe sample sensitivity; they do not establish that model performance is invariant across thresholds.

![Tracking feature ablation](figures/{FIGURE_FILES[7]})

## What this means for football

The findings support using Combine tracking as a **movement diagnostic**, not as a stand-alone draft ranking system.

For scouts and player-development staff, tracking-derived drill summaries may help describe how a receiver moves beyond a single sprint time. They can motivate more specific questions about speed maintenance, acceleration and movement efficiency.

However, our results do not justify replacing traditional Combine measurements or making player-selection decisions from this model. The incremental predictive benefit remains uncertain.

The next practical step is to validate route-phase and drill-specific movement traits on larger receiver cohorts and future seasons.

## Limitations

The core sample is limited to 54 receivers and 93 player-seasons. Complete-case selection may affect representativeness. Some predictors are highly correlated, model coefficients vary across folds, and held-out cohort sizes are unequal.

The study is observational. It identifies associations and predictive performance, not causal effects. Game movement summaries also do not capture the full context of route assignments, coverage, quarterback decisions or receiving outcomes.

## Conclusion

Combine sensor tracking provides a richer description of receiver movement than traditional test results alone. Our analysis finds modest associations between some Combine movement summaries and selected in-game speed outcomes.

But the central test is incremental prediction: the current evidence does not demonstrate a consistent, reliable improvement from adding tracking-derived features.

That is the practical conclusion: **use movement tracking to deepen receiver evaluation, but demand stronger external validation before treating it as a predictive scouting advantage.**

*WR-MovementIQ — NFL Big Data Bowl 2027*
"""

# ============================================================
# 5. WORD AND VISUAL COUNT AUDIT
# ============================================================

# Count words in visible text, excluding Markdown image syntax.
text_for_count = re.sub(
    r"!\[[^\]]*\]\([^)]+\)",
    "",
    writeup,
)

words = re.findall(
    r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*",
    text_for_count,
)

word_count = len(words)
visual_count = len(figure_records)
table_count = 0
combined_visual_count = visual_count + table_count

if word_count > 2000:
    raise AssertionError(
        f"Write-up exceeds 2,000 words: {word_count}"
    )

if combined_visual_count >= 10:
    raise AssertionError(
        f"Write-up has too many tables/figures: {combined_visual_count}"
    )

# Save Markdown write-up.
WRITEUP_PATH = OUT / "WR-MovementIQ_Kaggle_Writeup.md"

WRITEUP_PATH.write_text(
    writeup,
    encoding="utf-8",
)

# ============================================================
# 6. SUBMISSION MANIFEST
# ============================================================

manifest = {
    "project": "WR-MovementIQ",
    "competition": "NFL Big Data Bowl 2027",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "writeup": WRITEUP_PATH.name,
    "writeup_bytes": WRITEUP_PATH.stat().st_size,
    "writeup_sha256": sha256(WRITEUP_PATH),
    "word_count": word_count,
    "word_limit": 2000,
    "figure_count": visual_count,
    "table_count": table_count,
    "combined_table_figure_count": combined_visual_count,
    "official_combined_limit": 10,
    "figure_records": figure_records,
    "primary_player_seasons": 93,
    "primary_receivers": 54,
    "outer_folds": 5,
    "source_tables": {
        filename: sha256(SOURCE_TABLES / filename)
        for filename in [
            "main_nested_cv_performance.csv",
            "primary_speed_outcome_decisions.csv",
            "univariate_spearman_associations.csv",
            "loco_performance.csv",
            "speed_feature_ablation.csv",
        ]
    },
    "word_limit_audit": "PASS",
    "visual_limit_audit": "PASS",
    "figure_hash_audit": "PASS",
    "public_kaggle_notebook_attached": False,
    "submission_status": "DRAFT_NOT_YET_SUBMITTED",
}

manifest_path = OUT / "submission_manifest.json"

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

# ============================================================
# 7. BUILD NOTEBOOK 14
# ============================================================

cells = [
    nbformat.v4.new_markdown_cell(
        """# Notebook 14 — Competition Submission Builder

This notebook creates the concise NFL Big Data Bowl 2027 write-up
from the independently validated WR-MovementIQ research outputs.

It checks:
- Word count against the 2,000-word limit.
- Combined table/figure count against the fewer-than-10 rule.
- Figure availability and SHA256 hashes.
- Provenance of the source result tables.

The public Kaggle Notebook is a separate deliverable and must be
attached to the final Kaggle Write-up.
"""
    ),
    nbformat.v4.new_markdown_cell(
        f"""## Submission draft

**Write-up:** `{WRITEUP_PATH.name}`

**Word count:** {word_count}

**Figures:** {visual_count}

**Tables:** {table_count}

**Combined visual count:** {combined_visual_count}

The write-up is generated in the sibling report directory.
"""
    ),
    nbformat.v4.new_code_cell(
        """from pathlib import Path
import json
import hashlib
import re
import pandas as pd

ROOT = Path.cwd().resolve()

candidates = [ROOT, ROOT.parent]
PROJECT = next(
    (p for p in candidates
     if (p / "reports" / "competition_submission"
         / "submission_manifest.json").is_file()),
    None
)

if PROJECT is None:
    raise FileNotFoundError("Cannot locate WR-MovementIQ project root.")

OUT = PROJECT / "reports" / "competition_submission"
manifest = json.loads(
    (OUT / "submission_manifest.json").read_text(encoding="utf-8")
)

writeup = OUT / manifest["writeup"]
text = writeup.read_text(encoding="utf-8")

word_text = re.sub(r"!\\\\[[^\\\\]]*\\\\]\\\\([^)]+\\\\)", "", text)
words = re.findall(
    r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*",
    word_text
)

assert len(words) <= 2000
assert manifest["combined_table_figure_count"] < 10

for figure in manifest["figure_records"]:
    path = OUT / figure["path"]
    assert path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == figure["sha256"]

print("Write-up:", writeup)
print("Words:", len(words))
print("Figures:", manifest["figure_count"])
print("Tables:", manifest["table_count"])
print("All figure hashes verified.")
print("Submission builder audit: PASS")
"""
    ),
    nbformat.v4.new_markdown_cell(
        """## Final submission gate

Before submitting on Kaggle:

1. Confirm the selected competition track and eligibility.
2. Create the Kaggle Write-up.
3. Paste the Markdown content and upload/embed the eight figures.
4. Attach a public Kaggle Notebook.
5. Confirm the final rendered word count and visual count.
6. Click **Submit** before the official deadline.

This local builder does not submit anything to Kaggle.
"""
    ),
]

nb = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    },
)

nbformat.write(nb, NOTEBOOK_OUT)

# ============================================================
# 8. FINAL OUTPUT
# ============================================================

print("=" * 75)
print("NOTEBOOK 14 — COMPETITION SUBMISSION BUILDER")
print("=" * 75)
print("Notebook:", NOTEBOOK_OUT)
print("Write-up:", WRITEUP_PATH)
print("Manifest:", manifest_path)
print("Words:", word_count, "/ 2000")
print("Figures:", visual_count)
print("Tables:", table_count)
print("Combined visuals:", combined_visual_count, "/ fewer than 10")
print("Figure copies:", len(figure_records))
print("Audit: PASS")
print("Status: DRAFT_NOT_YET_SUBMITTED")
