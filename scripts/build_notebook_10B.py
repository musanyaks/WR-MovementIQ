
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# NOTEBOOK 10B — SAMPLE ELIGIBILITY SENSITIVITY ANALYSIS
# ============================================================

ROOT = Path.cwd()

PRIMARY_FILE = ROOT / "data/features/wr_analysis_dataset_primary.parquet"
CORE_FILE = ROOT / "data/features/wr_analysis_dataset_core_common.parquet"
FOLD_FILE = ROOT / "reports/modeling_design/outer_grouped_fold_assignments.csv"

OUT = ROOT / "reports/sample_eligibility_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [5, 8, 10, 12, 15]
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
N_BOOTSTRAP = 5000
RANDOM_SEED = 2027

# If automatic detection fails, enter the exact play-count column here.
PLAY_COUNT_COL = None

KEYS = ["draft_year", "nfl_id", "season"]

OUTCOMES = [
    "game_mean_speed_raw_median",
    "game_peak_speed_raw_median",
    "game_p95_speed_raw_median",
    "game_mean_acceleration_raw_median",
    "game_peak_acceleration_raw_median",
    "game_path_efficiency_median",
]

TRADITIONAL = [
    "forty",
    "vertical",
    "broad_jump",
]

TRACKING = [
    "combine_mean_speed_median_across_drills",
    "combine_peak_speed_median_across_drills",
    "combine_p95_speed_median_across_drills",
    "combine_mean_acceleration_median_across_drills",
    "combine_peak_acceleration_median_across_drills",
    "combine_path_efficiency_median_across_drills",
]

COMBINED = TRADITIONAL + TRACKING

MODEL_FEATURES = {
    "BASELINE": [],
    "TRADITIONAL": TRADITIONAL,
    "TRACKING_FULL": TRACKING,
    "COMBINED_FULL": COMBINED,
    "COMBINED_NO_PEAK_SPEED": [
        x for x in COMBINED
        if x != "combine_peak_speed_median_across_drills"
    ],
    "COMBINED_NO_P95_SPEED": [
        x for x in COMBINED
        if x != "combine_p95_speed_median_across_drills"
    ],
    "COMBINED_PEAK_SPEED_ONLY": [
        "forty",
        "vertical",
        "broad_jump",
        "combine_peak_speed_median_across_drills",
    ],
    "COMBINED_P95_SPEED_ONLY": [
        "forty",
        "vertical",
        "broad_jump",
        "combine_p95_speed_median_across_drills",
    ],
    "COMBINED_NO_ACCELERATION": [
        x for x in COMBINED
        if "acceleration" not in x
    ],
    "TRACKING_SPEED_ONLY": [
        "combine_mean_speed_median_across_drills",
        "combine_peak_speed_median_across_drills",
        "combine_p95_speed_median_across_drills",
    ],
}

ALPHAS = [
    0.001, 0.01, 0.1, 1, 10,
    100, 1000, 10000, 100000
]

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("=" * 78)
print("NOTEBOOK 10B — SAMPLE ELIGIBILITY SENSITIVITY")
print("=" * 78)

for path in [PRIMARY_FILE, CORE_FILE, FOLD_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

df = pd.read_parquet(PRIMARY_FILE)
core = pd.read_parquet(CORE_FILE)
fold_source = pd.read_csv(FOLD_FILE)

print("\nPrimary dataset:", df.shape)
print("Frozen core dataset:", core.shape)
print("Original fold assignments:", fold_source.shape)

# ------------------------------------------------------------
# 2. DETECT RECEIVER PLAY-COUNT COLUMN
# ------------------------------------------------------------

if PLAY_COUNT_COL is None:
    candidates = [
        "receiver_plays",
        "n_receiver_plays",
        "receiver_play_count",
        "total_receiver_plays",
        "game_plays",
        "n_plays",
        "plays",
        "play_count",
        "number_of_plays",
    ]

    matches = [c for c in candidates if c in df.columns]

    if len(matches) == 1:
        PLAY_COUNT_COL = matches[0]
    elif len(matches) > 1:
        print("\nPossible play-count columns:", matches)
        raise ValueError(
            "More than one play-count column was found. "
            "Set PLAY_COUNT_COL manually near the top of this script."
        )
    else:
        possible = [
            c for c in df.columns
            if "play" in c.lower()
        ]
        print("\nColumns containing 'play':", possible)
        raise ValueError(
            "Could not identify the receiver-play-count column. "
            "Set PLAY_COUNT_COL to the correct column name."
        )

print("\nPlay-count column:", PLAY_COUNT_COL)

# ------------------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required = (
    KEYS
    + [PLAY_COUNT_COL]
    + OUTCOMES
    + COMBINED
)

missing = sorted(set(required) - set(df.columns))

if missing:
    raise ValueError(f"Missing required columns in primary dataset: {missing}")

if not set(KEYS).issubset(core.columns):
    raise ValueError("Frozen core dataset is missing key columns.")

# Normalize keys and numeric fields.
for data in [df, core]:
    data["nfl_id"] = data["nfl_id"].astype(str)
    data["draft_year"] = pd.to_numeric(
        data["draft_year"], errors="coerce"
    )
    data["season"] = pd.to_numeric(
        data["season"], errors="coerce"
    )

df[PLAY_COUNT_COL] = pd.to_numeric(
    df[PLAY_COUNT_COL], errors="coerce"
)

for col in OUTCOMES + COMBINED:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 4. VALIDATE UNIQUE PLAYER-SEASON KEYS
# ------------------------------------------------------------

if df[KEYS].isna().any().any():
    raise ValueError("Primary dataset contains missing player-season keys.")

if df.duplicated(KEYS).any():
    duplicates = df.loc[df.duplicated(KEYS, keep=False), KEYS]
    raise ValueError(
        "Primary dataset contains duplicate player-season keys:\n"
        + duplicates.head(20).to_string(index=False)
    )

# Complete-case sample across all predictors and outcomes.
# This gives all model specifications the same eligible sample
# at each threshold, avoiding model-specific sample composition.
complete_columns = COMBINED + OUTCOMES

df["complete_case"] = df[complete_columns].notna().all(axis=1)

print("\nComplete-case rows:", int(df["complete_case"].sum()))
print(
    "Complete-case receivers:",
    df.loc[df["complete_case"], "nfl_id"].nunique()
)

# ------------------------------------------------------------
# 5. RECONSTRUCT ORIGINAL RECEIVER-TO-FOLD MAP
# ------------------------------------------------------------

if "nfl_id" not in fold_source.columns:
    raise ValueError("Fold assignment file has no nfl_id column.")

fold_col_candidates = [
    "outer_fold",
    "fold",
    "fold_id",
]

fold_col = next(
    (c for c in fold_col_candidates if c in fold_source.columns),
    None
)

if fold_col is None:
    raise ValueError(
        "Could not identify the fold column in the original assignments."
    )

fold_source["nfl_id"] = fold_source["nfl_id"].astype(str)
fold_source[fold_col] = pd.to_numeric(
    fold_source[fold_col], errors="coerce"
)

fold_map_df = fold_source[["nfl_id", fold_col]].dropna().copy()
fold_map_df[fold_col] = fold_map_df[fold_col].astype(int)
# Normalize original fold labels to 1, 2, 3, 4, 5 if necessary.
original_labels = sorted(fold_map_df[fold_col].unique())

if len(original_labels) != N_OUTER_FOLDS:
    raise ValueError(
        f"Expected {N_OUTER_FOLDS} original folds, found {original_labels}"
    )

fold_label_map = {
    old: new
    for new, old in enumerate(original_labels, start=1)
}

fold_map_df[fold_col] = fold_map_df[fold_col].map(fold_label_map)

if fold_map_df["nfl_id"].duplicated().any():
    counts = fold_map_df.groupby("nfl_id")[fold_col].nunique()
    if (counts > 1).any():
        raise ValueError(
            "At least one receiver has multiple original outer folds."
        )

original_fold_map = (
    fold_map_df.drop_duplicates("nfl_id")
    .set_index("nfl_id")[fold_col]
    .to_dict()
)

print(
    "\nReceivers with preserved original folds:",
    len(original_fold_map)
)

# ------------------------------------------------------------
# 6. ASSIGN FOLDS TO ADDITIONAL RECEIVERS
# ------------------------------------------------------------

def assign_receiver_folds(sample):
    """
    Preserve Notebook 07 folds for known receivers.
    Assign new receivers deterministically to the currently
    smallest fold by player-season count.
    No receiver is split across folds.
    """

    receiver_counts = (
        sample.groupby("nfl_id")
        .size()
        .sort_index()
    )

    assignment = {}
    fold_load = {f: 0 for f in range(1, N_OUTER_FOLDS + 1)}

    # Preserve original assignments for receivers in the frozen core.
    for nfl_id, count in receiver_counts.items():
        if nfl_id in original_fold_map:
            fold = int(original_fold_map[nfl_id])

            if fold not in fold_load:
                raise ValueError(
                    f"Unexpected original fold number: {fold}"
                )

            assignment[nfl_id] = fold
            fold_load[fold] += int(count)

    # Add previously unseen receivers to the least-loaded fold.
    new_receivers = [
        nfl_id for nfl_id in receiver_counts.index
        if nfl_id not in assignment
    ]

    for nfl_id in new_receivers:
        fold = min(
            fold_load,
            key=lambda f: (fold_load[f], f)
        )
        assignment[nfl_id] = fold
        fold_load[fold] += int(receiver_counts[nfl_id])

    result = sample[["nfl_id"]].copy()
    result["outer_fold"] = result["nfl_id"].map(assignment)

    if result["outer_fold"].isna().any():
        raise RuntimeError("Some receivers were not assigned an outer fold.")

    receiver_fold_counts = (
        result.groupby("nfl_id")["outer_fold"].nunique()
    )

    if (receiver_fold_counts != 1).any():
        raise RuntimeError("Receiver leakage detected across outer folds.")

    return result["outer_fold"].to_numpy(), fold_load


# ------------------------------------------------------------
# 7. MODEL FITTING AND INNER ALPHA TUNING
# ------------------------------------------------------------

def make_model(alpha):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def tune_alpha(X, y, groups):
    """
    Tune alpha using grouped inner CV.
    All preprocessing occurs inside each training fold.
    """

    unique_groups = pd.Series(groups).nunique()

    if unique_groups < 2:
        raise ValueError("Not enough receiver groups for inner CV.")

    n_splits = min(N_INNER_FOLDS, unique_groups)
    inner_cv = GroupKFold(n_splits=n_splits)

    scores = []

    for alpha in ALPHAS:
        fold_scores = []

        for train_idx, valid_idx in inner_cv.split(
            X, y, groups=groups
        ):
            model = make_model(alpha)

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx]
            )

            pred = model.predict(X.iloc[valid_idx])

            fold_scores.append(
                mean_absolute_error(y.iloc[valid_idx], pred)
            )

        scores.append({
            "alpha": alpha,
            "mean_inner_mae": float(np.mean(fold_scores)),
        })

    tuning = pd.DataFrame(scores).sort_values(
        ["mean_inner_mae", "alpha"]
    )

    return float(tuning.iloc[0]["alpha"]), tuning


# ------------------------------------------------------------
# 8. RUN ONE THRESHOLD
# ------------------------------------------------------------

all_predictions = []
all_fold_metrics = []
all_tuning = []
sample_flow = []
fold_audits = []

for threshold in THRESHOLDS:

    print("\n" + "=" * 78)
    print(f"MINIMUM RECEIVER PLAYS: {threshold}")
    print("=" * 78)

    sample = df[
        (df[PLAY_COUNT_COL] >= threshold)
        & df["complete_case"]
    ].copy()

    sample = sample.sort_values(KEYS).reset_index(drop=True)

    n_rows = len(sample)
    n_receivers = sample["nfl_id"].nunique()

    if n_receivers < N_OUTER_FOLDS + 1:
        warnings.warn(
            f"Threshold {threshold}: too few receivers. Skipping."
        )
        continue

    if sample.duplicated(KEYS).any():
        raise RuntimeError("Duplicate keys found after sample filtering.")

    folds, fold_load = assign_receiver_folds(sample)
    sample["outer_fold"] = folds

    sample_flow.append({
        "minimum_receiver_plays": threshold,
        "player_seasons": n_rows,
        "unique_receivers": n_receivers,
        "draft_years": sample["draft_year"].nunique(),
        "excluded_incomplete_rows": int(
            (
                (df[PLAY_COUNT_COL] >= threshold)
                & (~df["complete_case"])
            ).sum()
        ),
    })

    fold_audits.append({
        "minimum_receiver_plays": threshold,
        "fold_1_player_seasons": int(
            (sample["outer_fold"] == 1).sum()
        ),
        "fold_2_player_seasons": int(
            (sample["outer_fold"] == 2).sum()
        ),
        "fold_3_player_seasons": int(
            (sample["outer_fold"] == 3).sum()
        ),
        "fold_4_player_seasons": int(
            (sample["outer_fold"] == 4).sum()
        ),
        "fold_5_player_seasons": int(
            (sample["outer_fold"] == 5).sum()
        ),
    })

    print("Player-seasons:", n_rows)
    print("Unique receivers:", n_receivers)
    print("Fold player-season counts:", fold_load)

    # --------------------------------------------------------
    # OUTER CROSS-VALIDATION
    # --------------------------------------------------------

    for outcome in OUTCOMES:

        y_all = sample[outcome].astype(float)

        for model_name, feature_list in MODEL_FEATURES.items():

            print(
                f"Threshold {threshold} | {outcome} | {model_name}"
            )

            for outer_fold in range(1, N_OUTER_FOLDS + 1):

                test_mask = sample["outer_fold"] == outer_fold
                train_mask = ~test_mask

                train = sample.loc[train_mask].copy()
                test = sample.loc[test_mask].copy()

                train_groups = train["nfl_id"].to_numpy()

                if train["nfl_id"].nunique() < 2:
                    raise RuntimeError(
                        "Insufficient training receivers for inner CV."
                    )

                y_train = train[outcome].astype(float)
                y_test = test[outcome].astype(float)

                if model_name == "BASELINE":
                    best_alpha = np.nan
                    inner_best_mae = np.nan
                    prediction = np.repeat(
                        float(y_train.median()),
                        len(test)
                    )

                else:
                    X_train = train[feature_list].astype(float)
                    X_test = test[feature_list].astype(float)

                    best_alpha, tuning = tune_alpha(
                        X_train.reset_index(drop=True),
                        y_train.reset_index(drop=True),
                        train_groups
                    )

                    inner_best_mae = float(
                        tuning.loc[
                            tuning["alpha"] == best_alpha,
                            "mean_inner_mae"
                        ].iloc[0]
                    )

                    for _, row in tuning.iterrows():
                        all_tuning.append({
                            "threshold": threshold,
                            "outcome": outcome,
                            "model": model_name,
                            "outer_fold": outer_fold,
                            "candidate_alpha": row["alpha"],
                            "inner_mae": row["mean_inner_mae"],
                            "selected_alpha": (
                                row["alpha"] == best_alpha
                            ),
                        })

                    fitted = make_model(best_alpha)
                    fitted.fit(X_train, y_train)
                    prediction = fitted.predict(X_test)

                mae = mean_absolute_error(y_test, prediction)
                rmse = np.sqrt(
                    mean_squared_error(y_test, prediction)
                )

                r2 = (
                    r2_score(y_test, prediction)
                    if len(test) >= 2
                    else np.nan
                )

                all_fold_metrics.append({
                    "threshold": threshold,
                    "outcome": outcome,
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "train_player_seasons": len(train),
                    "test_player_seasons": len(test),
                    "train_receivers": train["nfl_id"].nunique(),
                    "test_receivers": test["nfl_id"].nunique(),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "best_alpha": best_alpha,
                    "inner_best_mae": inner_best_mae,
                })

                for i, (_, row) in enumerate(test.iterrows()):
                    all_predictions.append({
                        "threshold": threshold,
                        "outcome": outcome,
                        "model": model_name,
                        "nfl_id": row["nfl_id"],
                        "draft_year": row["draft_year"],
                        "season": row["season"],
                        "outer_fold": outer_fold,
                        "actual": float(y_test.iloc[i]),
                        "predicted": float(prediction[i]),
                        "absolute_error": abs(
                            float(y_test.iloc[i]) -
                            float(prediction[i])
                        ),
                        "best_alpha": best_alpha,
                    })


# ------------------------------------------------------------
# 9. BUILD OUTPUT TABLES
# ------------------------------------------------------------

predictions = pd.DataFrame(all_predictions)
fold_metrics = pd.DataFrame(all_fold_metrics)
tuning_results = pd.DataFrame(all_tuning)

if predictions.empty:
    raise RuntimeError("No predictions were generated.")

# Validate expected prediction coverage.
expected_groups = (
    len(THRESHOLDS)
    * len(OUTCOMES)
    * len(MODEL_FEATURES)
)

print("\nGenerated prediction rows:", len(predictions))
print("Generated fold metric rows:", len(fold_metrics))

# Pooled out-of-fold metrics.
pooled_metrics = (
    predictions.groupby(
        ["threshold", "outcome", "model"]
    )
    .apply(
        lambda g: pd.Series({
            "player_seasons": len(g),
            "unique_receivers": g["nfl_id"].nunique(),
            "mae": mean_absolute_error(
                g["actual"], g["predicted"]
            ),
            "rmse": np.sqrt(
                mean_squared_error(
                    g["actual"], g["predicted"]
                )
            ),
            "r2": (
                r2_score(g["actual"], g["predicted"])
                if len(g) >= 2 else np.nan
            ),
        }),
        include_groups=False
    )
    .reset_index()
)

# ------------------------------------------------------------
# 10. RECEIVER-LEVEL PAIRED BOOTSTRAP
# ------------------------------------------------------------

receiver_errors = (
    predictions.groupby(
        [
            "threshold",
            "outcome",
            "model",
            "nfl_id",
            "outer_fold",
        ]
    )["absolute_error"]
    .mean()
    .reset_index(name="receiver_mae")
)

baseline_errors = receiver_errors[
    receiver_errors["model"] == "BASELINE"
][
    [
        "threshold",
        "outcome",
        "nfl_id",
        "outer_fold",
        "receiver_mae",
    ]
].rename(columns={"receiver_mae": "baseline_receiver_mae"})

paired = receiver_errors.merge(
    baseline_errors,
    on=["threshold", "outcome", "nfl_id", "outer_fold"],
    how="inner"
)

paired["mae_difference"] = (
    paired["receiver_mae"]
    - paired["baseline_receiver_mae"]
)

bootstrap_rows = []
rng = np.random.default_rng(RANDOM_SEED)

for (threshold, outcome, model), group in paired.groupby(
    ["threshold", "outcome", "model"]
):

    if model == "BASELINE":
        continue

    fold_differences = {}

    for fold in sorted(group["outer_fold"].unique()):
        values = group.loc[
            group["outer_fold"] == fold,
            "mae_difference"
        ].to_numpy(dtype=float)

        if len(values) > 0:
            fold_differences[fold] = values

    if len(fold_differences) != N_OUTER_FOLDS:
        warnings.warn(
            f"Bootstrap fold coverage incomplete: "
            f"{threshold}, {outcome}, {model}"
        )

    bootstrap_estimates = np.empty(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        fold_means = []

        for fold, values in fold_differences.items():
            sampled = rng.choice(
                values,
                size=len(values),
                replace=True
            )
            fold_means.append(float(np.mean(sampled)))

        bootstrap_estimates[b] = np.mean(fold_means)

    point_estimate = np.mean([
        np.mean(values)
        for values in fold_differences.values()
    ])

    bootstrap_rows.append({
        "threshold": threshold,
        "outcome": outcome,
        "model": model,
        "receivers": group["nfl_id"].nunique(),
        "equal_fold_mae_difference": point_estimate,
        "ci_2_5": np.quantile(bootstrap_estimates, 0.025),
        "ci_97_5": np.quantile(bootstrap_estimates, 0.975),
        "probability_difference_below_zero": np.mean(
            bootstrap_estimates < 0
        ),
        "bootstrap_replicates": N_BOOTSTRAP,
    })

bootstrap_summary = pd.DataFrame(bootstrap_rows)

# ------------------------------------------------------------
# 11. COMPARE EACH THRESHOLD WITH THE 10-PLAY SPECIFICATION
# ------------------------------------------------------------

threshold_comparison = pooled_metrics.merge(
    pooled_metrics[
        pooled_metrics["threshold"] == 10
    ][
        ["outcome", "model", "mae", "rmse"]
    ].rename(columns={
        "mae": "ten_play_mae",
        "rmse": "ten_play_rmse",
    }),
    on=["outcome", "model"],
    how="left"
)

threshold_comparison["mae_change_vs_10"] = (
    threshold_comparison["mae"]
    - threshold_comparison["ten_play_mae"]
)

threshold_comparison["rmse_change_vs_10"] = (
    threshold_comparison["rmse"]
    - threshold_comparison["ten_play_rmse"]
)

# ------------------------------------------------------------
# 12. SAVE OUTPUTS
# ------------------------------------------------------------

outputs = {
    "sample_flow.csv": pd.DataFrame(sample_flow),
    "outer_fold_sample_audit.csv": pd.DataFrame(fold_audits),
    "eligibility_oof_predictions.csv": predictions,
    "eligibility_fold_metrics.csv": fold_metrics,
    "eligibility_pooled_metrics.csv": pooled_metrics,
    "eligibility_inner_tuning.csv": tuning_results,
    "eligibility_receiver_bootstrap.csv": bootstrap_summary,
    "eligibility_threshold_comparison.csv": threshold_comparison,
    "eligibility_receiver_level_errors.csv": receiver_errors,
}

for filename, table in outputs.items():
    table.to_csv(OUT / filename, index=False)

# ------------------------------------------------------------
# 13. REPRODUCIBILITY MANIFEST
# ------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = {
    "notebook": "10B_sample_eligibility_sensitivity",
    "primary_dataset": str(PRIMARY_FILE),
    "primary_dataset_sha256": sha256_file(PRIMARY_FILE),
    "frozen_core_dataset": str(CORE_FILE),
    "frozen_core_dataset_sha256": sha256_file(CORE_FILE),
    "thresholds": THRESHOLDS,
    "play_count_column": PLAY_COUNT_COL,
    "outcomes": OUTCOMES,
    "models": list(MODEL_FEATURES.keys()),
    "outer_folds": N_OUTER_FOLDS,
    "inner_folds_max": N_INNER_FOLDS,
    "alpha_grid": ALPHAS,
    "bootstrap_replicates": N_BOOTSTRAP,
    "random_seed": RANDOM_SEED,
    "sample_rule": (
        "Minimum receiver plays and complete cases across all "
        "nine predictors and six outcomes."
    ),
    "fold_rule": (
        "Preserve Notebook 07 assignments for existing core receivers; "
        "assign additional receivers deterministically to least-loaded folds."
    ),
    "preprocessing": (
        "StandardScaler fitted within each training fold in a Ridge Pipeline."
    ),
    "imputation": "None",
    "prediction_rows": len(predictions),
    "fold_metric_rows": len(fold_metrics),
    "outputs": list(outputs.keys()),
}

with open(
    OUT / "sample_eligibility_manifest.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=2)

# ------------------------------------------------------------
# 14. FINAL AUDITS
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("NOTEBOOK 10B — SAMPLE FLOW")
print("=" * 78)
print(pd.DataFrame(sample_flow).to_string(index=False))

print("\n" + "=" * 78)
print("POOLED MODEL PERFORMANCE")
print("=" * 78)
print(
    pooled_metrics[
        pooled_metrics["model"].isin(
            ["BASELINE", "TRADITIONAL", "TRACKING_FULL", "COMBINED_FULL"]
        )
    ].to_string(index=False)
)

print("\n" + "=" * 78)
print("RECEIVER-LEVEL BOOTSTRAP")
print("=" * 78)
print(
    bootstrap_summary[
        bootstrap_summary["model"].isin(
            ["TRADITIONAL", "TRACKING_FULL", "COMBINED_FULL"]
        )
    ].to_string(index=False)
)

# No receiver may appear in more than one fold per threshold.
leakage_check = (
    predictions.groupby(
        ["threshold", "nfl_id"]
    )["outer_fold"]
    .nunique()
)

assert (leakage_check == 1).all(), (
    "Receiver leakage detected across outer folds."
)

# Every model must have the same OOF key coverage within a threshold/outcome.
coverage = (
    predictions.groupby(
        ["threshold", "outcome", "model"]
    )
    .size()
)

coverage_counts = coverage.groupby(
    level=["threshold", "outcome"]
).nunique()

assert (coverage_counts == 1).all(), (
    "Model-specific OOF coverage differs within a threshold/outcome."
)

assert not predictions.duplicated(
    ["threshold", "outcome", "model", "draft_year", "nfl_id", "season"]
).any(), "Duplicate OOF prediction keys detected."

print("\n" + "=" * 78)
print("NOTEBOOK 10B COMPLETE")
print("=" * 78)
print("Output directory:", OUT.resolve())
print("OOF prediction rows:", len(predictions))
print("Fold metric rows:", len(fold_metrics))
print("Bootstrap comparisons:", len(bootstrap_summary))
print("Missing expected outputs:", [
    filename for filename in outputs
    if not (OUT / filename).exists()
])
print("Receiver leakage audit: PASS")
print("OOF coverage audit: PASS")
print("Duplicate prediction audit: PASS")
print("Manifest saved.")
