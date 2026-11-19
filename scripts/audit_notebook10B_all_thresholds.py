from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.cwd()
OUT = ROOT / "reports" / "robustness_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = ROOT / "data/features/wr_analysis_dataset_primary.parquet"
FOLD_FILE = ROOT / "reports/modeling_design/outer_grouped_fold_assignments.csv"
SAVED = ROOT / "reports/sample_eligibility_sensitivity"

THRESHOLDS = [5, 8, 10, 12, 15]
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
OUTER_FOLDS = 5
INNER_FOLDS = 4
TOL = 1e-10

KEYS = ["draft_year", "nfl_id", "season"]

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

COMBINED = TRADITIONAL + TRACKING

SPECS = {
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
    "COMBINED_PEAK_SPEED_ONLY": TRADITIONAL + [
        "combine_peak_speed_median_across_drills"
    ],
    "COMBINED_P95_SPEED_ONLY": TRADITIONAL + [
        "combine_p95_speed_median_across_drills"
    ],
    "COMBINED_NO_ACCELERATION": [
        x for x in COMBINED if "acceleration" not in x
    ],
    "TRACKING_SPEED_ONLY": [
        "combine_mean_speed_median_across_drills",
        "combine_peak_speed_median_across_drills",
        "combine_p95_speed_median_across_drills",
    ],
}

def make_model(alpha):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])

def tune_alpha(X, y, groups):
    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        raise ValueError("Insufficient receiver groups.")

    cv = GroupKFold(n_splits=min(INNER_FOLDS, n_groups))
    records = []

    for alpha in ALPHAS:
        scores = []

        for train_idx, valid_idx in cv.split(X, y, groups=groups):
            model = make_model(alpha)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            scores.append(
                mean_absolute_error(y.iloc[valid_idx], pred)
            )

        records.append({
            "candidate_alpha": float(alpha),
            "inner_mae": float(np.mean(scores)),
        })

    tuning = pd.DataFrame(records).sort_values(
        ["inner_mae", "candidate_alpha"]
    ).reset_index(drop=True)

    return float(tuning.iloc[0]["candidate_alpha"]), tuning

def assign_folds(sample, original_fold_map):
    counts = sample.groupby("nfl_id").size().sort_index()

    assignment = {}
    loads = {f: 0 for f in range(1, OUTER_FOLDS + 1)}

    for nfl_id, count in counts.items():
        if nfl_id in original_fold_map:
            fold = int(original_fold_map[nfl_id])
            assignment[nfl_id] = fold
            loads[fold] += int(count)

    new_receivers = [
        nfl_id for nfl_id in counts.index
        if nfl_id not in assignment
    ]

    for nfl_id in new_receivers:
        fold = min(loads, key=lambda f: (loads[f], f))
        assignment[nfl_id] = fold
        loads[fold] += int(counts[nfl_id])

    result = sample["nfl_id"].map(assignment)

    if result.isna().any():
        raise AssertionError("Unassigned receiver.")

    if sample.assign(_fold=result).groupby("nfl_id")["_fold"].nunique().max() != 1:
        raise AssertionError("Receiver leakage.")

    return result.to_numpy(), loads

def compare_tables(saved, rebuilt, keys, numeric_columns, label):
    saved = saved.copy()
    rebuilt = rebuilt.copy()

    for key in keys:
        if key == "nfl_id":
            saved[key] = saved[key].astype(str)
            rebuilt[key] = rebuilt[key].astype(str)

    if saved.duplicated(keys).any():
        raise AssertionError(f"{label}: duplicate keys in saved table.")
    if rebuilt.duplicated(keys).any():
        raise AssertionError(f"{label}: duplicate keys in rebuilt table.")

    merged = saved.merge(
        rebuilt,
        on=keys,
        how="outer",
        indicator=True,
        suffixes=("_saved", "_rebuilt"),
        validate="one_to_one"
    )

    unmatched = int((merged["_merge"] != "both").sum())
    rows = []

    for col in numeric_columns:
        a = pd.to_numeric(merged[f"{col}_saved"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_rebuilt"], errors="coerce")

        valid = merged["_merge"].eq("both")
        both_nan = a.isna() & b.isna()
        one_nan = a.isna() ^ b.isna()
        diff = (a - b).abs()

        matched = valid & (both_nan | ((~one_nan) & (diff <= TOL)))

        rows.append({
            "table": label,
            "field": col,
            "saved_rows": len(saved),
            "rebuilt_rows": len(rebuilt),
            "unmatched_keys": unmatched,
            "matching_values": int(matched.sum()),
            "compared_keys": int(valid.sum()),
            "max_absolute_difference": (
                float(diff[valid & diff.notna()].max())
                if (valid & diff.notna()).any() else 0.0
            ),
            "pass": bool(unmatched == 0 and matched.sum() == valid.sum()),
        })

    return pd.DataFrame(rows)

print("=" * 90)
print("INDEPENDENT NOTEBOOK 10B ALL-THRESHOLD RECONSTRUCTION")
print("=" * 90)

df = pd.read_parquet(PRIMARY)
fold_source = pd.read_csv(FOLD_FILE)

play_col = "receiver_plays"
required = KEYS + [play_col] + OUTCOMES + COMBINED
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

for data in [df]:
    data["nfl_id"] = data["nfl_id"].astype(str)
    data["draft_year"] = pd.to_numeric(data["draft_year"], errors="coerce")
    data["season"] = pd.to_numeric(data["season"], errors="coerce")

df[play_col] = pd.to_numeric(df[play_col], errors="coerce")

for col in OUTCOMES + COMBINED:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df.duplicated(KEYS).any():
    raise AssertionError("Duplicate primary dataset keys.")

df["complete_case"] = df[OUTCOMES + COMBINED].notna().all(axis=1)

fold_col = next(
    (c for c in ["outer_fold", "fold", "fold_id"] if c in fold_source.columns),
    None
)
if fold_col is None:
    raise ValueError("No fold column found.")

fold_source["nfl_id"] = fold_source["nfl_id"].astype(str)
fold_source[fold_col] = pd.to_numeric(
    fold_source[fold_col], errors="coerce"
)
fm = fold_source[["nfl_id", fold_col]].dropna().copy()
fm[fold_col] = fm[fold_col].astype(int)

labels = sorted(fm[fold_col].unique())
if len(labels) != OUTER_FOLDS:
    raise AssertionError(f"Unexpected fold labels: {labels}")

label_map = {old: new for new, old in enumerate(labels, start=1)}
fm[fold_col] = fm[fold_col].map(label_map)

if fm.groupby("nfl_id")[fold_col].nunique().max() != 1:
    raise AssertionError("Original fold map splits receivers.")

original_fold_map = (
    fm.drop_duplicates("nfl_id")
    .set_index("nfl_id")[fold_col].to_dict()
)

print("Primary dataset:", df.shape)
print("Complete cases:", int(df["complete_case"].sum()))
print("Original receiver fold assignments:", len(original_fold_map))

rebuilt_predictions = []
rebuilt_tuning = []
rebuilt_metrics = []
rebuilt_samples = []
rebuilt_fold_counts = []

for threshold in THRESHOLDS:
    print(f"\nReconstructing threshold {threshold}...")

    sample = df[
        (df[play_col] >= threshold) & df["complete_case"]
    ].copy()

    sample = sample.sort_values(KEYS).reset_index(drop=True)

    if sample["nfl_id"].nunique() < OUTER_FOLDS + 1:
        raise AssertionError(f"Too few receivers at threshold {threshold}.")

    folds, loads = assign_folds(sample, original_fold_map)
    sample["outer_fold"] = folds

    rebuilt_samples.append({
        "threshold": threshold,
        "player_seasons": len(sample),
        "receivers": sample["nfl_id"].nunique(),
        "folds": sample["outer_fold"].nunique(),
    })

    rebuilt_fold_counts.append({
        "threshold": threshold,
        **{
            f"fold_{f}_rows": int((sample["outer_fold"] == f).sum())
            for f in range(1, OUTER_FOLDS + 1)
        }
    })

    for outcome in OUTCOMES:
        for model_name, features in SPECS.items():
            for outer_fold in range(1, OUTER_FOLDS + 1):
                train = sample[sample["outer_fold"] != outer_fold].copy()
                test = sample[sample["outer_fold"] == outer_fold].copy()

                y_train = train[outcome].astype(float)
                y_test = test[outcome].astype(float)

                if model_name == "BASELINE":
                    best_alpha = np.nan
                    inner_best_mae = np.nan
                    pred = np.repeat(float(y_train.median()), len(test))
                else:
                    X_train = train[features].astype(float).reset_index(drop=True)
                    X_test = test[features].astype(float)
                    groups = train["nfl_id"].to_numpy()

                    best_alpha, tuning = tune_alpha(
                        X_train, y_train.reset_index(drop=True), groups
                    )

                    inner_best_mae = float(
                        tuning.loc[
                            tuning["candidate_alpha"] == best_alpha,
                            "inner_mae"
                        ].iloc[0]
                    )

                    for _, row in tuning.iterrows():
                        rebuilt_tuning.append({
                            "threshold": threshold,
                            "outcome": outcome,
                            "model": model_name,
                            "outer_fold": outer_fold,
                            "candidate_alpha": row["candidate_alpha"],
                            "inner_mae": row["inner_mae"],
                        })

                    fitted = make_model(best_alpha)
                    fitted.fit(X_train, y_train)
                    pred = fitted.predict(X_test)

                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred) if len(test) >= 2 else np.nan

                rebuilt_metrics.append({
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
                    rebuilt_predictions.append({
                        "threshold": threshold,
                        "outcome": outcome,
                        "model": model_name,
                        "nfl_id": str(row["nfl_id"]),
                        "draft_year": row["draft_year"],
                        "season": row["season"],
                        "outer_fold": outer_fold,
                        "actual": float(y_test.iloc[i]),
                        "predicted": float(pred[i]),
                        "absolute_error": abs(
                            float(y_test.iloc[i]) - float(pred[i])
                        ),
                        "best_alpha": best_alpha,
                    })

pred = pd.DataFrame(rebuilt_predictions)
tuning = pd.DataFrame(rebuilt_tuning)
metrics = pd.DataFrame(rebuilt_metrics)

print("\nRebuilt predictions:", len(pred))
print("Rebuilt tuning scores:", len(tuning))
print("Rebuilt fold metrics:", len(metrics))

saved_pred = pd.read_csv(SAVED / "eligibility_oof_predictions.csv")
saved_tuning = pd.read_csv(SAVED / "eligibility_inner_tuning.csv")
saved_metrics = pd.read_csv(SAVED / "eligibility_fold_metrics.csv")

pred_keys = [
    "threshold", "outcome", "model",
    "nfl_id", "draft_year", "season"
]
tuning_keys = [
    "threshold", "outcome", "model",
    "outer_fold", "candidate_alpha"
]
metric_keys = ["threshold", "outcome", "model", "outer_fold"]

prediction_audit = compare_tables(
    saved_pred, pred, pred_keys,
    ["outer_fold", "actual", "predicted", "absolute_error", "best_alpha"],
    "OOF_PREDICTIONS"
)

tuning_audit = compare_tables(
    saved_tuning, tuning, tuning_keys,
    ["inner_mae"],
    "INNER_TUNING"
)

metric_audit = compare_tables(
    saved_metrics, metrics, metric_keys,
    [
        "train_player_seasons", "test_player_seasons",
        "train_receivers", "test_receivers",
        "mae", "rmse", "r2", "best_alpha", "inner_best_mae"
    ],
    "FOLD_METRICS"
)

all_audits = pd.concat(
    [prediction_audit, tuning_audit, metric_audit],
    ignore_index=True
)

all_audits.to_csv(
    OUT / "notebook10B_all_threshold_reconstruction_audit.csv",
    index=False
)

pd.DataFrame(rebuilt_samples).to_csv(
    OUT / "notebook10B_rebuilt_sample_sizes.csv", index=False
)

pd.DataFrame(rebuilt_fold_counts).to_csv(
    OUT / "notebook10B_rebuilt_fold_counts.csv", index=False
)

print("\n" + "=" * 90)
print("AUDIT SUMMARY")
print("=" * 90)
print(all_audits.to_string(index=False))

print("\nPASS:", int(all_audits["pass"].sum()), "/", len(all_audits))
print("FAILED:", int((~all_audits["pass"]).sum()))

if all_audits["pass"].all():
    print("\nALL-THRESHOLD INDEPENDENT RECONSTRUCTION: PASS")
else:
    print("\nALL-THRESHOLD INDEPENDENT RECONSTRUCTION: FAIL")
    print("Inspect the saved audit CSV before accepting the outputs.")

print("\nAudit file:")
print(OUT / "notebook10B_all_threshold_reconstruction_audit.csv")
