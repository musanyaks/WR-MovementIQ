
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "reports/nested_ridge_modeling/oof_predictions.csv"
OUTPUT = ROOT / "reports/model_stability/oof_fold_calibration_audit.csv"

df = pd.read_csv(INPUT)

required = {
    "outcome", "model", "nfl_id", "season",
    "draft_year", "outer_fold", "actual", "predicted"
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
df["outer_fold"] = pd.to_numeric(df["outer_fold"], errors="raise")

df = df.dropna(subset=["actual", "predicted"])

def safe_corr(x, y, method="pearson"):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    if method == "pearson":
        return pearsonr(x, y).statistic

    return spearmanr(x, y).statistic


records = []

for (outcome, model), group in df.groupby(["outcome", "model"]):

    # Pooled diagnostics
    pooled_r = safe_corr(
        group["actual"], group["predicted"], "pearson"
    )

    pooled_rho = safe_corr(
        group["actual"], group["predicted"], "spearman"
    )

    # Centre actual and predicted values separately within each fold.
    centred = group.copy()

    centred["actual_centered"] = (
        centred["actual"]
        - centred.groupby("outer_fold")["actual"].transform("mean")
    )

    centred["predicted_centered"] = (
        centred["predicted"]
        - centred.groupby("outer_fold")["predicted"].transform("mean")
    )

    centred_r = safe_corr(
        centred["actual_centered"],
        centred["predicted_centered"],
        "pearson"
    )

    centred_rho = safe_corr(
        centred["actual_centered"],
        centred["predicted_centered"],
        "spearman"
    )

    # Fold-level metrics
    for fold, part in group.groupby("outer_fold"):

        actual = part["actual"].to_numpy()
        predicted = part["predicted"].to_numpy()

        actual_sd = np.std(actual, ddof=1)
        predicted_sd = np.std(predicted, ddof=1)

        calibration_slope = np.nan

        if actual_sd > 0:
            calibration_slope = (
                np.cov(actual, predicted, ddof=1)[0, 1]
                / np.var(actual, ddof=1)
            )

        records.append({
            "outcome": outcome,
            "model": model,
            "outer_fold": int(fold),
            "observations": len(part),
            "receivers": part["nfl_id"].nunique(),
            "mae": mean_absolute_error(actual, predicted),
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "r2": (
                r2_score(actual, predicted)
                if len(actual) >= 2 else np.nan
            ),
            "pearson_r": safe_corr(actual, predicted, "pearson"),
            "spearman_rho": safe_corr(actual, predicted, "spearman"),
            "prediction_sd_ratio": (
                predicted_sd / actual_sd
                if actual_sd > 0 else np.nan
            ),
            "calibration_slope": calibration_slope,
            "pooled_pearson_r": pooled_r,
            "pooled_spearman_rho": pooled_rho,
            "fold_centred_pearson_r": centred_r,
            "fold_centred_spearman_rho": centred_rho
        })


result = pd.DataFrame(records)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("=" * 85)
print("NOTEBOOK 08B PRELIMINARY — OOF FOLD CALIBRATION AUDIT")
print("=" * 85)

print(f"Input observations: {len(df):,}")
print(f"Unique receivers: {df['nfl_id'].nunique():,}")
print(f"Outcomes: {df['outcome'].nunique()}")
print(f"Models: {df['model'].nunique()}")
print(f"Output: {OUTPUT}")

print("\nFOLD-LEVEL RESULTS")
print(
    result[
        [
            "outcome", "model", "outer_fold",
            "mae", "r2", "pearson_r",
            "spearman_rho", "prediction_sd_ratio"
        ]
    ].to_string(index=False)
)

print("\nPOOLED VS FOLD-CENTRED CORRELATIONS")

summary = (
    result.groupby(["outcome", "model"])
    .agg(
        pooled_pearson=("pooled_pearson_r", "first"),
        fold_centred_pearson=("fold_centred_pearson_r", "first"),
        pooled_spearman=("pooled_spearman_rho", "first"),
        fold_centred_spearman=("fold_centred_spearman_rho", "first"),
        mean_fold_mae=("mae", "mean"),
        mean_fold_r2=("r2", "mean"),
        mean_fold_sd_ratio=("prediction_sd_ratio", "mean")
    )
    .reset_index()
)

print(summary.to_string(index=False))

print("\nAUDIT COMPLETE")
