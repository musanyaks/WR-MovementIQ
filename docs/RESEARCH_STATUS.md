# Research status and next work

This repository packages the initial reproducible data-ingestion and feature-engineering layer. The following project analyses have been performed in the user's Colab workflow and must be imported/reconciled as source notebooks or scripts before this repository can claim full end-to-end reproduction:

- Combine tracking structural and heading validation.
- Attempt-level turning reconstruction agreement checks.
- Receiver game tracking feature construction and event-window audits.
- Player linkage and integrated player-season profiles.
- Nested player-grouped Ridge validation.
- Leave-one-draft-cohort-out validation.
- Bootstrap uncertainty and paired model comparisons.
- Traditional feature ablation and missingness/collapse diagnostics.

## Key research cautions
- The Combine tracking `a` values observed in the current analysis were non-negative. They do not directly establish signed braking.
- The game features previously used in the analysis were explicitly restricted to the ball-snap-to-pass-forward window; the general-purpose feature builder in this repository currently computes whole player-play summaries. Do not compare the two without implementing and validating the same event window.
- Tracking-derived turning measurements showed cohort-dependent measurement differences. Treat cross-cohort comparisons cautiously.
- Only 55 players had complete cases for the traditional three-feature ablation in the described analysis; the broader modelling sample contained 66 players under the stated exposure threshold.
- A fair final comparison of traditional, tracking-only and combined models should use the same players, cohort splits, outcome aggregation and baseline.

## Next milestones
1. Import the exact executed Colab notebooks and saved result artifacts.
2. Implement the ball-snap-to-pass-forward game window in a separate, tested function.
3. Implement signed longitudinal acceleration/deceleration validation and sensitivity analysis before making braking claims.
4. Reproduce the nested grouped and cohort-holdout model experiments from checked-in code.
5. Run the common-sample model comparison and report prediction collapse diagnostics.
6. Generate the final competition report only after all results reconcile.
