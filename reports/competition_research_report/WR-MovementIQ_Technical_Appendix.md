# Appendix: Data Pipeline and Measurement Validation



**WR-MovementIQ | NFL Big Data Bowl 2027**



This appendix documents the data sources, feature-engineering procedures, measurement validation, modelling population, and reproducibility checks used to construct the WR-MovementIQ research dataset.



## A1. Raw data used



| File | Purpose |

|---|---|

| `combine_tracking.csv` | NFL Combine drill tracking observations; PLAYER entities used |

| `combine_results.csv` | Combine position filtering and traditional athletic measurements |

| `players.csv` | Player identifiers, positions, and draft cohorts |

| `player_play.csv` | Receiver-play metadata and play-level information |

| `game_tracking_2023.csv`–`game_tracking_2025.csv` | Regular-season player tracking data |

| `games.csv` | Game identifiers and season information |



`player_career_successes.csv` was not used in the final model specifications.



The analysis covers the 2023, 2024, and 2025 Combine cohorts and corresponding regular-season tracking seasons. Competition data are treated as research inputs and are not redistributed through this project.



## A2. Combine feature engineering



**Population definition.** Wide receivers were identified using `combine_position` in `combine_results.csv`. Only observations with `entity_type = PLAYER` were retained from Combine tracking. Ball-tracking observations were excluded.



The resulting Combine population contained **107 receivers and 1,771 drill attempts** across the 2023, 2024, and 2025 Combine cohorts.



**Primary receiver drills.** Eleven receiver-oriented drills were selected:



1\. Gauntlet

2\. Dagger Route (Left)

3\. Slant Route (Left)

4\. Curl Route (Right)

5\. Slot Strike Route (Left)

6\. Over-the-Shoulder Adjust

7\. Red Zone Fade (Right)

8\. Comeback Route (Right)

9\. Slot Sail Route (Right)

10\. Go Route (Right)

11\. Post-Corner Route (Right)



The 40-yard dash, vertical jump, and broad jump were retained as traditional Combine measurements rather than treated as receiver-drill tracking features.



**Attempt-level feature engineering.** Tracking observations were ordered chronologically within each attempt. Attempt-level measurements included duration, mean speed, peak speed, 95th-percentile speed, mean acceleration, peak acceleration, coordinate-derived path length, net displacement, and path efficiency.



Path efficiency was calculated as net displacement divided by path length. Valid values were constrained to the interval [0, 1].



**Player-level aggregation.** Attempt-level measurements were first summarized within drills. Drill-level summaries were subsequently aggregated across the selected receiver drills using medians.



This process produced six tracking-derived player-level predictors:



1\. Mean speed

2\. Peak speed

3\. 95th-percentile speed

4\. Mean acceleration

5\. Peak acceleration

6\. Path efficiency



The median across drills was used to construct the receiver-level Combine tracking predictors. This aggregation reduces the influence of unusually high or low measurements from individual attempts.



## A3. Measurement validation



Before using the tracking measurements, recorded speed and direction were examined against the underlying spatial coordinates.



**Sampling intervals.** Combine tracking observations were assessed for temporal spacing. The median sampling interval was 0.1 seconds. Gaps exceeding 0.15 seconds were evaluated at the attempt level.



**Heading convention.** Recorded direction was compared with coordinate-derived direction under four candidate angular conventions. The selected convention was:



```python

heading = degrees(atan2(dx, dy)) % 360

```



This represents clockwise angular measurement from the positive y-axis.



Across the validated comparisons, median heading error was approximately 1.02 degrees, mean error 2.77 degrees, and 95th-percentile error 7.57 degrees. Error was also examined across speed bands and draft cohorts because directional estimates can become less reliable at low speeds.



**Timestamp alignment.** Directional alignment was investigated using temporal lags from -2 to +2 frames. The lag investigation is documented as a validation exercise; no specific optimal lag is asserted here.



**Turning features.** Turning measurements included total direction change, mean direction change, 95th-percentile direction change, and turning rate. These were reconstructed from recorded and coordinate-derived headings using consecutive observations with intervals between 0.05 and 0.15 seconds.



A minimum speed threshold of 1.0 yd/s was used for the primary turning reconstruction, with alternative thresholds considered in sensitivity checks.



Turning reconstruction showed strong agreement for the evaluated threshold comparison: Pearson correlation was approximately 0.9983 and Spearman correlation approximately 0.9987. The median absolute error was approximately 1.43 degrees. Agreement varied across draft cohorts, so these measurements were not assumed to represent directly comparable underlying agility differences between cohorts.



**Final modelling decision.** Turning measurements were retained as measurement-validation outputs but excluded from the final model specifications. The six speed, acceleration, and path-efficiency predictors were used instead.



Importantly, acceleration magnitude was not interpreted as signed braking or deceleration.



## A4. NFL game feature engineering



Regular-season tracking was processed to construct receiver-play movement measurements over the interval between the snap and the pass throw.



The resulting game movement feature dataset contained **6,801 receiver-play records** across the 2023–2025 seasons.



Relevant observations were restricted to WR and SLOT_WR positions and to seasons on or after the player's draft year.



Play-level measurements included speed, acceleration, path length, path efficiency, and direction-change characteristics.



Play-level observations were subsequently aggregated into receiver-season profiles. These profiles contained median and mean movement summaries and were linked to Combine receiver profiles using player identifier and draft year.



The merged game player-season profile contained **146 receiver-seasons** before the final eligibility and complete-case restrictions. This is a different unit of observation from the 6,801 receiver-play records.



Receiver exposure was measured using `receiver_plays`. The primary analysis required at least 10 receiver plays, with alternative exposure thresholds evaluated separately.



## A5. Modelling population and validation



The frozen primary modelling dataset contained **93 player-seasons from 54 unique receivers**.



The draft-cohort distribution was:



| Draft cohort | Player-seasons |

|---|---:|

| 2023 | 45 |

| 2024 | 34 |

| 2025 | 14 |

| **Total** | **93** |



The primary analysis used six in-game movement outcomes:



\- Median game mean speed

\- Median game peak speed

\- Median game 95th-percentile speed

\- Median game mean acceleration

\- Median game peak acceleration

\- Median game path efficiency



The traditional Combine predictor specification used the 40-yard dash, vertical jump, and broad jump. The tracking-derived specification used the six Combine movement predictors described in Section A2. A combined specification included both predictor groups.



Complete-case eligibility was applied across the predictors and outcomes required by the final modelling specifications. Although the modelling pipeline includes median imputation functionality, no missing-value imputation was needed for the final complete-case observations.



**Primary validation.** Receivers were kept together when assigning outer validation folds. The primary evaluation used five-fold receiver-grouped cross-validation, with inner grouped validation for Ridge hyperparameter selection.



Feature standardization was fitted within training folds to prevent information leakage. Ridge regularization strength was selected using inner validation, with mean validation MAE as the selection criterion.



Leave-one-draft-cohort-out (LOCO) validation was also conducted to examine generalization to held-out draft classes.



The 93 observations represent player-seasons, not 93 independent receivers. The receiver-grouped validation design prevents observations belonging to the same receiver from being split between training and test folds in the primary cross-validation analysis.



The 2025 LOCO test cohort contained 14 receivers. Results for that cohort should therefore be interpreted with particular caution.



**Interpretation.** The research evaluates predictive associations between Combine movement characteristics and subsequent in-game movement measurements. It is observational and does not establish that Combine performance causes regular-season performance.



## A6. Reproducibility and audit trail



The project maintains separate raw-data, processed-data, feature, notebook, and reporting directories.



The analysis workflow includes:



\- Raw-data validation and player-key checks.

\- Combine attempt and player-profile reconstruction.

\- NFL receiver-play and player-season feature construction.

\- Frozen analysis datasets and validation assignments.

\- Nested Ridge predictions and hyperparameter-selection records.

\- Receiver-level bootstrap comparisons.

\- Draft-cohort generalization and eligibility sensitivity analyses.

\- Publication figures, research tables, and provenance manifests.



**Independent sensitivity reconstruction.** The independently reconstructed Notebook 10B sensitivity analysis evaluated receiver-play exposure thresholds of 5, 8, 10, 12, and 15 plays.



The corresponding complete-case sensitivity samples were:



| Minimum receiver plays | Player-seasons | Receivers |

|---:|---:|---:|

| 5 | 93 | 54 |

| 8 | 93 | 54 |

| 10 | 93 | 54 |

| 12 | 92 | 53 |

| 15 | 83 | 48 |



These sensitivity samples applied complete-case requirements across the nine predictors and six outcomes used in that analysis. They should not be confused with exposure-only eligibility counts before complete-case filtering.



Across the five sensitivity thresholds, the independent reconstruction audit checked:



\- **27,240** out-of-fold prediction records.

\- **12,150** inner-tuning records.

\- **1,500** fold-metric records.



Predictions, selected regularization parameters, tuning results, and fold metrics passed the independent reconstruction checks within numerical tolerance.



For the expanded sensitivity samples, existing core receiver fold assignments were preserved. Additional receivers were assigned deterministically to the least-loaded folds. This procedure is not described as a fresh GroupKFold assignment for every expanded sample.



**Audit limitations.** The original Notebook 10 sensitivity outputs were not accepted as authoritative because their saved tuning results failed independent reconstruction checks. The independently reconstructed Notebook 10B outputs are the authoritative sensitivity results.



The bootstrap analysis used a fixed seed of 2027. Other workflow seeds are recorded in the corresponding project configuration and manifests.



The project distinguishes between independently validated outputs and earlier analysis artifacts that were not accepted as authoritative. Reproducibility claims are limited to the specific datasets, outputs, and checks documented in the project audit trail.



**Public execution scope.** A public execution environment must have authorized access to the competition data and the required project dependencies. Local processed files and Windows-specific paths are not assumed to exist in that environment. The public notebook must identify its input-data requirements and must not claim full end-to-end reproducibility unless it actually reconstructs the required features and analyses from the available competition inputs.

