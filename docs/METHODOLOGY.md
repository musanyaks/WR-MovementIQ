# Methodology and measurement notes

## Research framing
The project examines observational associations between Combine receiver-drill movement and regular-season tracking characteristics. It does not identify causal effects.

## Coordinate geometry
For consecutive positions, step distance is `sqrt(dx² + dy²)`. Coordinate path length is the sum of step distances. Net displacement is the Euclidean distance between the first and last observations. Path efficiency is net displacement divided by path length when path length is positive.

The implemented heading convention is `degrees(atan2(dx, dy)) % 360`, clockwise from positive Y. This convention was empirically checked against recorded direction in the current project data and must be revalidated for changed source data.

## Turning
Heading differences are wrapped to [-180, 180). The implementation sums absolute changes for adjacent pairs where both recorded speeds meet the configured threshold. Report valid-pair coverage alongside turning summaries. Low-speed direction is unstable.

## Acceleration and braking
The source `a` field is not used as signed braking in this pipeline. A non-negative acceleration magnitude cannot distinguish acceleration from deceleration. A braking analysis requires a defensible signed longitudinal acceleration estimate, coordinate/time validation, smoothing decisions, and sensitivity analysis. Do not label acceleration magnitude or turning as braking.

## Game tracking window
The game feature function summarizes all available observations per player-play. It does not infer ball-snap or pass-forward event windows. If a pre-throw window is required, join the event markers and explicitly filter observations before feature calculation.

## Units
Features retain `_raw` where units have not been confirmed from the official data dictionary. Do not convert to physical units or state physical interpretations until units are verified.

## Predictive validation
Group all observations from the same `nfl_id` in one partition. Multiple player-seasons from the same athlete must not appear in both training and test partitions. Leave-one-draft-cohort-out evaluation should report held-out player counts and uncertainty.
