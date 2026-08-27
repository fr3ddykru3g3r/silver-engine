# IRIS v2 prospective blind validation protocol

Status: infrastructure frozen before the final v2 model exists. This protocol is supplementary to the four rolling-origin historical outer folds.

## Purpose
Create a small genuinely prospective audit in which model predictions are cryptographically frozen before the corresponding 24-hour GOES outcomes are collected/scored.

## Forecast unit
A forecast row is one connected active-region group at one issuance time. The model may use only definitive/available-at-issuance LOS HMI magnetogram information whose observation time is not later than the issuance timestamp. The selected HMI observation should be the latest available observation; record its T_REC and the data latency explicitly.

## Outcome clock
For prospective scoring the forecast window is `(issued_at_utc, issued_at_utc + 24 h]`. This is deliberately defined by issuance time rather than retrospectively shifting the window to a more convenient magnetogram timestamp.

## Freeze requirements
Before any future GOES outcomes for the window are queried, store:
- forecast issuance UTC timestamp;
- model/checkpoint SHA-256;
- code commit SHA;
- frozen protocol SHA-256;
- exact prediction CSV SHA-256;
- active-region identifiers and connected-region mapping;
- input HMI T_REC and data latency;
- predicted probability and frozen decision threshold;
- 24-hour horizon end.

The prediction file must contain no future flare/outcome column. The freeze script rejects outcome-like columns and rejects HMI observations later than issuance.

## Public timestamp
Upload the frozen ledger artifact to GitHub Actions (or commit only its hashes/metadata) immediately after issuance. The externally recorded GitHub timestamp is supplementary evidence that the prediction existed before the outcome window ended.

## Scoring
Only after every row's `horizon_end_utc` has passed, retrieve the authoritative GOES flare catalogue and apply the same M1+ event definition/NOAA-region attribution rules used by the historical project. Do not alter probabilities, thresholds, region mappings, or inclusion rules after outcomes are visible.

## Report
Prospective results are reported separately because the sample will be smaller than the historical backtest. Report TSS only if both classes are adequately represented; always report raw TP/TN/FP/FN, probabilities, Brier score, and every issued forecast. No selective removal of failed forecasts is permitted.
