# Luna B — SEPNET reproduction and baseline contract

Status: research and interface contract only. No IRIS-SEP model was trained by
this workstream, and no IRIS locked-test identity, feature, outcome, prediction,
or threshold was opened. The primary agent owns the eventual locked evaluation.

This workstream records the public SEPNET/SEPNET-O contract and a predeclared
interface for the classical baselines. It is intentionally isolated under
`workstreams/luna_b/`; it does not modify the shared benchmark contract, source
tree, notebook, report, or data.

## Source pins

The pins below are the versions used for the contract review. The two GitHub
commit IDs are full 40-character SHA-1 values. The public source was inspected
through its release, commit, raw-file, or primary documentation URL. A local
clone was not required for this contract and a shell `git ls-remote` attempt
could not resolve GitHub DNS; therefore the receipt does not claim a local
checkout was reproduced.

| Source | Exact pin or version | Role | Primary URL |
|---|---|---|---|
| SEP-Prediction (v1) | `v1.0`, commit `f9cff73adfa41c4fbffc73a8693c529d39e80995` | Legacy SEPNet code, data construction, and classical-model notebook | [release](https://github.com/yuyian/SEP-Prediction/releases/tag/v1.0), [commit](https://github.com/yuyian/SEP-Prediction/commit/f9cff73adfa41c4fbffc73a8693c529d39e80995), [archive](https://zenodo.org/records/19007072) |
| SEP-Prediction-V2 | commit `e138dcd72c1952a00e11e1a0b025337f9e7c93fb` | Current public PRISM/operational runner reference | [commit](https://github.com/yuyian/SEP-Prediction-V2/commit/e138dcd72c1952a00e11e1a0b025337f9e7c93fb), [repository](https://github.com/yuyian/SEP-Prediction-V2) |
| SEPNet paper | arXiv `2512.12786v3` | Published data/model description and reported protocol | [HTML paper](https://arxiv.org/html/2512.12786) |
| CCMC SEPNET v1 | page state inspected 2026-09-04 | Live model input/output description | [NASA CCMC model page](https://ccmc.gsfc.nasa.gov/models/SEPNET~1/) |
| FetchSEP/CLEAR | `CLEAR_Benchmark_v1.0`, commit `9edc492567854d8fbf6ba0251be3659f48e88a20` | Public event-definition and CLEAR benchmark provenance | [release](https://github.com/ktindiana/fetchsep/releases/tag/CLEAR_Benchmark_v1.0), [commit](https://github.com/ktindiana/fetchsep/commit/9edc492567854d8fbf6ba0251be3659f48e88a20) |
| SEP-PRISM/CLEAR dataset | Zenodo record `21297635` | Future IRIS primary cohort source | [Zenodo record](https://zenodo.org/records/21297635) |
| SEPVAL rules/data landing page | public campaign data page | Frozen test rules and public benchmark context; not opened by Luna B | [CCMC SEP assessment](https://ccmc.gsfc.nasa.gov/assessment/topics/SEP/campaign2020/data_sets.php) |

The retrieval date for this receipt is 2026-09-04. Source URLs are retained in
`source_receipt.json` and the machine-readable baseline contract in
`baseline_contract.json`.

## What SEPNET actually specifies

There are two related but non-identical contracts. They must not be silently
blended.

### Public/live SEPNET v1 contract

The NASA CCMC description defines SEPNET v1 as a 24-hour Earth forecast of the
probability of a >10 MeV, >10 pfu SEP event. It uses the preceding 24 hours of
HMI SHARP near-real-time data and the SWPC flare list, with realistic data
availability/timeliness caveats. The operational output is a probability
summary, a symmetric uncertainty estimate from the preceding window, and an
all-clear flag. The live page does not promise a peak-flux or time-of-peak
forecast. The issue time and horizon are therefore part of the adapter
contract, not merely a plotting convention.

### Archived training/code contract (v1 paper and repository)

The v1 repository and paper construct fixed, non-overlapping 24-hour predictor
windows followed by an immediately subsequent 24-hour forecast window. Source
events are aggregated into summary statistics (minimum, maximum, and average)
and the next-window SEP label is derived from CLEAR/FetchSEP and operational
labels. The reported feature groups are:

* 24 SHARP magnetic/geometry parameters;
* flare duration, peak duration, and log peak strength;
* DONKI CME latitude, longitude, half-angle, and speed.

The repository's `Data-Construct.R` also uses KNN imputation (`k=10`) for
partially missing SHARP values and fits the min-max scaler on the training
partition. The historical paper describes a dense multitask model with shared
layers 256 → 128 → 64 → 16, separate regression/classification heads, and a
sequence variant with a unidirectional LSTM (hidden size 64), one four-head
Transformer encoder, and a 16-dimensional final representation. The loss is a
regression term plus BCE/focal classification terms; reported focal settings
are alpha 0.25, gamma 2, and classification weight 10.

For SEPNET-O, the published procedure trains using the general SEP target and
re-optimizes the decision threshold against operational-SEP labels in a
training/validation pool. It is not valid to copy a threshold from a paper
table or select it on the locked IRIS test labels.

The V1 notebook's classical settings are preserved in the machine-readable
receipt: elastic-net logistic regression (`solver="saga"`, `C=1`,
`l1_ratio=0.5`, `max_iter=10000`), SVM (`SVC(probability=True)`), random forest
(`n_estimators=100`), and XGBoost (`n_estimators=100`, `max_depth=6`,
`learning_rate=0.1`, binary-logistic objective). The historical notebook uses
a hard 0.5 cutoff and leaves several random states unset. Those are source
observations, not IRIS acceptance criteria.

## IRIS-SEP baseline execution contract

Every required baseline must receive the same parent-frozen SEP-PRISM/CLEAR
row IDs, issue timestamps, feature snapshot, partitions, and target definition.
The baseline code may not rebuild a more favorable cohort for itself.

The authoritative roles are the ones in
`../../config/benchmark_contract.json`:

* `train`: fit imputation, scaling, feature selection, and model parameters;
* `validation_monitor`: early stopping only;
* `validation_calibration`: probability calibration only;
* `validation_threshold`: operating-threshold selection only;
* `locked_test`: one final evaluation after every design choice is frozen.

Partitions must remain chronological, connected-region/episode disjoint, and
purged for the 24-hour forecast horizon. A complete SEP episode stays in one
partition. All input values must be available by the issue timestamp after the
declared source-publication latency. Any row already above the operational
threshold at issue time is excluded or separately marked; it is not silently
treated as a new crossing.

The primary label is the new >10 MeV, >=10 pfu crossing in the next 24 hours.
Public SEPNET/CLEAR descriptions sometimes write the flux boundary as
“>10 pfu” while the IRIS parent contract declares “>=10 pfu”; this notation
difference must be resolved in the authoritative label receipt before any
comparison run. The parent contract, not a copied paper sentence, controls the
IRIS label.
The >100 MeV, >=1 pfu event, peak-flux, and onset outputs are secondary targets.
`Future_*` values, post-issue GOES values, labels, event-list publication times,
or any AIA representation trained with future SEP labels are forbidden input
features.

The five predeclared seeds for stochastic baselines are `[7, 13, 26, 42, 73]`.
No seed may change the frozen row set. For deterministic baselines, repeating
the adapter across these seeds is an audit of determinism, not five independent
claims.

### Required baseline adapters

The baseline names below match the parent contract. A baseline adapter emits a
continuous probability for every row and keeps the row identity attached. It
does not emit only thresholded classes.

| Adapter | Predeclared definition | Locked-test rule |
|---|---|---|
| Climatology | Constant primary-label prevalence estimated on `train` only. Report the degenerate threshold curve explicitly. | No test prevalence or class count may be used. |
| Persistence | Last causal operational threshold state available at issue time. Because the target excludes rows already above threshold, an all-clear/degenerate result is an expected possibility, not a reason to change the cohort. | Never use the future state to make the “persistent” state. |
| Elastic-net logistic | Standardize/impute using train only; `LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=1, max_iter=10000, random_state=seed, class_weight=None)`. | Hyperparameters are fixed before test; threshold and any calibration use validation roles only. |
| XGBoost | `XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=seed)`, with train-only preprocessing. | No test `eval_set`, early stopping, feature selection, or hyperparameter search. |
| Published SEPNET | Adapter to the public preceding-24-hour HMI-SHARP/flare contract, preserving source latency and the model's probability/uncertainty/all-clear semantics. | If a reproducible executable endpoint or weights cannot be obtained, mark `NOT_RUN`; do not substitute a new model under this name. |
| SEPNET-O reproduction | Reproduce the archived general-SEP training target and apply the operational-label threshold selection on validation roles only. | The operational threshold is never selected from locked-test outcomes. |
| HMI flare expert only | Existing HMI image/flare expert with all non-HMI/SEP-PRISM modalities disabled, using the same causal row IDs and splits. | Must use the same calibration and threshold roles as the other IRIS variants. |

SVM and random forest are retained as optional historical-reference baselines
because they appear in the public V1 notebook and paper. They are not a reason
to enlarge the primary leaderboard if time or compute is constrained. Any
optional result still has to satisfy the same cohort and receipt contract.

For IRIS-SEP, the canonical input groups are magnetic SHARP/SMARP sequences,
eruption context (flare/XRS/CME), and pre-event proton context. AIA is disabled
for these classical baselines. An AIA expert can only enter a later comparison
after an exact, unambiguous AARP/HMI identity bridge; date or nearest-timestamp
joins are not admissible.

### Calibration, threshold, and metrics

Probability calibration, when enabled, is fit on the dedicated calibration
role. The operating threshold is then selected on the separate threshold role.
The locked evaluation consumes those already-frozen objects once. For every
model, retain the unthresholded probability and report AUROC, AUPRC, Brier/Brier
skill, reliability/calibration error, POD, FPR, FAR, TSS, HSS, warning lead time,
and missed-event severity. The gate compares IRIS-SEP with the recalibrated
SEPNET-O on the exact same frozen rows and uses a paired bootstrap over complete
SEP episodes or quiet blocks. No leaderboard number is present in this
workstream because no locked evaluation has run.

## Reproducibility discrepancies and red flags

These are deliberate review findings; they are not silently “fixed” in the
public repository.

1. The V1 notebook uses a 0.5 class cutoff for classical models. IRIS must use
   a predeclared validation-only policy and retain continuous probabilities.
2. The V1 classical evaluation passes the regression target as both predicted
   and true values for classification-only models. That makes regression
   diagnostics look perfect and is a metric-contract defect. IRIS adapters must
   use missing regression outputs and omit regression metrics for a classifier.
3. V1 leaves several random states as `None`. IRIS uses explicit seeds and
   records the environment/configuration hashes.
4. V1's SEPVAL construction identifies historical windows by time but does not
   provide the parent contract's connected-region/episode disjointness and
   purge safeguards. It is a legacy reference, not an accepted IRIS split.
5. V1 aggregates the 24-hour history into summary statistics. This does not
   reproduce the causal temporal sequence proposed for IRIS, so it cannot be
   described as a full IRIS architecture result.
6. The paper reports five repeats for one split description and aggregates some
   metrics over 50 seeds elsewhere. This ambiguity is retained as a discrepancy;
   the IRIS handoff's five-seed rule controls this project.
7. V2 adds proton-flux/XRS inputs and an operational runner, but its public
   configuration includes a `random_iid` split. That is not the frozen IRIS
   chronological episode-disjoint split and must not override it.
8. The live CCMC page documents no peak-flux or time-of-peak output. Those IRIS
   secondary heads must not be attributed to published SEPNET.
9. Public source prose alternates between `>10 pfu` and the IRIS contract's
   `>=10 pfu` boundary. Do not infer that the two are equivalent without an
   explicit CLEAR/SEPVAL label-definition receipt.

## Run/receipt status

`baseline_contract.json`, `environment_manifest.json`, and
`source_receipt.json` are templates/receipts for the parent agent. The
`baseline_interface.py` and its test file only validate tuning-time partition
and prediction-frame semantics on in-memory toy records. They do not load
SEP-PRISM, SEPVAL, AARP, FITS, or any locked artifact.

Before any run, the parent agent must fill the cohort/partition/feature hashes
from the authoritative receipts and capture the actual Colab environment. A
future baseline run is valid only if its prediction file and configuration
receipts reference those exact hashes.
