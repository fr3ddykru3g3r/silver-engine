# Threshold policy reconciliation — 2026-09-06

## Decision

The frozen benchmark primary operating threshold remains the rule in `config/evaluation_policy_v1.json`:

> choose the threshold on the dedicated threshold role that **maximizes TSS**, breaking ties by the **lowest threshold**; do not refit on the test set.

`POD80_MIN_FAR` remains a separately reported reliability / matched-detection policy:

> among threshold-role cutoffs achieving POD >= 0.80, choose the one with minimum FAR.

It is useful for asking “how many false alerts do we pay for high detection?” but it is **not** allowed to silently replace the frozen MAX_TSS benchmark primary after development results have been inspected.

## Why this reconciliation is necessary

Several later architecture/missingness reports emphasized `POD80_MIN_FAR` because false alarms are important for the intended analyst workflow. That is scientifically useful, but the original frozen evaluation contract predates those development results and declares MAX_TSS as the operating-threshold objective.

Calling POD80 the new primary after inspecting development performance would create a post-result metric/policy shift.

## Reporting rule from this point forward

Every new table that includes thresholded performance should clearly separate:

1. **Benchmark primary — MAX_TSS**
   - threshold selected only on the threshold role;
   - TSS is the primary metric;
   - HSS/POD/FPR/FAR accompany it.

2. **Reliability diagnostic — POD80_MIN_FAR**
   - threshold selected only on the threshold role;
   - intended to expose the false-alert cost of high detection;
   - never described as guaranteed 80% POD outside the threshold-selection block.

3. **Matched-detection curve diagnostics**
   - 0.6/0.7/0.8/0.9 target POD grid;
   - used for comparison, not as a hidden deployed-threshold refit.

Probability metrics (Brier, Brier skill, ECE, AUPRC, AUROC) remain threshold-independent and should be reported for the same cohort.

## Historical results

No historical artifact is altered. If a run was reported under `POD80_MIN_FAR`, its result remains exactly that. It can be cited as development evidence under the POD80 policy, but it does not redefine the frozen benchmark contract.

In particular, the promoted cross-fitted stack’s approximately `0.5120` older-score TSS and `0.2359` later-monitor TSS currently highlighted in project summaries are POD80/minimum-FAR development-policy results. The corresponding MAX_TSS results must be reported separately whenever a final headline table is generated.

## Claim boundary

A model may look better under one operating policy and worse under another. Both declared policies must be shown. No superiority claim may be based on selecting whichever policy is favorable after viewing the comparison.