# IRIS-SEP FPR controller V2 result — 2026-09-12

Status: **CLOSED — RETROSPECTIVE DEVELOPMENT TARGET MET; NOT INDEPENDENT VALIDATION**

Study: `IRIS_SEP_FPR_CONTROLLER_V2`

Workflow run: `34636205524`

Execution commit: `7768a16cb8dcc89c781bb0cf3815010c4f3df408`

Artifact ID: `10277871843`

Artifact ZIP SHA-256: `c7756a24fee8519b5975bd6a0cac8e3f83303c88389e4a476ae500e7ea7f96ad`

Protected post-2025 outcomes accessed: **false**.

## Result

The V2 causal rolling negative-score quantile controller used the frozen Phase II `direct_onset_engineered` probabilities and changed only the alert operating policy. Its frozen controller settings were a 0.80 negative-score quantile, 120-day history, minimum 30 resolved negative rows, one-hour post-horizon label-availability buffer, and the original Phase II threshold as a floor.

On the already-inspected 2017 replay:

- FPR: **0.2392344498** (23.92%)
- POD: **1.0000**
- TSS: **0.7607655502**
- HSS: **0.0293954520**
- FAR: **0.9803921569**
- confusion: **TP=1, FN=0, FP=50, TN=159**

The source Phase II fixed-threshold operating point was:

- FPR: **0.5119617225**
- POD: **1.0000**
- TSS: **0.4880382775**
- HSS: **0.0089970892**
- FAR: **0.9907407407**
- confusion: **TP=1, FN=0, FP=107, TN=102**

Thus V2 reduced FPR by **0.2727272727 absolute** or **53.27% relative**, while retaining the sole positive onset in this replay. The requested FPR <= 0.30 development gate passed.

## Interpretation and claim boundary

This is **not** new independent evidence that the forecaster has TSS 0.761 or operational FPR 23.9%. The 2017 score period had already been inspected in Phase II V1, and V2 was explicitly created after V1 showed that a more conservative controller could reduce FPR but suppress the only positive onset. Therefore the V2 operating point is post-hoc retrospective engineering evidence.

The result establishes a useful engineering fact: the same frozen model probabilities admit a substantially lower-FPR causal alert policy without missing the one onset present in this replay. It does not solve the rare-event evidence problem: there is still only one positive onset, and FAR remains approximately 98% because the event base rate is extremely low.

No protected post-2025 outcomes were accessed. A defensible performance claim for this controller requires freezing V2 and evaluating it on genuinely uninspected or prospective data without further adjustment.
