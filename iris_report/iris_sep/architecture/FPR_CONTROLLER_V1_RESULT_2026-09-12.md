# IRIS-SEP FPR controller V1 result — 2026-09-12

Status: **CLOSED — FPR TARGET MET, DETECTION UTILITY FAILED**

Workflow run: `34635582111`

Execution commit: `6fa43da657879f351552ffb416c9f48df328214f`

Artifact ID: `10277940827`

Artifact ZIP SHA-256: `6acce6e7916f8aaaf735fd786d59b3d5283391b734d0bab0f438d14ffe25701f`

Protected post-2025 outcomes accessed: **false**.

## Result

The causal rolling negative-score quantile controller reduced retrospective 2017 FPR from **0.5119617225** to **0.1961722488**, an absolute reduction of **0.3157894737** and a relative reduction of **61.68%**.

However, it also suppressed the only positive onset in the 210-row score cohort:

- controller TP=0, FN=1, FP=41, TN=168
- POD=0.0
- FPR=0.1961722488
- FAR=1.0
- TSS=-0.1961722488
- HSS=-0.0093842985

The source Phase II fixed-threshold model had TP=1, FN=0, FP=107, TN=102, POD=1.0, FPR=0.5119617225 and TSS=0.4880382775.

## Interpretation

V1 proves that the false-positive rate can be pushed well below 30% with a causal adaptive threshold, but the 0.85 rolling negative-score quantile was too conservative at the sole positive event. Therefore V1 is not an acceptable operating point.

The 2017 score period was already inspected before this controller work. Any follow-up controller tuning on 2017 is post-hoc engineering only and cannot be represented as held-out or independent validation. Protected post-2025 outcomes remain sealed.
