# Request for a blinded, reproducible SEP-PRISM comparison

**Sending authorized by the user — not sent; email connection unavailable**

To: Yian Yu, University of Michigan  
Subject: Request for training-only SEP-PRISM artifact and blinded same-cohort evaluation

Dear Dr. Yu,

I am developing IRIS-SEP, a student research system for forecasting a new NOAA
operational SEP threshold crossing (>10 MeV, >=10 pfu) in the following 24
hours. I want to compare it fairly with SEPNET-PRISM without inspecting or
tuning on held-out identities or outcomes.

Would you be willing to provide or generate the following?

1. A development-only table produced on your side, excluding every held-out
   evaluation identity, with an immutable version, SHA-256, byte size, row
   count, ordered schema, units, and license/reuse terms.
2. The exact forecast issue-time convention, label definition, CLEAR catalogue
   version, treatment of intervals already above 10 pfu, and assumed
   publication latency for each modality.
3. A frozen episode-disjoint evaluation definition or, preferably, an opaque
   evaluator to which prediction probabilities can be submitted once after
   model selection is frozen.
4. SEPNET-PRISM or SEPNET-O probabilities on that identical evaluation cohort,
   together with the configuration and calibration/threshold procedure needed
   for a paired comparison.
5. Permission to publish derived metrics, small prediction/receipt files, and
   the exact attribution language you prefer. The full source dataset would not
   be redistributed.

The predeclared primary comparison is TSS. A claimed improvement would also
require a paired episode-level 95% bootstrap interval above zero, a lower
false-alarm ratio at matched detection, and no material degradation in
calibration or warning lead time. Negative or inconclusive results will be
reported.

Thank you for considering a benchmark structure that keeps the held-out set
genuinely blind.

Sincerely,

Kyros Goyal
