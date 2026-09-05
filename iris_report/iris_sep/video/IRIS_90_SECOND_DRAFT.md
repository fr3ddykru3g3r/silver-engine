# IRIS 90-second submission narrative — draft

Not a produced video. Time the student's actual delivery before submission.
Use receipt-derived visuals and cite sources on the end card. Do not describe
software regression coverage as an independent operational experiment.

## Narration

Satellite operators need radiation warnings that are both useful and trustworthy.
Our question is whether we can forecast a new solar proton threshold crossing
within twenty-four hours, with fewer false warnings on a fair comparison.

We built IRIS-SEP and tested it chronologically. The result was not a win:
XGBoost achieved a development TSS of point two eight seven, while our stabilized
neural model reached point two five eight. Their difference was inconclusive.

The experiment revealed a problem hidden by the average score. Performance
changed sharply across time periods, and the original neural model produced
invalid numerical outputs in the latest fold. Magnetic observations were also
missing across large parts of the historical data.

We added prototype checks for evidence, timestamps, missing inputs and invalid
outputs. The checks pass our synthetic fault suite, but that does not prove
operational reliability.

The decisive next experiment uses verified forecast-time proton and X-ray
context, a reproduced competitor, and one blinded evaluation. We will report
false warnings, detection, calibration and limitations together.

IRIS is research decision support. It does not control spacecraft. Our goal is
evidence an operator can audit, including when a forecast should be withheld.
