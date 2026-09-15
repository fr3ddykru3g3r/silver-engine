# Figure interpretation

All three graphics use the new strict preboundary historical diagnostic, not the original complete frozen endpoint. They are generated with AI-assisted deterministic Matplotlib code in `tools/render_audit_figures_v1.py` and should retain that disclosure when used.

1. `01_fixed_predictions`: TSS point estimates for three views of the same fixed forecasts. Connecting lines indicate a change in evaluation, not a learning curve. No independent superiority or onset rank reversal is established.
2. `02_decomposition`: exact multiplicity and persistence contributions to mapped-minus-onset TSS. Negative values are permitted. Attribution is algebraic, not a causal intervention on physical features.
3. `03_temporal_intervals`: paired bootstrap medians and percentile 95% intervals for joint-model contrasts with positive episodes grouped by 27-day, 90-day, quarter and year bins; 10,000 draws each. Historical fixed-forecast uncertainty; no calibrated null p-values.

PNG (300 dpi), vector PDF and SVG versions are supplied. The student must independently interpret them and author submission captions. These notes are internal technical descriptions.
