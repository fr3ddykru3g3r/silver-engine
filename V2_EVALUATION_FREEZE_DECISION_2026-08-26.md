# V2 Phase-II evaluation freeze — 2026-08-26

## Authoritative outer evaluation artifact
Use GitHub Actions run **32997936465**, artifact **iris-v2-frozen-rolling-origin-folds** (artifact id 9617314575), as the sole Phase-II historical outer-evaluation definition.

Reason: it constructs folds from the **192,235 eligibility-filtered/censored observations** and 2,195 eligible connected physical-region groups. Boundaries are defined from connected-region chronology without labels, with a 36 h boundary purge. The four disjoint future evaluation blocks contain 45, 39, 60, and 29 M1+-positive physical groups respectively (reported only after boundaries were frozen). Structural audit showed zero train/evaluation overlap and no reuse of an outer-evaluation group.

## Deprecated exploratory fold artifact
Run **32998247726** / `iris-v2-rolling-folds` is **not authorized for forecasting evaluation**. Its first implementation read the broader `training_manifest.csv.gz` directly and therefore counted ~646k rows rather than restricting to the integrity-locked eligible/censored population. It remains an engineering record only and must not be used for model results, tuning, plots, or claims.

## Phase-I single test
All previously exposed single-split test results remain exploratory Phase I. They may be reported transparently as motivating diagnostics but may not be used for Phase-II model/physics selection.

## Freeze rule
No Phase-II outer evaluation block may be inspected for model selection. Generator architecture, generic-fidelity stabilization, HJ/PIL coefficients, augmentation count, downstream architecture, threshold-selection rule, and destructive-control definitions must be selected using training/internal-validation information only. After freezing these choices, each outer fold is evaluated once and all folds are reported.
