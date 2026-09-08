# IRIS-SEP senior review — 8 September 2026

Reviewed base: `27692b5fac31d2e0094a66e0e2f3caa09a401a2a`.
Scope: inventoried 138 commits / 102 changed files since `90acde0`; reconstructed every changed file and verified its Git blob hash. Reviewed critical evaluation, inference and provenance paths, source tests, V3 development results and the latest source-audit artifact. This is an internal review, not external validation or a full retraining reproduction.

## Decision

Continue the research question: **can a forecaster retain useful event detection when a feed disappears, without silently inventing measurements?** Keep V3 as the frozen development candidate. Stop V4/backbone work and additional self-assigned maturity scores. The next deliverable is one complete, independently reproducible causal forecast and its correctly observed outcome, followed by a preregistered comparison with simpler alternatives.

The project has executable forecasting and fallback models. It has not established a large predictive advantage or operational usefulness. No award outcome follows from internal scores or a high test count.

## Evidence rechecked

V3 replay run: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34138370215
Archive SHA256: `0d6c38cc34d1d55604bade0611f37d5635620ad6112d35a9f193ef51b39b2390` (recomputed).

Score cohort: 3,219 rows / 21 positives; already inspected and used in development.

| V3 state | TSS | TP | FP | Brier | Captured in top 161 rows |
|---|---:|---:|---:|---:|---:|
| NO_XRS | 0.430492272 | 13 | 603 | 0.006434313 | 8 / 21 |
| NO_PROTON | 0.512999196 | 16 | 796 | 0.006417693 | 6 / 21 |

These values were independently recalculated from artifact predictions using the stored MAX_TSS thresholds. V1 has the same top-5% captures. NO_PROTON removes 18 false positives (814 to 796), approximately 2.2%, without changing its 16 detections. NO_XRS increases detections from 11 to 13 but also increases false positives from 498 to 603. Neither is an astronomical operational improvement. Existing paired intervals and their development-only interpretation remain preserved in BLACK_BOX_VALIDATION_2026-09-07.md; this review did not rerun those bootstrap intervals.

Top-5% selection over nine years is a retrospective ranking diagnostic. Its cutoff cannot be known at the first forecast issue. A prospective analyst-budget policy must fix its rule using prior data and report realized workload and capture; it cannot select the future top 5% of days.

The V3 retention allowance of +0.005 Brier is large relative to baseline Brier around 0.0064. Passing this gate is not evidence of material noninferiority. Do not edit historical criteria after seeing results; replace this interpretation in future preregistration with justified margins and interval-based comparisons.

## Fresh-source evidence

Audit run: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34191713209
Archive SHA256: `781743978753c207b64a473fb946f2a5078918f65502fb534f2b0c69068ae8f5` (recomputed).

The audit explicitly reports `fresh_forecast_probability_emitted=false`. NOAA proton observations contain a maximum gap of 6,900 seconds (115 minutes). JSOC NRT CEA SHARP lacks `CMASKL`, `MEANGBL` and `USFLUXL`; definitive CEA metadata lists them, but metadata existence does not establish issue-time availability. The job completed successfully as an audit while reporting the JSOC interface failure. Do not present green CI as a successful fresh forecast.

Do not fill these missing fields with zero, substitute similarly named quantities, or silently swap definitive and NRT products. By 10 September, either establish a source-equivalent causal interface or declare the current frozen model retrospective-only. A reduced causal-interface model would be a separately preregistered experiment with new training/evaluation, not the same frozen model with a relabelled input.

## Critical fixes made in this review

1. A two-sample proton series ending five minutes after issue previously yielded a negative 24-hour label. The new V2 outcome receipt requires fresh issue support and complete, finite, nonnegative samples at no more than five-minute gaps through the exact horizon endpoint. Missing or immature windows stay unresolved, including positive windows with incomplete support. Duplicate timestamps are rejected.
2. Outcome receipts are checked before model or comparator scoring. Duplicate forecasts/issue times and model-package changes cannot inflate or mix cohort support. Missing label rows fail; unresolved rows are counted separately.
3. Forecast seal semantic fields are revalidated, including probability ranges and the derived horizon. A matching self-hash alone does not make an invalid probability acceptable.
4. Twenty positive rows cannot automatically produce an independent-evaluation claim. The output now explicitly states independence is unverified. A self-reported timestamp plus SHA256 is not a trusted pre-outcome timestamp or source attestation.
5. Review enrichment uses the actual rounded review fraction, and numerical threshold metrics are explicitly distinguished from permission-filtered alerts.

Compatibility: old V1 label receipts are rejected by current scoring. Preserve old artifacts, regenerate V2 labels from adequately covered raw outcomes, and disclose this correction. Sampled crossings still require equivalence validation against the catalogue event definition; these changes do not establish that equivalence.

## Delivery priorities through 20 September

- **8–10 September:** merge the evaluation correction after source CI; archive verified result/model artifacts outside ordinary Git; resolve the exact causal source interface and witness mechanism. Replace stale current-status pointers. Keep full-data and missing-data states separate.
- **11–13 September:** complete one acquisition → causal aggregation → load-only model → admission → externally timestamped prediction → mature outcome replay. Witness the exact forecast digest before the outcome through a trusted server-side publication record; commit author dates and caller timestamps do not count. Record outages and abstentions as well as forecasts. These initial cases prove integration, not skill.
- **14–16 September:** lock the scientific protocol before evaluation. Compare V3 against V1 remaining-sensor fallback, no-fill and a simple abstention policy on identical cases; include the full-data teacher only as an unavailable-information reference. Report actual alerts, coverage, missed events, false alerts, Brier and uncertainty. Keep analyst-budget diagnostics separate from deployed rules. Blind evaluation must be custodian-controlled; no publisher reply means no assumed final test.
- **17–20 September:** generate paper tables and plots from receipts; finish synopsis, paper, recorded replay and 90-second video. Lead with the measurable missing-feed problem, the evidence and limitations. Document student contributions and external assistance accurately. An honest negative or inconclusive result remains in the paper.

Do not expect a few new daily forecasts to provide 20 independent events by the submission date. If no fresh evaluation is feasible, submit a retrospective reliability study with explicit limitations and continue prospective collection afterward. Do not turn inspected history into a new holdout or count duplicate/overlapping event windows as independent events.

## Remaining boundaries

Source-registry matching is an implemented consistency check, not independent proof of provider bytes or acquisition time. Sealed forecast builders still take hashes/probabilities as inputs; an authenticated end-to-end collector and externally witnessed ledger must establish their truth. Comparator scoring still requires an explicit common-cohort audit and verification that caller-supplied paired probabilities match sealed forecasts. Model artifact portability is supported by the earlier black-box report; this review did not repeat model loading.

This is a credible research direction with substantial engineering progress. A winning-level scientific result remains unproven. The deciding work is causal evidence and a fair useful-outcome comparison, not another model family or another subjective score.
