# IRIS-SEP two-week finish and real-world pilot plan

**Plan date:** 2026-09-11  
**Historical build baseline:** `7bd7f0659b8ce2f4310d5c83f7d81b6db439c40b`  
**Baseline source-only CI:** run `34573237824` — PASS  
**Scientific policy:** historical model development is frozen. Any later retrospective analysis requires a separately named preregistration and must not replace the frozen historical claim after score inspection.

## Mission

Run two parallel tracks without confusing their evidence classes.

1. **Competition completion:** turn the frozen historical result into a student-owned, internally consistent submission package within 14 days.
2. **Real-world reliability pilot:** package the audit/validity contribution for external workflow review and prepare approved prospective predictor collection under independent custody.

The project leads with the measurement question: **does a 24-hour SEP score represent advance warning of a new physical radiation storm, repeated representation of one physical episode, or recognition of a storm already active at issue time?**

The historical classifier remains a research comparator. The matched replay gives joint-XGBoost onset TSS `0.437`, sensitivity `48.24%`, false-alarm ratio `88.95%`, and false-positive rate `4.51%`. Proton-free XGBoost has the higher onset point estimate, but the frozen joint-minus-no-proton contrast is `-0.040 [-0.163,+0.083]`; model superiority is not established.

## Hard boundaries

- Do not inspect, count, stratify or score protected post-2025 outcomes on the development side.
- Do not tune the historical model, threshold, features or replay in response to inspected scores.
- Do not call the historical replay prospective, untouched, operationally validated or state of the art.
- Do not send company/operator outreach until the drafts have been reviewed.
- Do not start prospective collection until the applicable approval is complete and the execution manifest/custodian boundary are frozen.
- Do not convert synthetic missingness experiments into blanket operational certification. `5-20%` random transient-loss tolerance and material degradation at `40%` are research observations supplied for this pilot plan, not source-independent deployment thresholds.
- Do not promote magnetic-map reconstruction until a preregistered hidden real-map test beats persistence and preserves downstream new-onset utility.

# Track A — competition completion

## Days 1-3 — freeze the submission record

Target dates if work starts on 2026-09-11: **Sep 11-13**.

- Treat baseline commit `7bd7f...` and CI `34573237824` as the byte-identifiable historical build checkpoint.
- Assemble `submission/IRIS_2026_EVIDENCE_BUNDLE_V1/` from references to frozen evidence rather than silently copying mutable results.
- Audit every displayed denominator and metric against the authoritative replay receipts.
- Enforce the denominator guard: **228 = full-table onset windows; 85 = distinct matched replay onset episodes.** They answer different counting questions and must never be substituted for one another.
- Preserve the failed/negative freshness experiment, prior-exposure limitation, causal-feature limitation, no-purge limitation and false-alarm burden.
- Record the no-retrospective-tuning freeze.

**Gate A1:** evidence bundle manifest resolves, headline numbers agree across student-facing scaffolds, protected-data statement remains intact, and no result is relabelled as independent prospective evidence.

## Days 4-7 — student-owned final writing

Target: **Sep 14-17**.

- Each student rewrites the final abstract, paper, research plan, poster prose and 90-second spoken explanation in their own words.
- Each student completes the mastery checklist: operational target, three estimands, EMF, TSS, bootstrap unit, 228-versus-85 denominator distinction, FAR versus FPR, prior exposure, causal feature availability, missing-data boundary and prospective custody.
- Verify each retained citation against the original paper/publisher record actually read by the students.
- Complete current fair/IRIS/ISEF forms using truthful dates, roles, AI/programming support, mentor involvement and the exact status of the prospective extension.
- Re-check the live portal/rules before upload rather than relying on repository snapshots alone.

**Gate A2:** no final sentence remains that either student cannot explain and defend from the frozen evidence record.

## Days 8-11 — presentation production

Target: **Sep 18-21**.

Use two primary visuals:

1. mechanism graphic: one physical episode can create repeated positive windows and some issue times occur after threshold crossing;
2. fixed-alert comparison, titled **“Measured SEP forecast skill changes when repeated episodes and persistence are removed.”**

The central visual statement is that the **predictions and alerts are held fixed while the physical/statistical question changes**. Export the student-owned poster to print-ready PDF, test it at intended dimensions, and record an 85-90 second natural explanation rather than a memorized generated script.

**Gate A3:** print proof legible at judging distance, figures carry correct denominators, video timing passes, and every claim can be traced to a receipt.

## Days 12-14 — defense and submission gate

Target: **Sep 22-24**, or earlier if the live submission deadline requires it.

- Run three skeptical mock judging sessions with interruptions and exact-denominator questions.
- Require correct defenses of dependence, physical-unit bootstrap, FAR/FPR, missingness, prior exposure, novelty, protected custody and why the classifier itself is not deployable.
- Remove or correct any statement that cannot be defended from frozen evidence.
- Submit only after forms, citation audit, student-authorship review, print proof, video timing and evidence archive all pass.

**Gate A4:** every submission surface uses the same claim boundary and evidence class.

# Track B — SEP Forecast Validity and Evaluation Kit

## Product boundary

Package the work as an **SEP Forecast Validity and Evaluation Kit**, not a replacement operational forecaster. Its job is to expose whether a forecast record is interpretable, causally formed and auditable, and to evaluate forecasts under physical-event-aware estimands.

For each issue, the frozen interface exposes at least:

```json
{
  "issue_time": "UTC timestamp",
  "alert": true,
  "validity_state": "VALID | DEGRADED | ABSTAIN",
  "reason_codes": [],
  "input_age_seconds": {},
  "source_hashes": {},
  "model_hash": "sha256",
  "threshold_hash": "sha256",
  "ledger_previous_hash": "sha256"
}
```

For a fail-closed `ABSTAIN` record, the implementation sets `alert=null` so an invalid candidate prediction cannot be mistaken for an actionable alert.

Required fail-closed reasons include stale input, absent required feed, ambiguous proton semantics, structural unavailability, excessive transient loss, failed causal-availability receipt, ledger-integrity failure, future observation and uncertain provenance. Causal forward-fill inside a separately frozen, source-specific recovery envelope is marked `DEGRADED`, never silently upgraded to an observed value.

## Missing-data policy

The accepted structural/transient distinction remains authoritative:

- **structural unavailability:** never reconstruct and relabel as an observation; use fallback/mask-aware handling or abstain;
- **transient missingness:** causal recovery can be studied/applied only where the source regime supports the quantity and a separately frozen age/recovery limit is satisfied;
- **future observations:** forbidden;
- **severe/stale/ambiguous/provenance-uncertain cases:** `ABSTAIN`;
- **modest validated causal recovery:** `DEGRADED` with the recovery exposed.

The existing missingness contract is train-only research and contains no blanket operational percentage threshold. Therefore the supplied `5-20%` versus `40%` random-loss observation is retained as evidence context only. A real source-specific limit must be frozen before prospective execution.

Magnetic-map reconstruction remains `EXPERIMENTAL_NOT_PROMOTED` until a preregistered hidden real-map comparison beats persistence and downstream onset utility is preserved.

## Prospective execution

Before any approved prospective collection:

1. obtain the applicable fair/SRC approval for the planned extension;
2. appoint an independent adult custodian;
3. complete the execution manifest with exact source, rule/model, threshold, schema and code hashes;
4. verify causal availability for every required input;
5. begin append-only pre-issue capture on the frozen issue clock;
6. retain raw public responses, retrieval timestamps, source hashes, validity decisions, predictions and chain links before the target window occurs;
7. prevent development-side access to protected labels, episode identities and running protected event counts;
8. require the frozen information floor of at least **50 distinct onset episodes and 500 quiet blocks** before confirmatory interpretation;
9. return `INSUFFICIENT_CONFIRMATORY_INFORMATION` if the floor is not met;
10. accept the result without threshold rescue, model rescue or selective-period reporting.

The two-week sprint does **not** aim to accumulate 50 onset episodes. It prepares the governed prospective path.

## Stakeholder recruitment — prepare, do not send

Prepare a two-page operator brief, one-page validation sheet, five-minute demonstration, 30-minute workflow-interview request and narrowly framed pilot proposal. Candidate audiences include satellite operators, space-weather service providers, aviation-radiation teams, university space-weather groups and national forecasting organisations.

The interview asks:

- What operational decision changes when an SEP alert arrives?
- What false-alarm burden is tolerable for that decision?
- How much warning time is actually useful?
- What should happen when one or more feeds are delayed, stale or absent?
- What provenance, audit or reproducibility evidence is required?

**First impact milestone:** one documented external workflow review and one concrete requirement from that review incorporated into a new, explicitly versioned prospective protocol. Until then, do not claim company benefit.

# Acceptance criteria

The sprint is complete only when:

- the historical baseline remains byte-identifiable and its frozen replay receipts remain reproducible;
- no protected post-2025 outcome access has occurred;
- paper, poster, video, synopsis and PR use consistent denominators and claim boundaries;
- the new validity interface passes tests for future values, staleness, structural absence, ambiguity, ledger failure, causal forward-fill degradation, excessive loss and abstention;
- every genuine prospective record is created before its target window and later passes independent chain verification;
- physics reconstruction receives no promotion without its preregistered hidden real-map comparison;
- student ownership, citation, forms, print, timing and defense gates pass;
- impact claims remain conditional on documented stakeholder feedback or valid prospective evidence.

## End state

At day 14, the intended output is **a finished student-owned competition submission plus a reviewed, unsent external-pilot package and a tested prospective validity interface**. Independent future evidence remains future work by design; it must not be manufactured retrospectively during the sprint.
