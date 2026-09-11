# IRIS 2026 submission evidence bundle v1

**Evidence baseline:** `7bd7f0659b8ce2f4310d5c83f7d81b6db439c40b`  
**Baseline source-only CI:** `34573237824` — PASS  
**Evidence class:** strong methodological confirmation on previously development-exposed public historical data.  
**Protected outcomes:** post-`2025-09-10T00:00:00Z` outcomes remain sealed from the development side.

This directory is the submission audit index. It deliberately **references** frozen repository files and immutable workflow/artifact receipts instead of making mutable copies that could drift. `FROZEN_RESULTS.json` is the machine-readable source of truth for the final headline counts, TSS values, bootstrap contrasts, operating characteristics, artifact hashes and prospective-state boundary.

## Headline scientific record

- 14,464 SEP-PRISM daily windows; 650 stored positives; zero stored-versus-reconstructed target mismatches.
- 614 uniquely mapped positive windows represent 257 physical SEP episodes; EMF = 2.389.
- Full model-free audit: 418 persistence windows, **228 onset windows**, four ambiguity cases.
- Fixed replay: 7,558 chronological score issues.
- Matched inference: **85 distinct onset episodes + 1,080 quiet blocks**, 10,000 shared physical-unit bootstrap draws.
- Joint-XGBoost matched TSS: `0.726 -> 0.621 -> 0.437` for mapped occurrence -> episode-normalized occurrence -> causal new onset.
- Joint-XGBoost onset confusion: TP=41, FN=44, FP=330, TN=6,984; sensitivity 48.24%; false-alarm ratio 88.95%; false-positive rate 4.51%.
- Joint-XGBoost minus proton-free-XGBoost onset contrast: `-0.040 [-0.163,+0.083]`; reliable model superiority/rank reversal is **not established**.

## Denominator guard

**228 is not 85.**

- `228` = onset **windows** in the full 1986–2025 model-free table.
- `85` = distinct onset **physical episodes** in the frozen matched fixed-model replay inferential cohort.

Every paper, poster, video, synopsis and oral answer must preserve that distinction. `tests/test_submission_frozen_results_contract.py` now regression-tests this guard and the other headline submission values.

## Referenced submission scaffolds

- `architecture/IRIS_RESEARCH_PAPER_DRAFT_2026-09-10.md` — research-paper review scaffold; students must rewrite/check final prose.
- `architecture/IRIS_SUBMISSION_PACK_2026-09-10.md` — abstract/portal/video evidence scaffold.
- `architecture/ISEF_RESEARCH_PLAN_SCAFFOLD_2026-09-11.md` — research-plan scaffold.
- `architecture/POSTER_BLUEPRINT_2026-09-11.md` — poster structure.
- `architecture/VIDEO_STORYBOARD_2026-09-11.md` — 90-second structure.
- `architecture/IRIS_JUDGE_QA_2026-09-10.md` — skeptical defense set.
- `architecture/STUDENT_OWNERSHIP_DEFENSE_CHECKLIST_2026-09-11.md` — student mastery gate.
- `architecture/ISEF_2027_FORMS_AND_APPROVAL_MATRIX_2026-09-11.md` — forms/approval checkpoint; live rules must still be rechecked before submission.

## Two headline figures

1. `figures/episode_benchmark_mechanism_2026-09-10.svg`
2. `figures/sep_prism_fixed_model_tss_2026-09-10.svg`

Recommended fixed-alert figure title for the final student-designed display:

**Measured SEP forecast skill changes when repeated episodes and persistence are removed.**

## Reproducibility receipts

- Model-free confirmation workflow: `34438057070`; artifact `10136909164`; archive SHA-256 `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`.
- Fixed-model replay workflow: `34438987251`; source commit `296f302111371e8d421fb5f2a4569ea2c06bd6e1`; artifact `10137507101`; archive SHA-256 `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`.
- Baseline source-only workflow: `34573237824` on `7bd7f0659b8ce2f4310d5c83f7d81b6db439c40b` — PASS.
- Final technical freeze receipt: `architecture/FINAL_TECHNICAL_RESULTS_AND_FREEZE_RECEIPT_2026-09-11.md`.
- Machine-readable final results contract: `FROZEN_RESULTS.json`.

## Required limitations retained

- historical public files were already development-exposed;
- publication-time/issue-time availability is not proven for every historical predictor;
- no programmed purge existed at every historical role boundary;
- the classifier itself has an unacceptable operational false-alert burden;
- the negative freshness experiment remains preserved;
- prospective confirmation is frozen but not executed;
- no state-of-the-art, deployment, economic-benefit, company-benefit or award claim is supported.

Read `FROZEN_RESULTS.json`, `MODEL_DEVELOPMENT_FREEZE.md`, `NEGATIVE_RESULT_LEDGER.md`, `PROTECTED_DATA_STATEMENT.md`, `SUBMISSION_GATE_CHECKLIST.md` and `MANIFEST.json` before final submission.
