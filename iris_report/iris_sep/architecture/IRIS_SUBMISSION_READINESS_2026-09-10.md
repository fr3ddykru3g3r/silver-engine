# IRIS 2026 submission readiness — 2026-09-10

## Current official submission window

IRIS National Fair 2026 registration portal: **1 August 2026, 4:00 pm to 3 October 2026, 6:00 pm**.

Current submission guidance requires that the project **does not disclose school name, city name or state** in the project video, research paper, report, presentation or abstract.

## Portal-text requirements

Current IRIS submission guidance lists:

- Project Abstract: **250 words**
- Introduction & Objective: **100–150 words**
- Innovation: **50–100 words**
- Methodology: **150–250 words**
- Results and conclusions: **100–150 words**
- Acknowledgement and reference links: **50–100 words**
- Project video: **maximum 90 seconds**

Working drafts meeting those ranges are in:

`architecture/IRIS_SUBMISSION_PACK_2026-09-10.md`

## Research-paper status

Full working draft:

`architecture/IRIS_RESEARCH_PAPER_DRAFT_2026-09-10.md`

The draft includes:

- operational threshold/background;
- falsifiable research question;
- public benchmark audit;
- episode-normalized causal estimator;
- fixed-model methodology;
- physical-unit bootstrap design;
- post-result audit correction;
- main development results;
- limitations and claim boundary;
- reproducibility receipt.

Before submission, student researchers must read every paragraph and confirm they can explain it independently.

## Figure status

Judge-facing SVGs:

1. `figures/episode_benchmark_mechanism_2026-09-10.svg`
   - explains why one physical event can create several full-weight positive windows;
   - separates persistence and genuine onset.

2. `figures/public_benchmark_episode_audit_2026-09-10.svg`
   - 11,773 public windows;
   - 610 mapped positive windows / 256 physical episodes;
   - multiplicity factor 2.38;
   - 411 persistence vs 227 onset windows.

3. `figures/fixed_model_tss_by_evaluation_2026-09-10.svg`
   - compares the same fixed models under standard, episode-normalized and onset scoring;
   - keeps the five-onset-event limitation directly on the figure.

## Oral-defense status

`architecture/IRIS_JUDGE_QA_2026-09-10.md` contains the central questions judges are likely to ask, including novelty, SEPNET boundaries, TSS choice, multiplicity weighting, statistical power, post-result corrections, reproducibility and independent-vs-development evidence.

## Evidence checklist

Authoritative audit-corrected development benchmark:

- run `34371418428`
- commit `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`
- artifact ID `10112207048`
- artifact SHA-256 `418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`
- independent V2 verification: PASS
- final evidence-manifest verification: PASS

Public benchmark audit:

- upstream pinned commit `d0eb54e46b7dd6c760325e123d2ad86f9420fbff`
- workflow run `34362893938`
- artifact ID `10108593291`
- artifact digest `498104f66efb59ebd85795d628d799006f60266a8e3454bd2df3b127b988673d`

## Red-line claims

Do not submit or say:

- that all SEP papers overestimate forecasting skill;
- that SEPNET's final SEPVAL score is wrong;
- that the five-event onset sample proves a universal model ranking;
- that the project is operationally certified;
- that the project establishes state-of-the-art forecasting performance;
- that a competition or ISEF outcome is guaranteed.

## Strongest supported claim

**On the exposed cohorts studied here, conventional window-level SEP occurrence scoring can partly reward repeated representations of physical episodes and recognition of already-active storms. Physical-event normalization and causal new-onset scoring materially change measured skill for some fixed models and provide a more interpretable measurement of pre-onset warning ability.**

## Remaining legitimate work before submission

- student line-by-line review of the research paper and portal answers;
- rehearse the 90-second explanation until it is consistently below the limit without rushing;
- visually inspect exported SVG/PDF figures for legibility at presentation size;
- verify all project documents contain no school/city/state identifiers;
- keep protected post-2025 outcomes sealed;
- do not tune the five-event development cohort further;
- if additional scientific confirmation is attempted, preregister an independent extension before inspecting its outcomes.
