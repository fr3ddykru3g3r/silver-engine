# IRIS 2026 submission readiness — updated 2026-09-11

## Current scientific readiness

The project is now centered on the SEP-PRISM episode-normalized causal benchmark, not the older five-onset internal experiment and not a state-of-the-art forecasting-model claim.

Current historical confirmation:

- 14,464 pinned daily windows; 650 positives; zero reconstructed-target mismatches;
- 614 uniquely mapped positive windows / 257 represented physical episodes = EMF 2.389;
- 418 already-active persistence windows and 228 onset windows in the full model-free audit;
- 7,558 chronological fixed-model replay issues;
- matched inference on 85 onset episodes and 1,080 quiet blocks;
- 10,000 shared physical-unit bootstrap draws;
- all three frozen primary directional intervals below zero.

Evidence class: **strong methodological confirmation on previously development-exposed public historical data**. Protected post-2025 outcomes remain sealed.

## Submission status

The repository contains scientifically mature evidence, figures and rehearsal material, but generated prose is **not a substitute for student-authored final submission text**. The final portal abstract, research-plan text, poster and oral explanation must be checked and expressed in the student researchers' own words, with AI/programming support disclosed under the rules applicable to the ISEF-bound project.

## Current IRIS portal constraints already tracked by the project

Working guidance records:

- Project Abstract: 250 words
- Introduction & Objective: 100–150 words
- Innovation: 50–100 words
- Methodology: 150–250 words
- Results and conclusions: 100–150 words
- Acknowledgement and reference links: 50–100 words
- Project video: maximum 90 seconds
- do not disclose school name, city or state in the submission material

Before final upload, re-check the live IRIS portal because competition instructions can change.

## Category positioning

Recommended ISEF-facing category: **Physics and Astronomy (PHYS)**. The strongest subcategory framing is **Astronomy and Cosmology** if the form asks for the physical domain; **Theoretical/Computational Physics** is a defensible secondary fit if the form emphasizes method. Systems Software is weaker because the contribution is a scientific measurement/evaluation benchmark, not a software-engineering product.

## Figure status

Judge-facing current figures:

1. `figures/episode_benchmark_mechanism_2026-09-10.svg` — visual mechanism: one physical episode can generate multiple positive windows and persistence can be confused with onset.
2. `figures/sep_prism_fixed_model_tss_2026-09-10.svg` — current higher-powered matched replay: joint XGBoost 0.726 -> 0.621 -> 0.437, with no-proton and past-proton comparators.
3. `figures/public_benchmark_episode_audit_2026-09-10.svg` — historical external public audit retained as supporting evidence.
4. `figures/fixed_model_tss_by_evaluation_2026-09-10.svg` — earlier five-onset internal experiment; retain only as historical/development context and label it clearly if displayed.

The first two should carry the presentation. The old five-onset figure should not be the headline after the 85-episode replay.

## Oral-defense status

`architecture/IRIS_JUDGE_QA_2026-09-10.md` is now a higher-powered replay defense set. It explicitly covers:

- novelty without overclaiming;
- 228 onset windows versus 85 matched onset episodes;
- why physical-unit bootstrap is preferable to row bootstrap/ANOVA here;
- negative-block choice;
- causal feature availability;
- lack of programmed temporal purge;
- false-alarm ratio versus false-positive rate;
- exposed historical versus independent evidence;
- post-result correction integrity;
- frozen prospective falsification path.

## Reproducibility status

SEP-PRISM model-free confirmation:

- workflow `34438057070`
- artifact `10136909164`
- SHA-256 `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`

SEP-PRISM fixed-model replay:

- workflow `34438987251`
- artifact `10137507101`
- SHA-256 `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`
- supplied verifier: PASS
- independent reconstruction: PASS

An additional minimal recheck script is stored at `tools/recheck_sep_prism_primary_contrasts_v1.py`. It imports no benchmark-runner code and recomputes the three primary paired bootstrap contrasts directly from the persisted matched predictions and shared draw tensors.

Final source-only workflow at the current PR head lineage also passes under the registered environments.

## Compliance and forms checkpoint for an ISEF-bound project

Current methodology is software/data analysis of public space-weather data. On that description it does **not** involve human participants, vertebrate animals, potentially hazardous biological agents or hazardous chemicals/devices. Therefore those specialized approval forms are not triggered unless the methodology changes.

The team must still complete the universal current-year ISEF forms and disclosures required by its affiliated fair. The current 2027 forms set includes Forms 1, 1A, 1B and Student Support Disclosure Form 2A for all projects. There is no current “Form 8” in the 2027 forms list. Form 7 is relevant only if the work is a continuation from a prior competition year; RRI/QS forms apply only if an institution or qualified-supervisor situation is actually involved.

**Critical timing rule:** if the sealed prospective custodian phase is intended to become part of the same ISEF project after the affiliated fair, its phase/method must already be included in the approved research plan before the affiliated competition. Do not add a materially new study after competing.

## Red-line claims

Do not submit or say:

- that all SEP papers overestimate forecasting skill;
- that SEPNET's final published SEPVAL score is wrong;
- that the historical replay is untouched independent validation;
- that joint XGBoost is superior or proton history is universally harmful;
- that the replay demonstrates operational usefulness;
- that a competition outcome is guaranteed.

## Strongest supported claim

**On a preregistered historical replay, conventional SEP occurrence scores materially changed when repeated physical-event representation and already-active persistence were separated while fixed predictions and alerts were held unchanged. Physical-episode-normalized and causal new-onset reporting therefore provide complementary, more interpretable measures of pre-onset warning ability.**

## Remaining legitimate work before submission

- student line-by-line rewrite/ownership check of all final submission prose;
- complete the current ISEF/affiliated-fair forms and support disclosure accurately;
- print-test and simplify the two headline figures at judging distance;
- rehearse the 90-second explanation to time without rushing;
- memorize denominator distinctions and the FAR/FPR distinction;
- preserve the frozen artifacts and current PR/CI receipts;
- keep protected post-2025 outcomes sealed;
- do not tune the historical replay after score inspection;
- execute the prospective confirmation only through the frozen custodian protocol after all causal feature-availability gates pass.
