# IRIS-SEP read-only evidence index — 2026-09-11

Use this file at judging/review time to locate frozen evidence quickly. It is an index, not a substitute for the underlying receipts.

## A. Higher-powered SEP-PRISM model-free audit

- purpose: reconstruct the 24-hour operational target and quantify multiplicity/persistence independently of model skill;
- windows: `14,464`;
- stored positives: `650`;
- reconstructed target mismatches: `0`;
- mapped positives / represented episodes: `614 / 257`;
- EMF: `2.389`;
- persistence / onset windows: `418 / 228`;
- onset-state ambiguous positives: `4`;
- separate multi-episode-overlap positives: `36`.

Frozen receipt:
- workflow run: `34438057070`;
- artifact: `10136909164`;
- artifact SHA-256: `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`;
- row-level target reconstruction: PASS.

## B. Frozen SEP-PRISM fixed-model replay

- unique scored issues: `7,558`;
- persisted prediction rows: `45,348`;
- fixed comparators: `6`;
- matched positive units: `85` onset episodes;
- matched negative units: `1,080` quiet blocks;
- bootstrap draws: `10,000` shared physical-unit draws.

Frozen primary historical contrasts:

| Contrast | Point / median summary | Paired percentile 95% interval |
|---|---:|---:|
| joint XGB episode-normalized - mapped occurrence | about `-0.104/-0.105` | `[-0.146,-0.063]` |
| joint XGB onset - episode-normalized | about `-0.183/-0.184` | `[-0.247,-0.125]` |
| past-proton proxy onset - mapped | about `-0.506/-0.508` | `[-0.567,-0.444]` |

Matched joint-XGBoost TSS:
`0.726455 -> 0.621128 -> 0.437234`.

Frozen receipt:
- workflow run: `34438987251`;
- source commit: `296f302111371e8d421fb5f2a4569ea2c06bd6e1`;
- artifact: `10137507101`;
- artifact SHA-256: `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`;
- supplied verifier: PASS;
- independent reconstruction: PASS;
- exact seeded shared-draw regeneration: PASS.

Key artifact files:
- `predictions.csv`;
- `descriptive_results.csv`;
- `per_fold_results.csv`;
- `matched_episode_sensitivity_predictions.csv`;
- `matched_episode_point_results.csv`;
- `shared_bootstrap_draws.npz`;
- `bootstrap_contrasts.json`;
- `summary.json`;
- upstream rolling table and event catalogue copies;
- evidence hash manifest.

## C. Minimal independent primary recheck

Code:
`tools/recheck_sep_prism_primary_contrasts_v1.py`.

Purpose:
- do not import the benchmark runner;
- rebuild matched confusion contributions from persisted predictions;
- apply the stored shared unit-index tensors;
- recompute the three primary paired contrast distributions and intervals.

Use this to explain why computational verification is not simply “the same scoring code ran twice.”

## D. Earlier internal development benchmark

Retained for historical/audit continuity, not the current power claim.

- 936 scored issues;
- only five distinct onset episodes;
- joint-XGB multiplicity shift `-0.133 [-0.225,-0.041]`;
- joint-XGB persistence shift `-0.400 [-0.800,-0.125]`;
- current-proton-active persistence shift `-0.567 [-0.867,-0.267]`.

Audit-corrected receipt:
- run: `34371418428`;
- commit: `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`;
- artifact: `10112207048`;
- artifact SHA-256: `418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`;
- independent V2 verification: PASS;
- final evidence-manifest verification: PASS.

## E. Post-result correction boundary

The earlier internal evidence package had:
1. an OOF summary threshold bug; and
2. insufficiently explicit labeling of descriptive vs mapped inferential cohorts.

Correction boundary:
- no refit;
- no feature change;
- no hyperparameter search;
- no threshold reselection;
- no protected-outcome access;
- legacy files preserved.

## F. Current source-only software verification

The prospective-input code, ledger code, tests and historical benchmark source all passed the repository-wide source-only CI after implementation.

Verified source-code state:
- workflow run: `34572629561`;
- commit: `93ba6de102b44ee4f8dd62a77515bc445ec6307a`;
- pandas 2.3.2 job: PASS;
- pandas 3.0.1 job: PASS;
- full Python source compilation: PASS;
- registered source-only tests: PASS;
- JSON configuration parsing: PASS;
- data-dependent-test inventory gate: PASS.

Later documentation-only commits do not alter the source code covered by this receipt; use the PR checks for an even newer whole-head verification if available.

## G. Prospective confirmation contract

Study:
`IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1`.

State:
`FROZEN_NOT_EXECUTED`.

Frozen core:
- 00:00 UTC daily issue;
- `(issue, issue+24h]` >10 MeV, >=10 pfu onset target;
- already active at issue = persistence, never onset;
- every evaluated predictor must be demonstrably available before issue;
- missing/late required input = `ABSTAIN`;
- required deterministic comparator = `past_proton_active_proxy`;
- 10,000 shared physical-unit draws, seed `20260911`;
- information floor = >=50 distinct onset episodes + >=500 quiet blocks;
- primary prospective contrast = proxy `NEW_ONSET_CAUSAL TSS - MAPPED_OCCURRENCE TSS`;
- protected labels/episode identities remain under independent custody.

Core files:
- `config/prospective_operational_confirmation_v1_preregistration_2026-09-11.json`;
- `config/prospective_operational_execution_manifest_template_v1.json`;
- `architecture/PROSPECTIVE_OPERATIONAL_FEATURE_AVAILABILITY_CONTRACT_2026-09-11.md`;
- `architecture/PROSPECTIVE_CUSTODIAN_HANDOFF_2026-09-11.md`;
- `tools/run_custodian_prospective_episode_evaluation_v1.py`.

## H. New predictor-side prospective implementation

- source contract: `config/prospective_live_input_sources_v1.json`;
- deterministic rule: `config/past_proton_active_proxy_rule_v1.json`;
- pre-issue source capture: `tools/capture_prospective_input_snapshot_v1.py`;
- prediction/ledger builder: `tools/build_past_proton_proxy_prediction_v1.py`;
- ledger verifier: `tools/verify_prospective_prediction_ledger_v1.py`;
- source-only tests: `tests/test_prospective_input_pipeline_v1.py`;
- manual predictor capture workflow: `.github/workflows/iris-sep-prospective-predictor-capture.yml`;
- implementation note: `architecture/PROSPECTIVE_PIPELINE_IMPLEMENTATION_2026-09-11.md`.

This layer reads only public predictor inputs. It cannot set `features_verified_causal=true`, cannot inspect protected outcomes and cannot make a prospective-skill claim.

## I. Judge-facing figures

Headline:
- `figures/episode_benchmark_mechanism_2026-09-10.svg`;
- `figures/sep_prism_fixed_model_tss_2026-09-10.svg`.

Supporting/historical:
- `figures/public_benchmark_episode_audit_2026-09-10.svg`;
- `figures/fixed_model_tss_by_evaluation_2026-09-10.svg` (five-onset internal study; label as historical if used).

Plot generator:
`tools/plot_sep_prism_external_replay_v1.py`.

## J. Claim boundary card

Supported now:
- repeated physical-event representation is present in current public SEP rolling data;
- persistence and new onset are quantitatively different parts of the daily occurrence problem;
- fixed-model measured skill changes materially for the preregistered historical replay when these mechanisms are separated;
- physical-event/onset-specific reporting is scientifically justified as complementary evaluation.

Not supported now:
- universal bias in previous SEP studies;
- invalidity of SEPNET's final SEPVAL score;
- independent prospective confirmation;
- state-of-the-art model superiority;
- operational deployment/certification;
- economic savings;
- award outcome.
