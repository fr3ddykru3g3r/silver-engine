# Repository reconstruction

Snapshot retrieved for this audit on 15 September 2026. Branch naming is not evidence of authority; compare SHAs and ancestry. This is a filtered independent clone, not the pre-existing dirty checkout.

| Branch | SHA | Audit-only / branch-only commits |
|---|---|---|
| codex/iris-base-report-2026-08-31 | `f6f6c62af27bd1cd91acc788b385da8f85644695` | 562 / 34 |
| codex/iris-sep-award-validation-v1-20260913 | `13f692a90fc3006181d1c4488da78ff135f24f87` | 0 / 0 |
| codex/iris-sep-continuation-20260905 | `63ece2bfec5998c49687801103ba4dce305603da` | 175 / 0 |
| codex/iris-sep-episode-benchmark-v1-20260909 | `972cee3d8ddfc1c7bea2fcb2b7f261f7a758fcdb` | 54 / 12 |
| codex/iris-sep-fpr-controller-v1-20260912 | `b622c3d934271b6019cc74e78a5604869bb29681` | 36 / 0 |
| codex/iris-sep-fpr-controller-v2-20260912 | `92bf009d0b50d2a9acf7e96c40516d1e36dd7eff` | 12 / 0 |
| codex/iris-sep-onset-forecaster-phase2-20260911 | `1a318dbf8b6953d01eb76fb7b98d92266a95e900` | 43 / 0 |
| codex/iris-sep-onset-forecaster-v2-20260911-prereg | `bc577948340cb9641ac392e785f2edd9a745c144` | 54 / 0 |
| codex/iris-sep-onset-forecaster-v2-20260911 | `c943224f34993a3d3cad5a2539696e206926939f` | 54 / 7 |
| codex/iris-sep-senior-review-20260908 | `d0a7152647840e9dec3eae948b90e950ec119652` | 285 / 1 |
| iris-benchmark-seeds | `e71e9037a3a9cb888b2c3de7a8728f46b017c84f` | 571 / 44 |
| iris-benchmark-v1 | `a6811c1f19c21faf5d50f5747e7b6eb77fc92f6d` | 571 / 37 |
| iris-cdr-data | `ca23b8d092b1ce4c47e7026ff69d45a9987781a8` | 571 / 32 |
| iris-cdr-v1 | `989f4e535143a7f70fe5fb7fb7534b16c1af7bda` | 571 / 65 |
| iris-gate0-data | `7c27cdcf88d971cd056c46413d5140057b83cadd` | 571 / 17 |
| iris-generator-historical | `3c789ce7f1ea70ad818741b4caf82b6ee337bd6f` | 571 / 49 |
| iris-historical-data | `a6f1964954d06c3337f95a74829be53c5d33c91c` | 571 / 20 |
| iris-label-integrity | `cc741829756cd33b3be9ba04397f5df9d7a278d5` | 571 / 25 |
| iris-label-resolution | `6404f62c0c57fadfe015e7b141cf71485a230767` | 571 / 23 |
| iris-model-v1 | `e3a741f23ce20dbe0d31b383d4ae44bc214d2f18` | 571 / 33 |
| iris-pil-real-validation | `0344129285602e4637219ec50eef76f3a52beecf` | 571 / 31 |
| iris-v2-physics-transfer | `29fd7b7ea8ce1c23cb11321baac699eeac5add47` | 571 / 131 |
| iris-v3-cvae | `b42575f83feac1772ca692fb56a6d950488fe333` | 562 / 4 |
| main | `cdf0b2a5bf37fe793f948158adc4cf007c5141f0` | 562 / 0 |

## All pull requests in the retrieved repository listing

| PR | State | Title | Base ← head |
|---|---|---|---|
| [#1](https://github.com/fr3ddykru3g3r/silver-engine/pull/1) | closed | Electron + React skeleton for local subscription tracker (JSON storage, IPC, UI components) | main ← codex/design-electron-app-for-tracking-subscriptions |
| [#2](https://github.com/fr3ddykru3g3r/silver-engine/pull/2) | closed | Sync IRIS v2 physics-transfer project to main | main ← iris-v2-physics-transfer |
| [#3](https://github.com/fr3ddykru3g3r/silver-engine/pull/3) | open | IRIS-SEP: robust solar-radiation forecasting when data goes missing | main ← codex/iris-sep-continuation-20260905 |
| [#4](https://github.com/fr3ddykru3g3r/silver-engine/pull/4) | closed | Fix incomplete SEP outcome labels and update scientific assessment | codex/iris-sep-continuation-20260905 ← codex/iris-sep-senior-review-20260908 |
| [#5](https://github.com/fr3ddykru3g3r/silver-engine/pull/5) | open | IRIS-SEP: confirm episode-normalized onset evaluation on frozen SEP-PRISM replay | codex/iris-sep-continuation-20260905 ← codex/iris-sep-episode-benchmark-v1-20260909 |
| [#6](https://github.com/fr3ddykru3g3r/silver-engine/pull/6) | open | IRIS-SEP: Phase II genuine-onset forecaster | codex/iris-sep-episode-benchmark-v1-20260909 ← codex/iris-sep-onset-forecaster-v2-20260911 |

PR #5 is episode-benchmark → continuation, recorded as 133 commits and 110 changed files (16,559 additions / 173 deletions). It does not contain all subsequent phase-II/controller/award work. PR #6 is a separate Phase-II line. Closed does not imply merged; the retrieved closed research PRs were unmerged.

The award and episode branches diverge at `bc577948340cb9641ac392e785f2edd9a745c144`, with 54 award-only and 12 episode-only commits. The episode tip `972cee3...` corrects the NSRRD source decision by failing closed on unresolved live sensor binding. The award tip's newer-looking date does not supersede that correction. The audit records it without merging branches.

## Dependency graph and causal boundaries

Public harmonized historical observations → catalog event intervals → daily historical features and next-day occurrence labels → chronological fit/threshold/score roles → frozen model probabilities and thresholds → alerts → mapped/episode/onset measures → paired cluster draws → qualified historical claim.

The public table has 259 joint / 255 no-proton features after excluding future/label/timestamp fields. Retrospective harmonization and imputation are not proof that those fields existed at forecast issuance. Public dataset availability today does not establish historical causal availability. The no-proton comparator removes explicit proton features, not every correlated proxy.

Outer roles: fit 1986–1999 / threshold 2000–2004 / score 2005–2010; fit 1986–2004 / threshold 2005–2010 / score 2011–2017; fit 1986–2010 / threshold 2011–2017 / score 2018–the original endpoint. Strict audit horizon restriction is applied afterward without refitting. No calibration split is used in this frozen benchmark. Threshold maximizes threshold-role TSS with the declared tie rule; five XGBoost seeds are aggregated. A threshold role is not a score role, and prior public-data exposure means chronological roles are not independent discovery protection.

The catalog overlap audit checks interval mapping, onset eligibility, active windows and ambiguity. It is not a rederivation of NOAA event intervals from raw >10 MeV integral flux. The past-proton ≥10 proxy tests past-window radiation, not instantaneous physical active state.

## Workflows and artifact access

Recent-run metadata (latest 100 retrieved), artifact metadata and full-ref commit metadata were preserved locally in `audit_evidence/repository`. Fourteen ZIP artifacts were downloaded and SHA256-pinned. The evidence manifest records every ZIP identity. Successful workflow status is not by itself proof of scientific correctness; result receipts and code were compared. An alternate released-architecture comparison failed and does not establish a same-cohort superior model.

Representative immutable execution links:

- Model-free audit: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34438057070
- Frozen replay: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34438987251
- Multiplicity mechanism: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34741573375
- Source inventory: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34741130836
- Corrected NSRRD rejection: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34690384293
- Direct Phase II: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34628549602
- Controller V1/V2: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34635582111 and https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34636205524
- Separate PR #6 Phase II: https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34624151136

## Coverage qualification

The repository file inventory covers current architecture/config/tools/tests/submission files with hashes and syntax checks. START_HERE, CURRENT_STATUS, freeze/prospective/source/status documents and central evaluation implementations were examined. All remote branch tips and PRs were reconstructed. This does not mean every historical file or workflow log was read line by line, every old model rerun, or every generated artifact recovered. The pre-existing local checkout has uncommitted work beyond these remote results; it was preserved and not treated as authoritative frozen evidence.
