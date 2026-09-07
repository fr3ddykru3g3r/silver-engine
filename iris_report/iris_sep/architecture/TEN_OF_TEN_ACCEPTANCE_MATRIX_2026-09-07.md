# IRIS-SEP — objective 10/10 acceptance matrix

`10/10` is not a subjective score and it is not a promise of competition success. A category receives `10/10` only when the evidence below is complete. Anything else remains explicitly below gate.

| Category | 10/10 acceptance gate | Current status |
|---|---|---|
| **Problem usefulness** | Named operational user; real failure mode documented from primary sources; output supports a concrete decision without requiring an autonomous spacecraft command. | **Near gate.** Operator problem is well grounded: forecast trust under stale/missing/incompatible space-weather inputs. Need at least one external operator/user review or equivalent requirements validation. |
| **Scientific question** | One falsifiable primary question, frozen target, predeclared controls, negative outcomes preserved. | **Gate met for current development study.** NEW >10 MeV, >=10 pfu crossing in 24 h; missing-data reliability question is falsifiable. |
| **Input causality/provenance** | Every predictor has observation/publication time and native/reconstructed lineage; no future-fitted source harmonization enters a prospective claim; UNKNOWN fails closed. | **Not met.** Released aggregate table contains interpolation/backcast provenance that prevents a strict prospective native-input claim. Requires a causal high-cadence/source-native pipeline or a narrower retrospective claim. |
| **Forecast skill** | Fresh independent cohort; paired advantage over a credible same-cohort comparator has positive 95% interval; lower/equal FAR at matched detection; no material calibration/lead-time degradation. | **Not met.** Development evidence is encouraging but inspected; FAR is still extremely high. Publisher/blinded/fresh evaluation remains required. |
| **Missing-data reliability** | Event-bearing and quiet outages; coverage + misses + false alerts + calibration; explicit modality/horizon bounds; independent metric audit; unsupported states fail closed. | **Strong development evidence, not final gate.** Quiet + event-terminal experiments independently audited. Availability-conditioned fallback is the next candidate. Fresh/real outage replay still required for 10/10. |
| **Recovery quality** | Recovery tested against deliberately hidden truth; simple baselines included; reconstruction only promoted if it beats simpler methods downstream and does not worsen calibration; uncertainty/horizon frozen. | **Not met.** Forward-fill is best simple probability-preserving arm; median/no-fill rejected as universal defaults. Reduced physics has not earned promotion; aggregate-interface tests do not establish raw-sensor reconstruction accuracy. |
| **Software/reproducibility** | Clean checkout passes source tests across pinned + portability environments; immutable receipts/hashes; no data absence masquerades as software failure; reference result independently recomputable. | **Near gate.** Dual-pandas source CI and independent audits exist. Final load-only model package and reference-prediction reproduction are still required. |
| **Deployability** | Versioned model bundle with all specialist models, feature order, fusion/calibration/threshold parameters and hashes; load-only inference; admission layer consumes real inference provenance; tamper failures tested. | **Not met.** This is the main Days 9–14 engineering task. |
| **Comparator fairness** | Same target/cohort/eligibility/issue-time information for named external comparator or explicitly contextual comparison; no cross-paper leaderboard claim. | **Partially met.** Released SEPNET-PRISM architecture was tested same-cohort under IRIS recipe. NOAA remains contextual until issue-time/horizon alignment is fair. |
| **Operator safety contract** | Observed/alternate/recovered/degraded/abstain states are machine-readable; no reconstructed value is relabelled observed; unsupported states cannot emit NORMAL; adversarial/tamper tests pass. | **Strong development gate.** Validity-envelope synthetic fault tests are excellent; operator missing-data resolver exists. Must bind it to the promoted load-only inference bundle for 10/10. |
| **Submission/presentation** | Every headline number generated from verified receipts; limitations and negative results included; synopsis, paper, figures and 90-second video agree with current evidence. | **Not met.** Drafts exist but must be regenerated after final evidence/package freeze. |

## Stop rules

1. Do **not** improve a score by redefining the target, moving outage blocks, dropping bad seeds, changing the primary threshold after inspection, or opening the locked test.
2. Do **not** call a development result fresh evidence.
3. Do **not** call aggregate-interface forward-fill a reconstruction of the underlying raw five-minute physical signal.
4. Do **not** promote physics because it is more impressive; it must beat the causal simple control.
5. Do **not** add a larger architecture until a new information source or a demonstrated residual failure justifies it.
6. Do **not** claim company/NOAA/SEPNET superiority unless the comparison is same-cohort and uncertainty clears the frozen gate.

## Highest-leverage sequence to close the remaining gates

1. Complete and audit the preregistered **availability-conditioned specialist fallback**. Prefer it to imputation where it passes.
2. Freeze `operator_missing_data_policy_v1` from evidence; unsupported state = ABSTAIN.
3. Export the complete promoted model as a checksum-addressed **load-only bundle** and prove reference prediction reproduction.
4. Connect the admission/validity layer to that exact bundle and add tamper/missing/stale/recovered demonstration replays.
5. Build the strictly causal source-native/high-cadence XRS/proton pipeline needed to remove the aggregate-table provenance limitation.
6. Obtain genuinely independent evaluation or clearly submit as a retrospective development study if unavailable.
7. Regenerate all paper/video tables directly from receipts.

The project is allowed to become `10/10` only by closing these gates, not by increasing model complexity or wording claims more aggressively.
