# Operator verification card — proposed research deliverable

Status: historical verification tool; not approved for operational warning decisions. No operator adoption or economic benefit has been demonstrated.

| Decision | Appropriate question | Required companion information |
|---|---|---|
| Continue precautions during radiation exposure | How well are positive decision windows recognized? | Duration, alert burden, latency, energy threshold and spacecraft environment |
| Compare representation of physical episodes | How well is a uniformly selected episode's window recognized? | Multiplicity distribution and any-event detection reported separately |
| Issue a new warning | Was an eligible new onset predicted before it happened? | Actual issue/arrival time, lead-time distribution, missed onsets, false alerts and abstentions |

For a candidate model require: immutable issue-aligned predictions; probability and threshold; source release time and quality flags; target/event contract; matching negative cohort; explicit quiet-block definition; frozen selection history; model/version hash; all three estimands; paired uncertainty; causal input audit; alert burden; and independent validation status.

The present joint-model diagnostic has onset POD 48.24%, FPR 4.51%, FAR 88.95%. These numbers are for daily historical decision rows, not continuous alert episodes or spacecraft-specific harm. The alternate V2 controller's FPR 23.92% and FAR 98.04% illustrate why the two rates must not be confused. Do not infer a cost-saving policy without specifying and validating the decision costs and alert process.

Acceptance of a live system requires additional evidence not supplied by this project: historical issuance-compatible sources, independent prospective outcomes, verified sensor/channel mapping, reliable failure/abstention behavior, application-specific costs, and relevant operator review. A useful immediate output is a transparent comparison report that makes incompatible verification choices visible.
