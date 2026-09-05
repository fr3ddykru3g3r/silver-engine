# SOL continuation prompt

Continue IRIS-SEP in https://github.com/fr3ddykru3g3r/silver-engine on
`codex/iris-sep-continuation-20260905`. Inspect the PR and head SHA first; do not
assume these changes are on main. Do not restart or ask for recorded information.

Read `iris_report/iris_sep/START_HERE.md`, `architecture/PUBLICATION_REVIEW_2026-09-05.md`,
`architecture/CONTINUATION_STATUS.md`, `FUTURE_PLAN.md`, the active benchmark and
evaluation JSON contracts, and `evidence_checkpoint/INDEX.json`.

Objective: a scientifically defensible daily probability of a NEW >10 MeV,
>=10 pfu crossing in 24 hours, with demonstrated benefit to operator review.
No award, ten-award, breakthrough, economic-savings or industry-superiority promise.

Current evidence: legacy monitor XGBoost .257, compact .232, fixed blend .276
(inconclusive; worse matched-detection FAR). New train-only rolling diagnostic:
XGBoost .287, elastic net .276, signed-log compact .258; paired intervals cross
zero. Original compact failed latest fold with nonfinite logits. No final
NEW-crossing benchmark or published SEPNET-O equivalence exists. No locked data
was accessed. The synthetic 10,000-trial fault results are regression coverage
of hand-authored cases, not independent operational performance or novelty.

Do these next, in order:
1. Inspect source, current Git/PR status and local artifacts. Run source-only
   checks; run full artifact verification only when dependencies/data exist.
   Missing artifacts are not scientific failures. Record commands and hashes.
2. Diagnose the nonfinite checkpoint with a controlled layer-level replay.
   Earlier code-mutation and overflow explanations were unproven. Check feature
   support, missing masks, scaling and model all-missing output behavior.
3. Bind admission V2 policy and inference arrays to immutable evidence. The
   current caller-supplied policy/arrays leave the prototype incomplete. Use
   independent adversarial tests and meaningful coverage/false-warning metrics.
4. Inspect any user-supplied publisher reply. The training-only/blinded request
   has already been sent; timestamp/id unknown, no reply supplied so far. Never
   resend. Do not send any new email or other message without authorization.
5. Final training waits for verified training-only data, crossing/episode
   semantics, latency/licensing and reproduced comparator. Never download a
   mixed train/test table or access locked identities/outcomes during development.
6. With safe data, freeze a bounded train-only chronological batch: strong fixed
   classical models, faithful comparator, compact baseline, proton context, then
   XRS. Freeze exact feature/config/fold hashes before running. No further tuning
   on inspected monitor or inner score blocks. Keep failed experiments.
7. Freeze candidate before blinded evaluation; preserve every existing gate.
8. Turn verified receipts into a synopsis, full research paper, figures and a
   90-second video. Existing paper/narratives are drafts. Record student and
   tool contributions truthfully.

Execution: local-first; Colab only after a measured heavy-compute need. No
computer-use/browser automation. Use at most three Luna subagents concurrently
for independent bounded tasks with disjoint write scopes; SOL owns integration
and claims. Never store a JSOC email in code, reports, notebooks or receipts.

Storage: use API or shallow blob-filtered sparse checkout. Compare local and
remote before synchronizing; don't assume either is newer. No force push,
history rewrite, auto-merge, blind staging, or full-history clone. Large data,
models and notebook outputs stay outside ordinary Git with provenance hashes.
Original workspace has no .git; don't recreate a large one there. Preserve root
iris-model and namespaced historical snapshots.

Cleanup: the user wants old versions removed after safe preservation. Inventory
exact paths, references and sizes. Delete only verified backed-up, unreferenced
duplicates or reproducible caches; write a cleanup manifest with recovery source.
Never delete the only copy of artifacts or files needed by pinned receipts.
If safe backup isn't available, preserve and report the storage blocker.

Before ending: publish reviewed source and small receipts on a codex branch,
open/update an unmerged PR, verify remote head and changed paths, and update
START_HERE/status with results, exact tests, artifact locations, remaining
blockers and a precise next action. Do not claim uploaded artifacts that were
excluded. The user has explicitly authorized GitHub publication to this repo.
