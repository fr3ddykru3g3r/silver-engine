# Repository cleanup — 20 September 2026

Before edits: `a6f6ba2e8cc0a6fc1dea6e449a9cfd0b379d46e1`. Remote heads and open PRs were refreshed; the scientific branch tips were unchanged from the September 15 audit.

Four branch names were removed atomically with creation of exact-SHA archive tags under `archive/2026-09-20/`. Per-ref expected-SHA leases prevented deleting a branch that advanced during cleanup. Remote tag identities and branch absence were independently checked after the push.

- `codex/iris-sep-fpr-controller-v1-20260912`
- `codex/iris-sep-fpr-controller-v2-20260912`
- `codex/iris-sep-onset-forecaster-phase2-20260911`
- `codex/iris-sep-onset-forecaster-v2-20260911-prereg`

Each is an ancestor of the retained award-validation branch; none was a head or base of an open PR. The exact original SHAs and restoration refs are in `branch_cleanup_20260920.json`. To restore a particular branch, create it at its recorded archive tag. Historical workflow branch triggers were retained as provenance; they can be used again if a branch is explicitly restored. No scientific workflow was dispatched for cleanup.

Preserved: default `main`; open PR #3/#5/#6 branches; award-validation evidence; legacy branches containing unique commits; the episode branch used by the existing dirty local working copy. Age alone is not evidence of redundancy. No branch was merged and no uncommitted user work was overwritten. The main branch remains older than the research work; this cleanup does not claim to have consolidated divergent histories into main.

The current audit branch contains the paper-ready historical evidence reference and safe fixes. Its results are separate from the frozen primary archive. Older source-equivalence corrections that live on the episode branch remain important and are documented in the audit; the branch was not deleted.
