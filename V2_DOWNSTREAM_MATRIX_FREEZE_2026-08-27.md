# V2 downstream matrix freeze — 2026-08-27

This is the authoritative downstream arm definition for the locked v2
experiment. It must be reviewed before any outer-test metrics are inspected.
The older morphology and six-arm smoke artifacts cannot substitute for this
matrix.

## Arms

| science arm | implementation arm | training additions | classification weighting |
|---|---|---|---|
| `R` | `real` | none | none |
| `Rw` | `real_weighted` | none | balanced positive weight |
| `D` | `duplicate` | exact count of duplicated real positives | none |
| `L0` | `synthetic` + `base` manifest | exact synthetic positive count | none |
| `L2` | `synthetic` + `hj` manifest | exact synthetic positive count | none |
| `L3` | `synthetic` + `hj_pil` manifest | exact synthetic positive count | none |

`pil`-only is a mechanistic factorial diagnostic, not a primary science arm
and cannot replace `L3`.

## Frozen controls

All six arms use the same real training/validation/test subsets, preprocessing,
`FlareCNN` architecture, optimizer, learning rate, focal-loss gamma, number
of optimizer steps, evaluation schedule, and replicate seeds. Only the listed
training addition or the intentional `Rw` class-weighting policy changes.

For `D`, `L0`, `L2`, and `L3`, the number of added positive examples is
identical. A mismatch is a hard failure.

For `Rw`,

\[
\mathrm{pos\_weight} =
\frac{N_{\mathrm{real\ negative\ train}}}
{N_{\mathrm{real\ positive\ train}}}.
\]

## Evaluation lock

- Threshold selection uses validation TSS only and is frozen before test use.
- Test data are evaluated once after the threshold is frozen.
- Primary metric: TSS.
- Bootstrap unit: connected active-region group, never an image window.
- Primary comparisons: `Rw - R`, `D - R`, `L0 - D`, `L2 - D`, and `L3 - D`.
- No later threshold, seed, preprocessing, exclusion, or arm selection may
  replace this run after test metrics are seen.

## Gate

This matrix is not authorized until the corresponding generator passes the
independent train-only fidelity/physics gate and the remaining administrative
requirements are documented. A null or failure against `D` is a valid result.
