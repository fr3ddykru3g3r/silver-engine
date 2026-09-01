# IRIS continuation reconciliation — 2026-09-01

## Decision applied

The BASE report is a valid BASE-stage record, not a final project result. It
does not establish forecasting improvement, causal magnetic structure, or a
breakthrough. The downloaded GitHub morphology artifact is exploratory and is
not the locked primary experiment.

## Independent checks

- The BASE acquisition receipt records 5,273 valid FITS files.
- The BASE model reached step 1,200 after a resume from step 400 without the
  original optimizer/RNG state.
- The corrected generic distance ratio is 7.681 against a ceiling of 8.0.
- Connected-region bootstrap joint-pass fraction is 0.508.
- The real-versus-generated proxy classifier AUC is 0.9998.
- The 100-step HJ/PIL screens failed the generic gate.
- No forecaster or TSS result is present in the BASE PDF.
- The downloaded GitHub artifact from Actions run `33242831692` contains the
  exploratory arms `base`, `pil`, `pil_blur`, `geometry_flip`, and
  `block_shuffle`; it does not contain the required `Rw`, `L2`, or `L3` primary
  arms. Its PIL-vs-duplicate TSS interval crosses zero.

The three downloaded evidence files are preserved under
`artifacts/exploratory_github_artifact/`. Their SHA-256 values are recorded in
`artifacts/exploratory_github_artifact/SHA256SUMS.txt`.

## Exact primary matrix recovered from the full source bundle

| Science arm | Implementation | Addition/weighting |
|---|---|---|
| `R` | `real` | real-only, unweighted |
| `Rw` | `real_weighted` | real-only, balanced positive weight |
| `D` | `duplicate` | duplicated real positives, unweighted |
| `L0` | `synthetic` + `base` | matched BASE synthetic positives |
| `L2` | `synthetic` + `hj` | matched Hale/Joy synthetic positives |
| `L3` | `synthetic` + `hj_pil` | matched Hale/Joy+strong-PIL synthetic positives |

The unweighted arms use `pos_weight = 1.0`. `Rw` uses the real-only training
ratio `N_real_negative / N_real_positive`. `pil`-only, `pil_blur`,
`geometry_flip`, and `block_shuffle` are auxiliary mechanistic controls and
cannot replace `L3`.

## Gate status

The full FITS/evidence/checkpoint inputs are locally present and can be
validated without reacquisition. The long confirmatory physics gate and the
locked downstream matrix still require an off-laptop execution environment.
No new generator, CNN, forecasting, or test result is fabricated in this
reconciliation note.
