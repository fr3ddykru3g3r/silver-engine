# IRIS synthetic-augmentation locked-test protocol — frozen 2026-08-26

This file freezes the six-arm downstream experiment before any arm in this protocol is evaluated on the historical test partition.

## Generator arms

The synthetic arms use the four matched generators launched in GitHub Actions run **32964195149** after the JSOC transport-only retry patch. The scientific generator settings remain fixed:

- seed 2026;
- 800 optimizer steps;
- diffusion horizon 200;
- U-Net base channels 32;
- same training subset and sampling policy across arms;
- 250 synthetic positive examples per downstream arm (2 per each of up to 125 positive connected regions);
- arms: unconstrained `base`, Hale/Joy population constraint `hj`, PIL-gradient constraint `pil`, and combined `hj_pil`.

The transport patch changes only retry/backoff behavior for immutable FITS downloads and does not alter sample selection or data values.

## Downstream six-arm comparison

All arms use the same FlareCNN and optimization budget:

- width 48, dropout 0.20;
- focal BCE, gamma 1.5;
- AdamW, learning rate 3e-4;
- 1,200 optimizer steps, validation every 300 steps;
- seed 2026;
- real training subset: 4 temporally spread endpoints per connected region, maximum 2 positives per region;
- validation subset: 6 endpoints per connected region, maximum 2 positives per region;
- the focal positive-class weight is computed from the real-only training subset and held fixed across every arm.

Arms:

1. `real`: real data only.
2. `duplicate`: real data + 250 duplicated positive real observations, connected-region balanced.
3. `base`: real data + 250 unconstrained synthetic positives.
4. `hj`: real data + 250 Hale/Joy-constrained synthetic positives.
5. `pil`: real data + 250 PIL-gradient-constrained synthetic positives.
6. `hj_pil`: real data + 250 combined-constraint synthetic positives.

For each arm, the checkpoint and classification threshold are selected using validation TSS only. Test data are not used in this selection.

## Frozen test protocol

- Test subset: 6 temporally spread endpoints per connected region, maximum 2 positives per region, seed 2028.
- Each arm is evaluated on exactly the same sample IDs.
- Report TSS, HSS, recall, FPR, precision, AUROC, AUPRC, Brier score, Brier skill score and ECE10.
- Use 5,000 connected-region cluster-bootstrap replicates per arm.
- Prespecified paired connected-region comparisons: every augmented arm minus `real`, and every synthetic arm minus `duplicate`.
- Positive deltas favor the first arm for TSS/HSS/recall/precision/AUROC/AUPRC/BSS; lower is better for FPR/Brier/ECE, so their signs must be interpreted accordingly.

## One-shot rule

After this workflow exposes the test metrics, these six configurations are frozen. Any later tuning is a new experiment and cannot replace the results of this protocol.
