# Remote primary continuation receipt

- Workflow: `IRIS Exact Primary Physics Gate and Six-Arm Matrix`
- Run: [33469232832](https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/33469232832)
- Branch: `codex/iris-base-report-2026-08-31`
- Workflow revision: `23f7d37f50e7ad8771b55be8678c02742f8f1304`
- Overall conclusion: `failure` because the train-only physics gate failed
- Physics arms: BASE, L2/HJ, and L3/HJ+PIL each completed their 1,200-step command and independent audit successfully
- Gate receipt: `remote_physics_gate_33469232832.json`
- Forecast metrics accessed: `false`
- Downstream six-arm matrix: skipped by the gate

## Gate values

| arm | generic ratio | geometry distance | PIL distance | geometry improvement | PIL improvement | arm result |
|---|---:|---:|---:|---:|---:|---|
| BASE | 112.1441 | 2.0618 | 52.0751 | — | — | fail |
| L2/HJ | 9.4172 | 0.5934 | 0.7041 | 71.22% | 98.65% | fail: generic gate |
| L3/HJ+PIL | 10.5017 | 0.6347 | 1.1102 | 69.22% | 97.87% | fail: generic gate |

The targeted geometry and PIL distances improved substantially relative to the fresh BASE arm, but the generic fidelity requirement remained unsatisfied for both constrained arms. Therefore no TSS, AUROC, AUPRC, HSS2, Brier/BSS, or calibration result exists for this continuation.

## Remote artifact inventory

- `physics-base`: 128,708,169 bytes
- `physics-hj`: 128,499,619 bytes
- `physics-hj_pil`: 128,611,721 bytes
- `iris-primary-physics-gate`: 569 bytes

The artifacts are retained by GitHub Actions for 30 days from the run. The full downstream FITS acquisition is preserved separately by [run 33470782803](https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/33470782803) as `iris-complete-downstream-fits` (3,972,017,708 bytes; 10,486/10,486 required records validated).
