# NSRRD-V1 source-gate correction — 2026-09-12

The first NSRRD-V1 source-only run established complete daily `sci_` coverage from 2023-01-01 through 2025-04-03, stable GOES-16 L2 dimensions, explicit 13-channel energies and quality metadata, and a 13-channel energy-schema match to the current live feed within the frozen 5% tolerance. It did **not** prove which directional sensor reduction the live JSON publishes.

Independent label-free inspection found that a 2026-09-05 upright GOES-18 witness strongly tracked the westward sensor more closely than the eastward sensor. That diagnostic was not part of the first immutable gate, and the available seven-day live response contained only `yaw_flip=0`. Therefore it cannot validate the yaw-flipped half of the proposed reduction.

This correction supersedes the first run's training-permission wording. The source interface remains incomplete until immutable, source-only witnesses cover both yaw states and all 13 channels. `NOAA_SGPS_REPROCESSED_RETROSPECTIVE_DEVELOPMENT_V1` must return `NSRRD_SOURCE_REJECTED_DO_NOT_TRAIN` until that condition passes. No labels, model scores, protected outcomes, or frozen submission results were used or changed.
