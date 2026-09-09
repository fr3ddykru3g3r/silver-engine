# IRIS-SEP freshness-crossover source readiness — 2026-09-09

## Disposition

`READY_TO_BEGIN_RETROSPECTIVE_CONTROLLED_DELAY_DEVELOPMENT_STUDY`

This readiness decision applies only to `IRIS_SEP_FRESHNESS_CROSSOVER_STUDY_V1`. It does not unblock `NOAA_CAUSAL_REDUCED_INPUT_V1`, does not change the frozen V3 interface, and does not create independent final evidence.

## What is demonstrated

1. NASA OMNIWeb explicitly exposes five-minute integral proton fluxes above 10, 30 and 60 MeV for GOES-13 over 2011 through late 2017. The GOES-13 page exposes eastward, westward and `Av` quantities. OMNI documentation states that the GOES-13 value included in five-minute OMNI is the average of the NOAA eastward- and westward-looking fluxes.
2. The >10 MeV OMNI record-format documentation gives units of `1/(cm^2 s sr)` and uses fill values for missing records. The generic GOES-13 browser heading that contains `/MeV/nuc` is therefore not used as the unit authority for the integral >10 MeV quantity.
3. NOAA/NCEI documents GOES-15 XRS availability beginning in 2010-09 and provides operational one-minute XRS averages for GOES 3-15. The operational XRS archive was processed for operational use when collected.
4. NOAA's operational XRS readme documents the archive scaling corrections used here: divide archived XRS-A by `0.85` and archived XRS-B by `0.7` to recover true flux. It also documents that averaged-data timestamps are approximately 1-3 seconds after the start of the averaging interval.
5. NOAA/NCEI distinguishes operational GOES 1-15 XRS/particle archives from later science-quality reprocessing. This study freezes the operational XRS route and NASA OMNI GOES-13 integral-proton route and will not silently substitute science-quality XRS values.
6. NCEI documents that averaged operational particle products contain corrected flux and that filenames containing `cpflux` contain integrated proton data. This independently supports the integral-proton semantics underlying the legacy archive, while the frozen study uses NASA OMNI's documented GOES-13 E/W average for its primary proton series.

## Protected-data / independence reconciliation

The 2011-2017 interval overlaps earlier IRIS-SEP development planning/inspection. It therefore cannot be relabelled as a new untouched final benchmark merely because the feature set or research question changed.

Consequences:

- the study is explicitly retrospective development/mechanism evidence;
- the 2017 role is named `retrospective_score`, not locked or independent test;
- no award, operational-certification or population-wide safety claim may be based on this score role;
- an independent confirmation claim requires a genuinely uninspected cohort under a separately frozen contract.

## Frozen study choices

Authoritative contract: `config/freshness_crossover_study_v1_preregistration_2026-09-09.json`.

Primary proton input:

- NASA SPDF/OMNIWeb GOES-13 five-minute `Prot flux_Av (>10 MeV)`;
- `Av` is used because NASA explicitly documents how it is constructed from the E/W NOAA measurements;
- E and W are sensitivity diagnostics only and cannot replace the primary series after outcomes are inspected;
- this is not claimed to be identical to an SWPC primary operational stream.

Primary XRS input:

- NOAA/NCEI GOES-15 operational one-minute XRS averages;
- XRS-A true flux = archived A / 0.85;
- XRS-B true flux = archived B / 0.7;
- only information timestamped at or before issue time is eligible;
- fill/nonfinite data and nonzero operational quality flags are unavailable observations;
- science-quality XRS substitution is forbidden within this study.

The two modalities need not come from the same spacecraft because the experiment asks about information freshness, not same-platform sensor fusion. This is a controlled retrospective information-age experiment, not an operational constellation replay.

## Delay experiment boundary

The delay grid is frozen to `0, 5, 15, 30, 60, 120, 360, 720, 1440` minutes. Delay/outage manipulation occurs on timestamped source records before feature aggregation. Post-hoc masking of already-computed daily feature vectors is not equivalent and is forbidden for the primary experiment.

All methods are evaluated on the same forecast opportunities. Event-positive abstentions are counted as unwarned events for end-to-end accounting and also reported separately.

## Remaining checks before training code consumes values

The acquisition implementation must still emit a machine-readable manifest containing:

- source URL/path for every downloaded file;
- byte size and SHA-256;
- observed coverage interval;
- parsed timestamp convention;
- fill/quality-variable mapping;
- missing intervals;
- source version or archive identity when present.

If the actual file schema contradicts the frozen semantics above, training stops and the contract is amended only as a separately versioned study before outcome inspection.

## Primary references audited

- NASA SPDF OMNIWeb GOES-13/14 five-minute integral-proton browser and OMNI data documentation.
- NOAA/NCEI GOES 1-15 Space Weather Instruments documentation.
- NOAA/NCEI `GOES X-ray Sensor (XRS) Operational Data`, version 1.5, 17 June 2022.
- NOAA/NCEI GOES-SEM operational archive metadata.

## Claim boundary

Source metadata readiness establishes that a retrospective controlled-delay experiment can be constructed without inventing a >10 MeV quantity or silently mixing XRS processing regimes. It does not establish useful predictive signal, a freshness crossover, superiority of a switching rule, or independent final forecast skill.
