# Luna A data/provenance receipt

Retrieved 2026-09-04 (UTC). Scope was limited to authoritative public metadata, protocol documentation, and small pinned source artifacts. The locked SEPVAL/CLEAR rows and evaluation outcomes were not opened or copied.

## Deliverables

- `provenance_manifest.json` is the machine-readable manifest draft. Its SHA-256 is `e6db14d2f939f2241dc24ed45e7a495b0bdd922334862a0ebeeb3b2081c1b9d4`.
- `zenodo_21297635_record.json` pins SEP-PRISM Zenodo v1 (`10.5281/zenodo.21297635`), including the only exposed archive (`SEP-PRISM.zip`, 4,881,194,881 bytes, MD5 `772fcfb2c0a92c487924a4d48fc5021f`). The archive was not downloaded.
- `CLEAR_Benchmark_Dataset_V2_0_Documentation.pdf` is the NASA CCMC CLEAR v2.0 documentation, 219,455 bytes, SHA-256 `f40da3a4b2892c5ea464b305557be6d82f1843751cb0567fa5f3d30adda3f055`.
- `zenodo_15555244_sepval_record.json` pins SEPVAL `SEPVAL2023v2` (`10.5281/zenodo.15555244`).
- `SEPVAL2023_RulesofParticipation_v4.pdf` is the public protocol artifact, 2,352,857 bytes, MD5 `27b5ee47f426247099add06ff831756d` (matches Zenodo metadata), SHA-256 `72454fbfba16fa87abdf2b911e79c9887ab5aca9276a81e045a8751521406515`.
- `sources/fetchsep_v2/` contains small files pinned to FetchSEP tag `CLEAR_Benchmark_v2.0`, commit `db4ffe391941a4fefc4aa8b272c456397ecf34b0`.
- `sources/sepnets_v1/` contains the small SEPNET model/preprocessing/README files pinned to tag `v1.0`, commit `f9cff73adfa41c4fbffc73a8693c529d39e80995`.
- `sepnets_prism_v2_lfs_pointers.txt` records a filtered-tree acquisition lead: the public SEPNET-PRISM v2 commit `e138dcd72c1952a00e11e1a0b025337f9e7c93fb` exposes an LFS pointer for the 34,861,986-byte model-ready table (`sha256:4691cedd...`). No LFS data blob was checked out.

## Findings that affect the benchmark freeze

1. SEP-PRISM v1 reports `Analyzed Data/rolling_combinded_seq_24hours.csv` as 14,464 rows x 274 columns, with 650 operational-positive windows (`Future_OSEP_label`) and 2,122 general-positive windows (`Future_GSEP_label`). It is nested in a 4.9 GB archive, so the table itself still needs an approved acquisition path and an exact file hash. The 650 value is a rolling-window count, not a distinct-event count.

2. CLEAR v2 distinguishes operational threshold events from above-background events. The documentation reports 263 periods at `>10 MeV, 10 pfu`, 88 at `>100 MeV, 1 pfu`, and 532 `>10 MeV above background` periods. The operational list is NOAA GOES integral flux as-is with background-identified values set to zero. Onset peak is explicitly experimental/automatic, and quality flags include interrupted periods, gaps, timing affected by gaps, already-enhanced starts, and likely spikes.

3. SEPVAL source prose disagrees on cardinality. The Zenodo description and SEPNET paper state 33 SEP / 30 quiet periods; the v4 rules PDF states 30 SEP / 33 quiet periods. This must be resolved from the frozen raw files after the primary agent authorizes label access. No count was inferred here.

4. SEPVAL rules require causal inputs only, regular-cadence forecasts starting 24 hours before flare peak, and issue-time accounting from the latest input plus processing latency. Forecast JSON should include input identities/timestamps, triggers, and a training-inclusion Boolean. These rules should be reflected in the IRIS contract.

5. FetchSEP v2.0 is MIT-licensed and its `opsep` defaults are `>10 MeV, 10 pfu` and `>100 MeV, 1 pfu`; it emits event summaries and CCMC SEP Scoreboard JSON. The JSON template is preserved for schema reference only; it is not a label source.

6. SEPNET v1.0 has no formal LICENSE file/declaration in the pinned repository root. Public code/data availability in the paper is not treated as a redistribution license. The public `rolling_combinded_testing.csv` and SEPVAL event files were deliberately held back because they contain locked identities/outcomes.

## Lock and rights status

- No SEPVAL event/non-event rows, CLEAR core-list rows, SEPNET testing rows, predictions, or metrics were read.
- SEP-PRISM is explicitly CC BY 4.0 in Zenodo. SEPVAL is explicitly CC BY 4.0 in Zenodo. FetchSEP is explicitly MIT. CLEAR and SEPNET redistribution terms were not explicit in the retrieved sources; preserve attribution and verify before public packaging.
- No JSOC credential is present in this workstream. No JSOC acquisition was performed.

## Primary source URLs

- [SEP-PRISM Zenodo record](https://doi.org/10.5281/zenodo.21297635)
- [NASA CCMC CLEAR benchmark page](https://ccmc.gsfc.nasa.gov/swxcoe/clear/benchmark.php)
- [SEPVAL Zenodo record](https://doi.org/10.5281/zenodo.15555244)
- [SEPVAL workshop rules/challenge](https://ccmc.gsfc.nasa.gov/community-workshops/ccmc-sepval-2023/)
- [FetchSEP CLEAR v2.0 release](https://github.com/ktindiana/fetchsep/releases/tag/CLEAR_Benchmark_v2.0)
- [SEPNET v1.0 release](https://github.com/yuyian/SEP-Prediction/releases/tag/v1.0)
