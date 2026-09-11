# Custodian handoff — prospective SEP confirmation V1

**Study:** `IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1`  
**Development-side status:** outcomes sealed; no positive counts or event identities requested.

## Separation of roles

The development team supplies only the frozen preregistration, a completed execution manifest with model/rule hashes, append-only timestamped prediction records, source-readiness/feature-availability receipts, and the aggregate-only evaluator.

The independent custodian alone has access to protected target outcomes and episode identities until the result package is frozen.

## Before unsealing anything

The custodian must verify:

1. `study_id == IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1`;
2. the protected candidate period starts no earlier than `2025-09-10T00:00:00Z`;
3. the execution manifest status is `EXECUTION_FROZEN`;
4. every evaluated model/rule has a 64-hex artifact/rule hash and feature-schema hash;
5. every evaluated input is marked causal only after an issue-time availability receipt exists;
6. every prediction was created no later than its issue time;
7. there is exactly one prediction per frozen model per issue; and
8. no model, threshold, feature schema or target definition was changed using protected outcomes.

If any check fails, stop with `BLOCKED_CONTRACT_VIOLATION` and do not reveal protected labels.

## Sealed evaluator input

CSV columns expected by `tools/run_custodian_prospective_episode_evaluation_v1.py`:

- `issue_time_utc`
- `prediction_timestamp_utc`
- `model_id`
- `alert`
- `probability`
- `mapped_occurrence`
- `onset_eligible`
- `onset_label`
- `active_at_issue`
- `episode_id`
- `quiet_block_id`
- `ambiguous_positive`

The sealed CSV stays with the custodian. It must not be committed to this repository or returned to the development team.

## Execution

```bash
python iris_report/iris_sep/tools/run_custodian_prospective_episode_evaluation_v1.py \
  --custodian-mode \
  --sealed-input /secure/protected_rows.csv \
  --execution-manifest /secure/frozen_execution_manifest.json \
  --output-dir /secure/aggregate_result
```

The program is designed to emit only `aggregate_results.json` and `execution_receipt.json`. It does not write row-level predictions, labels, event identities or bootstrap row samples.

## Information-floor behavior

The custodian may compute the protected episode/block counts only inside the sealed execution. If there are fewer than 50 distinct onset episodes or 500 quiet blocks, return the aggregate disposition `INSUFFICIENT_CONFIRMATORY_INFORMATION`.

Do **not** disclose which positive events exist merely to help redesign the experiment.

## Frozen inference

The evaluator uses 10,000 paired physical-unit bootstrap draws with seed `20260911`. The same sampled onset episodes and quiet blocks are reused in each paired contrast.

The mandatory contrast is the past-proton-active proxy's `NEW_ONSET_CAUSAL - MAPPED_OCCURRENCE` TSS difference. A prospective confirmation requires both the information floor and an upper 95% paired percentile limit below zero.

Optional learned-model contrasts are evaluated only if those models were already present in the frozen execution manifest. They are reported regardless of sign.

## What may be returned

After execution, return only the aggregate files plus cryptographic hashes of the sealed input, execution manifest, aggregate result and evaluator source commit.

Do not return the sealed CSV, row-level target labels, episode IDs, positive dates or event identities until the scientific conclusion is frozen and the governance plan explicitly permits disclosure.

## Development-side interpretation

- `PROSPECTIVE_CONFIRMATION`: required floor met and required contrast supported.
- `NO_PROSPECTIVE_CONFIRMATION`: floor met but required interval does not support the frozen direction.
- `INSUFFICIENT_CONFIRMATORY_INFORMATION`: floor not met.
- `BLOCKED_CONTRACT_VIOLATION`: execution contract was not valid.

All four are legitimate outcomes. None may trigger post-hoc model rescue on the same protected cohort.
