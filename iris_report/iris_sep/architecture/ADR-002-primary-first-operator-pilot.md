# ADR-002: Primary-first research pilot for satellite operators

Status: accepted on 2026-09-04; locked-test outcomes not accessed.

## Decision

The first validated IRIS-SEP product has one modeled scientific output:

> Calibrated probability of a new >10 MeV, >=10 pfu SEP threshold crossing
> within 24 hours.

The public experiment and award claim are centered on this output. The >100 MeV
event, peak-flux quantiles, and onset-time outputs remain schema-compatible but
are disabled and reported as unavailable until each has complete labels,
appropriate censoring or conditional-loss semantics, validation receipts, and
an ablation showing that it does not weaken the primary forecast.

The three forecast-time experts remain magnetic state, eruption evidence, and
pre-event particle context. AIA is excluded from the first validated model
because no authoritative AARP-to-HMI crosswalk exists.

## Product boundary

The first company-facing release is an auditable **research pilot**, not an
operationally certified warning service. It recommends no spacecraft command
and never controls a spacecraft. An operator remains responsible for action.

Every forecast must expose:

- issue time and 24-hour horizon;
- calibrated primary probability and all-clear probability;
- calibration and operating-policy identifiers;
- model version and evidence-receipt hash;
- per-source observation time, publication time, and freshness;
- missing modalities and ensemble uncertainty;
- `VALID`, `DEGRADED`, or `ABSTAIN` forecast status;
- an advisory `NORMAL`, `MONITOR`, `PREPARE`, or `PROTECT` state only when a
  validation-frozen policy authorizes it.

If the critical-input, freshness, schema, model-version, or evidence checks
fail, the system abstains rather than silently emitting a normal forecast.

## Scientific boundary

Architecture readiness is not benchmark success. No award, superiority, false
alarm reduction, or operational-performance claim is permitted until the
reproduced SEPNET-O and IRIS predictions use identical frozen identities and
the entire paired benchmark gate passes.

## External validation required before operational use

- independent historical replay by an operator;
- prospective shadow-mode evaluation across a predeclared period;
- data-provider latency and outage characterization;
- security, availability, incident-response, and change-control review;
- human-factors review of advisory states and false-alarm costs;
- explicit licensing and redistribution approval for every input and model;
- organization-specific acceptance criteria and sign-off.

These requirements cannot be satisfied by a science-fair benchmark alone.
