# IRIS-SEP judge strategy

## The research question

Can a forecast-time multimodal model use one daily, availability-constrained
24-hour observation window to predict a **new** >10 MeV, >=10 pfu SEP
threshold crossing in the following 24 hours with higher TSS and fewer false
alarms at matched detection than a reproduced SEPNET-O on one frozen,
episode-disjoint benchmark?

That sentence is the project. The neural architecture is a method used to test
it, not the headline.

## Why this is the right level of complexity

Recent top ISEF projects repeatedly have the same visible structure: one narrow
problem, one non-obvious method, decisive validation, and a concrete
consequence. The work can be technically sophisticated, but the public
explanation stays measurable. Examples include a cheap-GPU exoplanet detector,
a shorter antiviral synthesis with yield and cost consequences, and an MCMC
origami simulator validated against known solutions.

IRIS-SEP should therefore be drawn as only three solid expert blocks:

1. magnetic state;
2. eruption evidence;
3. pre-event particle context;

They feed one gated 24-hour probability forecast. The first benchmark is daily,
because that is the cadence supported by the comparison data; hourly operation
is future work and must not appear as a demonstrated capability. AIA pretraining
is a dashed, optional branch unless the identity bridge is exact. Secondary
heads and ablations belong in supporting material.

The safe V1 development table does not include particle-history or XRS inputs.
Accordingly, the current development run tests only the magnetic and eruption
branches. The three-stream architecture cannot be presented as trained until a
safe, frozen training-only V2/CLEAR cohort is acquired.

## What the judges must understand in three minutes

1. Solar energetic particles can damage or disrupt spacecraft electronics and
   instruments.
2. A warning system that catches storms but produces too many false alarms is
   operationally expensive.
3. IRIS combines three physically motivated observation histories, using only
   information available at the forecast time.
4. Every candidate is compared with persistence, classical models, and a
   reproduced SEPNET-O on the same locked benchmark.
5. The result is stated as storms detected, false warnings, calibration, and
   warning time—not training loss or accuracy alone.

Preferred final impact sentence, populated only from approved receipts:

> At the same storm-detection rate, IRIS produced X fewer false warnings per
> year and provided a median Y hours of warning.

## Current evidence, stated before the final benchmark

The current local V1 experiment is a pipeline-development result on a publisher
legacy target, not the final new-crossing benchmark. The five-seed neural
ensemble did **not** establish an advantage over XGBoost: its validation-monitor
TSS was 0.232 versus 0.257, and the paired 95% bootstrap interval for the TSS
difference crossed zero. These values are selection-side diagnostics and must
not be placed in the final headline table.

This negative result is a design constraint. IRIS must earn its final claim on
the frozen new-crossing cohort against the reproduced comparison model; visual
polish or added architectural complexity cannot substitute for that gate.

## Evidence hierarchy

The main board or opening slides should contain:

- the single research question;
- the three-block model diagram;
- the chronological leakage-control diagram;
- the same-cohort headline table;
- false-alarm versus detection and reliability plots;
- one event replay, one quiet replay, and one failure;
- a compact ablation table;
- an explicit limitations box.

The student must be ready to explain what they personally built, which work was
tool- or mentor-assisted, why each split exists, what a false alarm costs, why
TSS is primary, and which result would falsify the claimed advantage.

## Language guardrails

- Say **forecast-time** or **availability-constrained**, not “causal inference,”
  unless a separate causal-identification study is completed.
- Say **historical operator replay**, not operational certification.
- Keep the failed synthetic physics gate and the temporal next-magnetogram
  simulator separate from SEP forecasting evidence.
- Never call the work a breakthrough unless the complete paired gate passes.
- Report a negative or inconclusive result plainly.
- Do not say hourly, real-time, deployed, or company-ready when describing the
  demonstrated V1 experiment.

## Source basis and limitation

This strategy is based on the current official [ISEF Grand Award judging
criteria](https://www.societyforscience.org/isef/grand-award/criteria/),
[Society for Science judging guidance](https://www.societyforscience.org/isef/affiliated-fair-network/judging-at-your-fair/),
[2025 full awards](https://www.societyforscience.org/press-release/regeneron-isef-2025-full-awards/),
[2026 full awards](https://www.societyforscience.org/press-release/regeneron-isef-2026-full-awards/),
the [top-award archive](https://www.societyforscience.org/isef/awards/yancopoulos-innovator-award/),
and the [IRIS Team India archive](https://iris.exstemplar.com/isef-winners.html).
Official archives expose titles, awards, selected summaries, and rubrics—not
every scorecard or interview—so this is a broad official-record review, not a
claim to have reconstructed every judge's private decision.
