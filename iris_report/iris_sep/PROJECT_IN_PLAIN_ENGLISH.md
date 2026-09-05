# IRIS-SEP — project in plain English

## The project in one sentence

**When some solar measurements temporarily go missing, can a simple physics-based reconstruction keep a 24-hour solar-radiation-storm forecast useful better than ordinary fill-in methods?**

That is the project.

The forecast target stays the same: the probability that a **new** >10 MeV solar energetic particle event will cross **10 pfu within the next 24 hours**.

## Why this matters

Solar-radiation-storm forecasts use measurements of the Sun and near-Earth particle conditions. Real data feeds are not perfect: measurements can be missing, delayed or temporarily unavailable.

A forecasting system then has three choices:

1. guess the missing value with a simple statistical rule;
2. use a physics model to estimate what the missing solar state probably looked like;
3. admit that the data are too incomplete and refuse to give a normal forecast.

IRIS-SEP asks which choice is actually safest and most useful.

## The main experiment

We start with periods where the real measurement is known.

Then we deliberately hide part of it.

We ask several methods to fill the gap:

- **No fill:** keep the value missing and let the forecast model know it is missing.
- **Simple fill:** use only earlier data or values learned from the training set.
- **Physics fill:** use a simple physical model of how the relevant solar quantity changes through time.

Because we hid data that we actually know, we can compare every reconstruction with the real answer.

Then we run the same SEP forecaster with each version and ask the question that matters most:

> **Which method preserves the radiation-storm forecast best?**

A physics reconstruction only survives if it performs better than the simpler methods on the same hidden-data cases.

## What we measure

For the missing solar measurement itself, we measure how close each reconstruction is to the real hidden value.

For the final SEP forecast, we measure whether it still detects events without creating too many false alarms, whether its probabilities remain well calibrated, and how often the system has to abstain because the data are not trustworthy enough.

The important result is not "the reconstructed solar map looks realistic." The important result is whether reconstruction preserves **forecast usefulness**.

## A crucial distinction

Not every blank value is something that should be reconstructed.

### Temporary gap

The instrument normally measures the quantity, but one section is missing or delayed.

This is eligible for the reconstruction experiment.

### Measurement never existed in that era

An older instrument simply did not measure the same quantity.

This is **not** treated as a temporary gap. We do not generate a fake historical measurement and call it observed data.

If a comparable real measurement from another instrument exists, we test that real source first.

## Where physics fits

Physics is deliberately kept simple.

We do **not** begin by building a giant simulation of the whole Sun.

We first test the simplest physical model that can describe the missing quantity. Only if that clearly improves the forecast do we consider a more advanced physical simulation.

So the order is:

**real observations -> simple missing-data methods -> simple physics -> only then more complex physics if needed.**

## What would count as a strong result

A strong result would be something like:

> When a solar data feed has a temporary gap, ordinary fill-in methods lose forecast skill quickly, while a physics-based reconstruction preserves more of the original forecasting ability. When the gap becomes too large, the system's uncertainty rises and it automatically stops pretending the forecast is reliable.

That would show both a scientific result and a practical reliability improvement.

A negative result is also scientifically useful:

> If simple statistical filling works just as well as physics, then the more complicated physics model is unnecessary.

The project is designed to report that result too.

## What we are NOT claiming

We are not claiming that we perfectly simulate the Sun, that every missing solar measurement can be recovered, that the system is operationally certified, or that it will win a competition.

Those claims would require experimental evidence that does not exist yet.

## Thirty-second explanation for a judge

"We are building a 24-hour solar-radiation-storm forecaster. One problem is that real solar measurements can temporarily disappear. Instead of automatically guessing the missing data, we take measurements we already know, deliberately hide them, and compare three choices: leave them missing, fill them statistically, or reconstruct them with simple solar physics. We then test which option best preserves the actual storm forecast. If the reconstruction becomes unreliable, the system is designed to say so rather than give a confident-looking forecast from bad data."

## The research question

**Can a causal physics-based reconstruction of temporarily missing solar observations preserve 24-hour NEW-SEP forecasting skill better than simpler missing-data methods, while uncertainty tells the system when to abstain?**

## The rule that keeps the project simple

Every added method must answer one question:

> **Does this improve the forecast on a controlled experiment?**

If the answer is no, we remove it from the final model.
