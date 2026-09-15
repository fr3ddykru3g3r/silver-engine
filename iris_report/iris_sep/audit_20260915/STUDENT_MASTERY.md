# Mastery syllabus — reasoning, not memorization

Each session ends with an unaided explanation and a worked example. A mentor should change the numbers or assumptions to test understanding. Record what the student actually did; do not treat reading these notes as independent discovery.

| Session | Concepts | Demonstration required |
|---|---|---|
| 1. Physical target | SEP protons accelerated in solar/heliospheric processes; integral >10 MeV channel; energy versus flux | Explain why 10 MeV is particle energy and 10 pfu is flux, not ten particles or ten MeV total radiation. One pfu is one particle per square centimetre per second per steradian. Explain integral vs differential channels. |
| 2. Time and event labels | Past feature interval, issue, future horizon, onset, active episode, end, ambiguity | Draw three daily forecasts around one multi-day episode; classify onset and active positives without seeing later information at issue time. Separate catalog semantics from raw physical truth. |
| 3. Confusion matrices | TP, FN, FP, TN; POD, FPR, FAR, TSS, HSS | For TP1 FN0 FP50 TN159 compute POD1, FPR50/209, FAR50/51, TSS1−50/209. Explain why good FPR can coexist with poor FAR. |
| 4. Probabilities | Brier, calibration, discrimination, thresholds | Compute squared errors for p=.8 on y1 and y0; explain reliability as observed frequency within a probability group. Describe why changing a threshold does not recalibrate probabilities. |
| 5. Development design | Fit/calibration/threshold/score, temporal exposure, selection | Draw both benchmark and Phase-II roles. Explain why a chronological split is not necessarily an untouched independent test. Explain why one score event cannot validate a tuned controller. |
| 6. Clusters | Episode multiplicity, pseudoreplication, informative size | Create events with n=(1,3), r=(0,1). Row POD=.75; episode POD=.5. Explain what each sampling experiment asks. |
| 7. Proof | Finite-population covariance and weighting | Derive gap=.25 for the preceding example: E[n]=2, E[r]=.5, E[nr]=1.5, Cov=.5. Then reverse r and explain the sign. Prove equality iff covariance zero. |
| 8. Persistence | One-onset assumption, episode-average vs any alert | Construct an episode with missed onset but later alerts. Explain why any-alert success can be high with poor onset prediction. Derive the second decomposition term. |
| 9. Uncertainty | Paired episode/quiet-block bootstrap, temporal clusters | Resample whole toy events using identical indices for two models. Explain why independent draws waste pairing and row resampling breaks dependence. Explain why bootstrap sign fraction is not a null p-value. |
| 10. Research defense | Novelty, causal source, uncertainty, falsification, ownership | Reproduce one table from safe archived inputs; explain three limits; distinguish own work from assistance; identify a result that would weaken the application claim. |

## Formula card to derive before using

POD=TP/(TP+FN); FPR=FP/(FP+TN); FAR=FP/(TP+FP); TSS=POD−FPR.
HSS=2(TP*TN−FP*FN)/[(TP+FN)(FN+TN)+(TP+FP)(FP+TN)].
Brier=mean((p−y)^2). Weighted versions replace counts/means with explicitly defined masses. Undefined denominators must be acknowledged.

Calibration concerns predicted probabilities matching event frequencies. TSS concerns one threshold. AUROC ranks positive/negative scores. None alone establishes warning utility. Utility also depends on alert duration, lead time, missed-event cost and false-alert cost, which this audit does not measure operationally.

## Three exercises with reasoning checks

1. Equal multiplicities: choose any r values with all n=2. The gap must be zero. This falsifies the claim that clustering always inflates the score.
2. No covariance but dependence: n=(1,2,3), r=(1,0,1). Compute Cov0. Explain why zero correlation is weaker than independence.
3. Negative cohort changes: keep sensitivity fixed and increase FPR by .1 in the episode view. TSS difference changes by .1 even with zero multiplicity covariance. State the missing assumption before using the theorem.

## Pass criteria

The student can do the toy calculations, identify a wrong denominator, explain the one-event limitation without prompting, locate a result hash, reproduce the safe guard refusal, distinguish ddof0 from sample covariance, and decline an unsupported novelty/operational claim. A fluent rehearsed speech without these demonstrations is not mastery.
