# Novelty audit — internal source assessment

This is an audit of proposed claims, not a student submission bibliography. The student must locate and read the underlying work and construct their own references. Coverage is a bounded web search and inspection of available primary texts; absence from this search is not proof of priority. Publisher access restrictions mean some comparisons use author preprints or official technical material.

| Proposed novelty | Classification / confidence | Relevant primary evidence and boundary |
|---|---|---|
| Covariance identity for reweighting | Definitely known; high | [Frank, Price-equation analysis](https://onlinelibrary.wiley.com/doi/10.1111/j.1420-9101.2012.02498.x) explicitly treats changes of weighted averages. No new-theorem claim. |
| Random individual vs random cluster estimands | Definitely known; high | [Nevalainen, Datta and Oja](https://arxiv.org/abs/1803.01175), original Statistical Papers work predates its 2018 arXiv posting; informative cluster size and inverse-size weighting. |
| Generalized weighting / bounds / rank changes | Likely known across sampling and statistics; high | Algebraic consequences of reweighted expectations; no separate novelty established. |
| Event verification in solar forecasting | Definitely known; high | [Kanzelhöhe event verification](https://link.springer.com/article/10.1007/s11207-018-1312-7), Solar Physics 2018; solar flare detection is adjacent, not identical to SEP onset forecasting. |
| SEP warning and event-validation systems | Definitely known; high | [ESA ASPECS technical report](https://nebula.esa.int/sites/default/files/neb_tec_studies/2775/public/4000120480_EX.pdf); pre/post-event forecasting architecture is established. |
| Problems comparing SEP targets and scores | Definitely known; high | [Whitman et al. review](https://ntrs.nasa.gov/api/citations/20230001732/downloads/Whitman_particle_models_1-s2.0-S0273117722007244-main.pdf), model/target heterogeneity; [NOAA verification](https://www.spaceweather.gov/content/forecast-verification). |
| Repeated physical episodes in a daily SEP table | Adaptation to this dataset; moderate | [SEP-PRISM Data](https://arxiv.org/html/2607.16160v1) explicitly uses fixed nonoverlapping 24-hour windows. The audit quantifies multiplicity; do not misdescribe it as overlap leakage. |
| Fixed predictions + matched occurrence/episode/onset views + exact mechanism + paired evidence | Potentially novel combination; moderate | The specific combined empirical comparison may be a contribution. Search did not establish universal priority. Require a domain expert to challenge this comparison before any priority claim. |
| New operational model or SOTA | Unsupported | [SEPNET](https://arxiv.org/html/2512.12786v1) includes SEPVAL evaluation; [SEPNET-PRISM](https://arxiv.org/html/2606.14440v1) uses its own protocol. Different targets/cohorts cannot support a leaderboard comparison. |
| Generalized persistence decomposition as new theory | Insufficient evidence of novelty | It is an exact partition of means under explicit onset-support assumptions. Present as an explanatory derivation, not a discovery of unknown mathematics. |

The search covered SEP/onset forecasting, persistence, episode/event normalization, clustered forecast verification, informative cluster size, size-biased sampling, TSS, rolling-window evaluation, ASPECS, SEPVAL, SEPNET/PRISM and operational verification. Not every retrieved paper had full-text access, and this is not a formal systematic review.

Specific caution for SEPNET: a discrepancy in a released training/feature table does not invalidate an independently defined final SEPVAL experiment. This project should identify the exact source table and evaluation it audits. Conventional occurrence prediction is relevant to continuing radiation exposure; onset prediction is relevant to a new warning. Previous researchers did not necessarily overlook that distinction.

To challenge the application claim, a reader should look for a prior SEP study holding predictions and operating thresholds fixed while simultaneously matching the negative cohort, inverse-weighting event multiplicity, separating onset from active windows, and attributing the difference exactly. Finding such a study changes the novelty wording to replication/extension; it does not make the empirical audit false.
