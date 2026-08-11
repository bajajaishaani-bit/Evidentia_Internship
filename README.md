
# Health Financing as Risk Transfer: Measuring Financial Protection Efficiency Across Countries
 
**A novel efficiency framework that reframes national health financing systems as risk-transfer mechanisms, and measures how efficiently countries convert financing inputs into household financial protection against catastrophic health expenditure.**
 

 
---
 
##  30-Second Summary
 
Countries are usually judged on health financing by two disconnected questions: *how much do they spend?* and *how many people are protected from catastrophic costs?* Nobody asks the question financial economists ask about a portfolio: **how much protection are you getting per unit of financing risk you've taken on?**
 
This project builds **Financial Protection Efficiency (FPE)** : a Sharpe-ratio-style efficiency score for national health systems, using Data Envelopment Analysis (DEA), with two original methodological corrections that make cross-country comparison defensible. It then tests whether FPE responds to governance and financial-sector structure in theoretically sensible ways, using a Simar-Wilson double-bootstrap truncated regression.
 
**Headline findings:**
- Government Effectiveness and Control of Corruption move FPE in *opposite* directions, despite usually moving together across countries.
- Banking-sector depth *lowers* FPE; capital-market depth *raises* it, consistent with the bank-based vs. market-based risk-sharing literature.
- The findings are stable across every model specification tested (4 specifications, 528-537 country-year observations, 2,000 bootstrap replications).
---
 
## Why This Project Exists
 
Health financing systems are conventionally evaluated by *how much* they spend or *how broadly* they cover a population, rarely by how efficiently they convert financing into actual household protection. Two countries can report identical catastrophic health expenditure (CHE) rates while one spends far more per dollar of financing to get there. Existing metrics: CHE incidence, impoverishment rates, out-of-pocket share — cannot tell these countries apart, because they measure *protection* without ever considering the *cost* of achieving it.
 
This project builds the missing instrument.
 
---
 
## The Core Idea
 
In financial economics, a Sharpe ratio doesn't ask "what was the return?" , it asks "how much return did you get *per unit of risk*?" Two portfolios with identical returns can have very different Sharpe ratios if one took on much more risk to get there.
 
**Financial Protection Efficiency (FPE) applies the same logic to health systems.** Instead of asking "how much financial protection does this country provide?", it asks: **"how much protection does this country generate per unit of financing input, relative to its peers?"**
 
This project is the first, to our knowledge, to import risk-adjusted return logic from financial economics directly into health-financing efficiency measurement using a formal nonparametric efficiency method (DEA), rather than as a loose metaphor.
 
---
 
## Table of Contents
 
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [Stage 1 — Constructing FPE (DEA)](#stage-1--constructing-fpe-dea)
  - [The Two Methodological Corrections](#the-two-methodological-corrections)
  - [Stage 2 — What Predicts FPE? (Simar-Wilson Regression)](#stage-2--what-predicts-fpe-simar-wilson-regression)
- [Data Sources](#data-sources)
- [Results](#results)
- [Discussion Highlights](#discussion-highlights)
- [Limitations (Stated Plainly)](#limitations-stated-plainly)
- [Tech Stack](#tech-stack)
- [Reproducing This Analysis](#reproducing-this-analysis)
- [Paper Status & Where to Read It](#paper-status--where-to-read-it)
- [Citation](#citation)
- [Author](#author)
---
 
## Repository Structure
 
> Adjust this tree to match your actual folder layout before publishing — the structure below reflects the project's logical stages.
 
```
.
├── data/
│   ├── raw/                     # Untouched source pulls (WHO GHED, GHO, WGI, IMF FD, OGHIST)
│   ├── processed/                # Merged, harmonised, list-wise-deleted analysis panel
│   └── data_dictionary.md        # Variable definitions and construction notes
├── stage1_dea/
│   ├── build_frontier.py         # Income-stratified, SCI-filtered, year-by-year BCC-DEA (HiGHS LP solver)
│   ├── care_avoidance_filter.py  # UHC SCI ≥ 50 eligibility filter
│   └── income_stratification.py  # OGHIST-based peer-group assignment
├── stage2_regression/
│   ├── simar_wilson_bootstrap.py # Double-bootstrap truncated regression (B = 2000)
│   └── specifications.py         # Models 1–4 (aggregated vs. disaggregated WGI/FD)
├── diagnostics/
│   └── vif_check.py              # Multicollinearity diagnostic (WGI composite vs. FD components — open item, see Limitations)
├── manuscript/
│   ├── manuscript_draft.docx
│   └── references.bib
├── outputs/
│   ├── tables/                   # Tables 1–4 as reported in the paper
│   └── figures/
├── requirements.txt
└── README.md
```
 
---
 
## Methodology
 
### Stage 1 : Constructing FPE (DEA)
 
FPE is estimated using an **output-oriented BCC (variable returns to scale) Data Envelopment Analysis**, run separately for each year, following the nonparametric efficiency-measurement tradition of Charnes, Cooper & Rhodes (1978) and Banker, Charnes & Cooper (1984).
 
**Why VRS instead of CRS?** Health financing systems cannot plausibly operate at the same optimal scale regardless of a country's income level — a low-income system's efficiency scale is structurally different from a high-income one. Using constant returns to scale would conflate scale differences with genuine inefficiency.
 
**Why solve year-by-year instead of pooling?** So a country's efficiency score reflects its position relative to peers facing the *same global financing conditions* in that year, not against countries observed a decade earlier under a completely different cost structure.
 
**Inputs (2):**
| Variable | Why it's in the model |
|---|---|
| Current health expenditure (CHE) per capita, PPP-adjusted | Direct financing input |
| Out-of-pocket (OOP) share of CHE | Captures financing composition, not just volume |
 
**Output (1):**
| Variable | Why it's the output |
|---|---|
| 100 − CHE10 (financial protection rate) | Direct, interpretable measure of protection against catastrophic spending |
 
**Variables considered and explicitly excluded**, with reasoning:
- **GGHE-D** (government health expenditure) : collinear with OOP share (near-complementary shares of CHE)
- **GDP per capita** : CHE per capita is already an expenditure figure; adding GDP would double-count scale effects already handled by income stratification
- **UHC Service Coverage Index** : excluded from the input/output set specifically because it is *used elsewhere* as the eligibility filter; including it too would create circularity
### The Two Methodological Corrections
 
This is the paper's primary technical contribution.
 
**1. Care-avoidance eligibility filter (UHC SCI ≥ 50).**
A naive DEA frontier can be fooled: a very poor, high-OOP-share country can show *low* CHE not because its financing system works, but because households simply forgo care they can't afford. Left unfiltered, these countries anchor the efficiency frontier trivially. The SCI filter excludes countries where low measured catastrophic expenditure plausibly reflects foregone care rather than genuine financial protection — before the frontier is even constructed.
 
**2. Income-group stratification (World Bank OGHIST).**
With only 2 inputs and 1 output, a pooled, unstratified DEA frontier fails outright: countries with unusual input combinations trivially achieve "efficient" status regardless of whether they're actually well-run. Stratifying by historical income classification (High / High-Middle / Low-Middle / Low, by year) ensures countries are only benchmarked against genuine peers, restoring the frontier's discriminatory power.
 
### Stage 2 — What Predicts FPE? (Simar-Wilson Regression)
 
DEA efficiency scores are bounded in [0,1] and serially correlated by construction — standard OLS is invalid here. Stage 2 uses **Simar & Wilson's (2007) double-bootstrap truncated regression**:
 
- Maximum likelihood truncated regression (not OLS) at each bootstrap draw
- **B = 2,000** bootstrap replications
- Bias-corrected coefficients with **percentile confidence intervals**
- p-values computed as the two-sided proportion of the bootstrap distribution crossing **zero** — not tested against the bias-corrected point estimate (a bug we caught and fixed during development; testing against the point estimate rather than zero silently invalidates inference)
- Year fixed effects throughout
**Four specifications**, because the level of aggregation for governance and financial development turns out to *matter*:
 
| Model | Governance | Financial Development |
|---|---|---|
| 1 | WGI composite | FD composite |
| 2 | GE, CC, RL (disaggregated) | FD composite |
| 3 | WGI composite | 6 FD sub-components (disaggregated) |
| 4 | GE, CC, RL | 6 FD sub-components |
 
This stage is explicitly framed as a **construct-validity check**, not a causal model , the cross-sectional/short-panel design cannot support causal claims about the determinants of efficiency.
 
---
 
## Data Sources
 
All data is publicly available; no primary data collection, human subjects, or restricted-access material is involved.
 
| Source | Provides | Used In |
|---|---|---|
| WHO Global Health Expenditure Database (GHED) | CHE per capita PPP, OOP share, GGHE-D | Stage 1 inputs |
| WHO Global Health Observatory (GHO) | CHE10, UHC Service Coverage Index | Stage 1 output + eligibility filter |
| World Bank Historical Income Classifications (OGHIST) | Country-year income group | Stage 1 stratification |
| Worldwide Governance Indicators (WGI), World Bank | Government Effectiveness, Control of Corruption, Rule of Law | Stage 2 covariates |
| IMF Financial Development Index (Svirydzenka, 2016) | Banking-sector depth/access, capital-market depth (leaf-level sub-components only) | Stage 2 covariates |
 
**Samples:** Stage 1 : 528 country-year observations, 96 countries, 2000–2021. Stage 2 : 537 country-year observations, 111 countries, 2001–2020. Years/countries missing any Stage 1 variable are dropped list-wise rather than imputed, to avoid basing efficiency scores on estimated rather than observed inputs.
 
---
 
## Results
 
### Governance operates through two opposite channels
 
| Predictor | Model 2 β | Model 4 β | Direction |
|---|---|---|---|
| Government Effectiveness | −0.257 (p < .001) | −0.196 (p < .001) | **Lowers** FPE |
| Control of Corruption | 0.128 (p = .032) | 0.142 (p < .001) | **Raises** FPE |
| Rule of Law | n.s. (p = .436) | n.s. (p = .968) | No effect |
 
The aggregated WGI composite alone (Model 1: β = −0.130, p < .001) hides this entirely — it looks like "governance quality reduces efficiency," which is the opposite of what's actually happening once GE and CC are separated.
 
### Financial-sector structure operates through two opposite channels
 
| Predictor | Model 3 / Model 4 β | Direction |
|---|---|---|
| Financial Institutions Access (banking) | −0.586 / −0.519 (p < .001) | **Lowers** FPE |
| Financial Institutions Depth (banking) | −0.524 / −0.481 (p < .001) | **Lowers** FPE |
| Financial Markets Depth (capital markets) | +0.443 / +0.460 (p < .001) | **Raises** FPE |
 
### An honestly unresolved sensitivity
 
The WGI composite's coefficient is strongly negative in Model 1 (β = −0.130, p < .001) but statistically indistinguishable from zero in Model 3 (β = −0.004, p = .852) — on the *identical sample*, differing only in how financial development is aggregated. We flag this explicitly rather than picking whichever model tells a cleaner story: a formal VIF diagnostic on the Model 4 regressor set is still pending. The disaggregated GE/CC results, by contrast, are stable across every specification in which they appear, which is why the paper's central governance claims rest on those, not on the unstable composite.
 
---
 
## Discussion Highlights
 
- **Why do GE and CC diverge?** Corruption is a direct leak — diverted money never reaches patients, so less corruption mechanically raises the protection-per-dollar ratio (Shleifer & Vishny, 1993). Government Effectiveness instead proxies administrative *capacity and scale* — a more capable bureaucracy can mean a bigger, more elaborate system, which is not the same thing as a leaner, more efficient one. "Does this system work well" and "does this system waste little" are different questions.
- **Why do banking depth and capital-market depth diverge?** Drawing on Gertler & Gruber's (2002) finding that Indonesian households coping with health shocks through credit recovered less than 30% of lost income, we argue credit-based coping *postpones* cost rather than *transferring* risk. A country financing health costs mainly through household credit (bank-based) looks structurally different from one financing it through pooled public risk-sharing instruments (market-based) (Allen & Gale, 2000) — even though this is inferred from country-level correlation and the causal direction cannot be established here.
- **Rule of Law's null result is treated as informative**, not as noise: if FPE responded to *every* governance variable, that would suggest it's just picking up generic "good country" variance rather than a real financing-specific mechanism.
---
 
## Limitations (Stated Plainly)
 
- **Low-dimensionality frontier.** Even with both corrections, 2 inputs / 1 output limits the frontier's discriminatory power relative to richer DEA specifications.
- **WGI sub-dimension non-orthogonality.** GE and CC are not fully independent by design; the pending VIF diagnostic (see [Discussion Highlights](#discussion-highlights)) is an open robustness question, not a resolved one.
- **Behavioral hazard in OOP share.** Patients facing high out-of-pocket costs may forgo care that was actually needed — the SCI filter addresses the most extreme cases but not behavioral responses within the eligible sample.
- **No causal identification.** Cross-sectional/short-panel structure; governance/financial-structure ↔ FPE relationships are robust correlations, not causal effects, and reverse causality is plausible.
- **Standard cross-country health-financing measurement error** — inconsistent survey years, differing national CHE-threshold definitions, coverage gaps in lower-income countries.
- **Relative, not absolute, welfare measure.** A high FPE score means a country is efficient *relative to its peer frontier* — not that its households are safe from catastrophic expenditure in absolute terms. This is a direct analogue to a high Sharpe ratio not implying high absolute returns.
---
 
## Tech Stack
 
- **Python** : DEA and regression implementation
- **scipy (HiGHS LP solver)** : linear programming for the BCC-DEA frontier
- **pandas / numpy** : data harmonisation and panel construction
- **Custom Simar-Wilson bootstrap implementation** : double-bootstrap truncated regression, B = 2000, bias-corrected percentile CIs
---

 
**Aishaani Bajaj**
Fourth-year Economics undergraduate | Research conducted as part of an internship project
 
This project was built to demonstrate original methodological contribution (not just applied replication): two novel corrections to a standard DEA framework, a from scratch double-bootstrap inference implementation, and a construct , FPE , that imports a genuinely cross-disciplinary idea (risk-adjusted return) into a new empirical domain.
 
*Feedback, issues, and questions are welcome , please open an issue on this repository.*
