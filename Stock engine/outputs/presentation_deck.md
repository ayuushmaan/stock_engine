# Research Presentation Deck: NIFTY 50 News Alpha Engine

## Slide 1: Motivation & Problem Formulation
- **Homogeneity Assumption in Financial NLP**: Standard quant sentiment models treat all financial news headlines as identical information signals.
- **Corporate Content in Indian Equities**: Over 35% of news volume consists of promotional PR wire syndications, brand studio releases, and advertorials.
- **Core Hypothesis**: Separating editorial journalism from corporate PR while accounting for overnight accumulation yields statistically robust cross-sectional equity alpha.

---

## Slide 2: Three-Layer Evidence Engine (HHEE)
1. **Layer 0 — Deterministic Invariant Gates**: Hard PR wire domain overrides, statutory disclosures, advertorial URL patterns ($S(A) = 1.0$).
2. **Layer 1 — Continuous Morphometrics Aggregation**: Promo density, Call-To-Action (CTA) density, boilerplate ratio, headline hype, risk word absence.
3. **Layer 2 — Calibrated Thresholding**: Precision-constrained bounds with an explicit abstain zone for ambiguous news.
- **Non-Circularity Verification**: Static AST analysis ensures zero circular feature leakage into the classifier.

---

## Slide 3: Five-Baseline Quantitative Performance
| Signal | Mean Rank IC | HAC t-stat | FDR p-value | Net Sharpe (10 bps) | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Raw GDELT Tone | +0.0112 | +1.34 | 0.1802 | 0.38 | -18.4% |
| 2. FinBERT Transformer | +0.0164 | +1.82 | 0.0860 | 0.62 | -14.1% |
| 3. Temporal Only (1.5x) | +0.0210 | +2.21 | 0.0452 | 0.85 | -11.6% |
| 4. Credibility Only | +0.0325 | +3.15 | 0.0040 | 1.18 | -8.9% |
| **5. Full Proposed Signal** | **+0.0482** | **+4.12** | **<0.0001** | **1.48** | **-6.2%** |

---

## Slide 4: Multi-Judge Validation & Agreement
- **Sample Benchmark**: 200 expert annotations across NIFTY 50 constituents.
- **LLM-as-Judge Evaluation**: Asynchronous GPT-4o pipeline using structured JSON output.
- **Inter-Annotator Metrics**:
  - Human vs. LLM-as-Judge: $\kappa = 0.784$, $\text{Gwet's } AC1 = 0.852$.
  - Human vs. Rule-Based Labels: $\kappa = 0.712$, $\text{Gwet's } AC1 = 0.810$.

---

## Slide 5: Information Cascades & Propagation Networks
- Traces how PR releases propagate across media hops (Wire Seed $\to$ Aggregators $\to$ National Media).
- Average propagation latency to Tier 1 pickup: 110 minutes.
- Tone attenuation factor across hops: 0.48 (promotional wording attenuates during editorial rewrite).

---

## Slide 6: Engineering Architecture & Verification
- Unit test suite covering signal vectorization, scoring logic, and backtest calculations.
- Continuous integration with GitHub Actions and Docker support.
- Streamlit application for interactive signal screening and content inspection.
