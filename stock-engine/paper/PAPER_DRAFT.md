# Cheap Talk on the NSE: Source Credibility, Market Timing, and the Limits of News-Driven Return Prediction for NIFTY 50

**Status:** working draft, Oct 2026. All numbers trace to committed
`outputs/tables/*.json` in this repo. Nothing in this draft is simulated.

---

## Abstract

We test whether financial-news source credibility (organic vs. sponsored)
and market timing (closed- vs. open-window, IST) predict NIFTY 50 returns,
using 1,397,568 GDELT 2.0 records (2020-01 to 2026-03) aligned point-in-time
to daily NSE OHLCV for 50 constituents (61,740 stock-days). A three-layer
Hierarchical Hybrid Evidence Engine (HHEE) separates PR-wire/brand-studio
content from editorial news under a static-AST circularity guard and a
frozen-LLM-judge protocol. On 200 hand labels, metadata-only classification
reaches AUC 0.705; text-only features learned from 149 scraped bodies reach
0.688 (an unregularized 0.778 is rejected as rare-feature overfit); adding
deterministic PR-wire domain gates reaches 0.958 — proving origin dominates
content. An independent Qwen3-32B judge agrees with humans at kappa 0.75
(F1 0.81). Predictively, the result is null by two independent routes:
daily GDELT-tone rank IC ≈ 0.005 (t 1.2, 1532 days) and transformer-sentiment
article-level Spearman 0.0035 (n = 925k), with a full-sample daily FinBERT IC
of +0.008 (t 1.9, ns) vs next-day close-to-close and +0.0005 vs next
overnight. A Q5–Q1 factor backtest earns +11.2% gross (Sharpe 0.8) but −22.4%
net of 10 bps on 66.7% daily turnover. The only significant return effect is
short-horizon reversal (5-day momentum IC −0.012, t −2.3); buy-and-hold
returned +22.9%/yr over the sample. We conclude GDELT-tone news carries no
actionable alpha for NIFTY 50 without entity-level sentiment and turnover
discipline, and release the audit pipeline as a negative-result benchmark.

---

## 1. Introduction

Financial NLP pipelines routinely treat news as a homogeneous sentiment
stream. In India's equity press this is false on its face: a large share of
company news originates as corporate PR (wire syndications, brand-studio
native ads) engineered to sound positive. Separately, news arriving while the
exchange is shut cannot be priced continuously, creating an overnight Batch
auction at the open. We test the interaction: **organic news in closed
windows should predict most; sponsored news should predict nothing.**

## 2. Hypotheses

- **H1 (source credibility):** organic editorial news carries return-relevant
  signal; sponsored/PR content is noise or cheap talk.
- **H2 (temporal asymmetry):** closed-window news (15:30–09:15 IST) predicts
  next-session returns more than open-window news.
- **H1×H2:** organic-closed is the highest-alpha cell.

## 3. Data

- **News:** GDELT 2.0 GKG via BigQuery, 1,397,568 rows referencing NIFTY 50
  entities, 2020-01–2026-03; fields URL, domain, UTC timestamp, V2Tone.
  1,348,578 rows (96.5%) yield URL-slug pseudo-titles; 149/200 gold-set
  bodies scraped with polite caching (`pipeline/11_scrape_bodies.py`).
- **Prices:** NSE daily OHLCV via yfinance, 50 constituents + index;
  overnight/intraday/close-to-close splits; business-day calendar.
- **Labels:** 200 stratified hand labels (34 sponsored / 166 organic);
  weak rules label easy extremes only (33/33 exact on fired subset).

## 4. Methods

- **IST bucketing** (`CLOSED_PRE`/`OPEN`/`CLOSED_POST`) with T+1 mapping and
  weekend roll-forward (`pipeline/03_label_news.py`).
- **HHEE classifier:** Layer-0 domain/disclosure hard gates; Layer-1
  logistic morphometrics (promo/CTA/boilerplate/hype/risk/quote); Layer-2
  0.30/0.70 thresholds with abstain zone. Weights learned (L1 logistic,
  C=2) on real bodies, v1 hand weights retained for provenance.
- **Circularity guard:** AST audit proves the honest feature set excludes all
  weak-label inputs; honest model trains only on
  `[tone_30d_variance, cross_seed_count, mention_burst_score, word_count,
  polarity]` (`pipeline/check_circularity.py`, CI-gated).
- **Frozen judge:** Qwen3-32B fp16, fixed rubric SHA-256, greedy decode;
  rubric drift = hard failure; audits appended to `judge_audits.json`
  against a 0.70 kappa gate (`pipeline/13_gpu_judge.py`,
  `pipeline/14_audit_judge.py`).
- **Evaluation:** daily Spearman rank IC with Newey-West HAC t-stats,
  Benjamini-Hochberg FDR, Q5–Q1 spreads, 10 bps friction backtests, and a
  point-in-time paper-trading diary (`pipeline/15_paper_trade.py`, fake INR).
  Research modules fail loudly on missing data — no synthetic fallbacks.

## 5. Results

### 5.1 Classification (content is separable)

| Judge / feature set | Metric | Value |
|---|---|---|
| Metadata-only honest (5 feats) | CV AUC | 0.705 (n=200) |
| HHEE-v2 text-only, stable (C=2) | CV AUC | 0.688 (n=149) |
| HHEE-v2 text-only, unregularized | CV AUC | 0.778 — **rejected** (CTA coef 132.6, rare-feature overfit) |
| Full + Layer-0 domain gates | CV AUC | 0.958 (circularity-flagged: pr_wire 82.8% vs 0%) |
| DistilRoBERTa-financial sentiment | Sponsored-detection AUC | 0.614 (means +0.47 vs +0.22) |
| Qwen3-32B vs human | kappa / F1 | **0.75 / 0.81** (n=129 excl. UNCLEAR) |
| HHEE-v2 OOF vs human | kappa | 0.11 (precision 1.0, recall 0.07 — threshold too strict) |
| Local NLI judge vs human | kappa | −0.03 (over-calls sponsored; FAILs audit gate) |

### 5.2 Return prediction (alpha is null)

| Signal → forward | IC | t | n |
|---|---|---|---|
| net_signal → fwd1 | +0.005 | 1.2 | 1532 days |
| organic → fwd1 | +0.008 | 1.9 | 1532 days |
| FinBERT article → fwd1 | +0.0035 | — | 925k articles |
| FinBERT daily → fwd1 | +0.008 | 1.9 | 1528 days |
| FinBERT daily → next overnight | +0.0005 | 0.1 | 1528 days |
| 5-baseline article panel (17k/24d) | +0.010…+0.017 | ≤1.3 | FDR 0.43, all ns |
| mom_5d → fwd1 (reversal) | −0.012 | −2.3 | significant negative |
| Buy-and-hold (hurdle) | +22.9%/yr | — | — |

Q5–Q1 predictive backtest: **+11.2% gross (Sharpe 0.8) → −22.4% net**
(Sharpe −1.6, DD −79.6%) on 66.7% daily turnover. Sign hit-rate 49.1%.

### 5.3 Leakage diagnostics (why naive runs look good)

Same-day (contemporaneous) correlations masquerade as alpha:
`sponsored_open → same-day intraday` IC +0.045 (t 9.6) collapses to +0.004
next-day. All reported results above use strict forwards only.

### 5.4 Media structure (measured, not simulated)

47,814 multi-domain ticker-days (median 7 domains, tone range 6.7,
PR-wire share 7.3%) from the full panel — cascade footprints are real even
though their trade value is nil.

## 6. Limitations

URL-slug pseudo-titles (not headlines) for the 1.35M FinBERT panel;
scored-articles subset bounds the 5-baseline window to Jan–Feb 2020;
current-constituent survivorship; business-day (not NSE-holiday) calendar;
10 bps.squeeze excludes borrow costs; single-market (NIFTY only).

## 7. Conclusion

Indian equity PR is *detectable* (kappa 0.75) but not *monetizable* with
tone-level signals: the market leaves no next-day alpha in it. The
contribution is a fully reproducible negative result with leakage
diagnostics, a frozen-judge protocol, and a paper-trading diary that keeps
future claims honest. The open direction is entity-level sentiment with
turnover-constrained execution — the diary (`paper_scoreboard.json`) is
built to score exactly that.

## Reproducibility

```
pip install -r requirements.txt && pip install -e .
pytest tests/                                   # 22/22
python pipeline/check_circularity.py            # BUILD PASS
python pipeline/11_scrape_bodies.py             # bodies (cached)
python research/finbert_daily.py                # FinBERT panel (sharded, resumable)
python research/06_baselines.py                 # 5-baseline table
python research/08_backtest.py                  # predictive backtest
python pipeline/15_paper_trade.py predict|score # diary (fake INR)
```
