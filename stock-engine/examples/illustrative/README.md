# Illustrative / Simulated Outputs (Quarantined)

These files contain **simulated or placeholder data** and must NOT be cited as empirical results.

| File | Reason |
|------|--------|
| `ILLUSTRATIVE_cross_market_replication.json` | Simulated priors for S&P 500 comparison — no US data ingested |
| `ILLUSTRATIVE_propagation_cascades.json` | `simulate_propagation_cascades()` draws random numbers — latency/attenuation figures are from the ranges typed into that function |
| `ILLUSTRATIVE_signal_decay_analysis.json` | Shows IC 0.277 at day 1 (t=31.8) — ~30× honest results; likely leaked/stale (same-day returns used as forward) |
| `ILLUSTRATIVE_h2_predictive.json` | Stale/unverified; notebook 03_hypothesis_h2.py reports organic×closed IC = -0.016 (t=-2.82) but this file shows +0.010 (t=1.85) — sign discrepancy unresolved |

**Measured equivalents (use these instead):**
- Cross-market: Not available — listed as future work
- Propagation: `outputs/tables/propagation_measured.json`, `propagation_cascade_analysis.json`, `propagation_measured_cascades.json`