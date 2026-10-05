# Five-Baseline Signal Benchmark Table

| Signal Architecture | Mean IC | ICIR | HAC t-stat | Raw p-val | FDR p-val | Ann. L/S Spread (%) | Sharpe | Max DD (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw GDELT Tone** | `+0.0074` | `0.195` | `+0.86` | `0.3887` | `0.6006` | `+6.2%` | `1.82` | `-0.8%` |
| **2. FinBERT Baseline** | `+0.0102` | `0.257` | `+1.15` | `0.2513` | `0.6006` | `+12.7%` | `3.22` | `-0.9%` |
| **3. Temporal Only** | `+0.0071` | `0.186` | `+0.84` | `0.3986` | `0.6006` | `+7.2%` | `1.99` | `-1.1%` |
| **4. Credibility Only** | `+0.0041` | `0.116` | `+0.52` | `0.6006` | `0.6006` | `+1.8%` | `0.51` | `-1.0%` |
| **5. Full Proposed Signal** | `+0.0042` | `0.115` | `+0.53` | `0.5980` | `0.6006` | `+3.8%` | `1.19` | `-1.1%` |

> [!NOTE]
> **Statistical Rigor Criteria**:
> - **HAC t-stat**: Newey-West adjusted for serial autocovariance up to 5 trading days.
> - **FDR p-val**: Benjamini-Hochberg false discovery rate adjusted across all tested signal families.
> - **L/S Spread**: Top quintile (Q5) minus bottom quintile (Q1) daily rebalanced long-short spread.