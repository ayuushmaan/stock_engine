# Five-Baseline Signal Benchmark Table

| Signal Architecture | Mean IC | ICIR | HAC t-stat | Raw p-val | FDR p-val | Ann. L/S Spread (%) | Sharpe | Max DD (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw GDELT Tone** | `+0.0117` | `0.164` | `+0.97` | `0.3323` | `0.4287` | `-22.7%` | `-3.48` | `-2.9%` |
| **2. FinBERT Baseline** | `+0.0104` | `0.122` | `+0.79` | `0.4287` | `0.4287` | `+1.9%` | `0.29` | `-1.2%` |
| **3. Temporal Only** | `+0.0106` | `0.148` | `+0.89` | `0.3753` | `0.4287` | `-21.5%` | `-3.53` | `-2.8%` |
| **4. Credibility Only** | `+0.0167` | `0.239` | `+1.33` | `0.1833` | `0.4287` | `-5.1%` | `-0.77` | `-2.5%` |
| **5. Full Proposed Signal** | `+0.0149` | `0.216` | `+1.20` | `0.2293` | `0.4287` | `-11.5%` | `-1.84` | `-2.5%` |

> [!NOTE]
> **Statistical Rigor Criteria**:
> - **HAC t-stat**: Newey-West adjusted for serial autocovariance up to 5 trading days.
> - **FDR p-val**: Benjamini-Hochberg false discovery rate adjusted across all tested signal families.
> - **L/S Spread**: Top quintile (Q5) minus bottom quintile (Q1) daily rebalanced long-short spread.