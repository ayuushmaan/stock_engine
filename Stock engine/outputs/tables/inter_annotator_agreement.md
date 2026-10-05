# Inter-Annotator Agreement & Validation Report

**Sample Count Evaluated**: 200 articles

| Judge Comparison | Cohen's Kappa (κ) | Gwet's AC1 | Raw Agreement | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Human Vs Llm** | `0.000` | `0.799` | `83.0%` | `0.000` | `0.000` | `0.000` |
| **Human Vs Weak Rules** | `1.000` | `1.000` | `100.0%` | `1.000` | `1.000` | `1.000` |

> [!NOTE]
> **Interpretation of Agreement Standards (Landis & Koch, 1977)**:
> - κ > 0.80: *Almost Perfect Agreement*
> - 0.60 < κ ≤ 0.80: *Substantial Agreement*
> - 0.40 < κ ≤ 0.60: *Moderate Agreement*
> - κ ≤ 0.40: *Fair to Poor Agreement (Signals noisy pseudo-labels)*