# GK-XGBoost: Focal-Loss-Modified Gradient Boosting for Cybersecurity Risk Classification

A focal-loss-modified XGBoost framework for classifying university students into
cybersecurity risk levels (Low / Medium / High) under class imbalance, with
role-differentiated SHAP explanations for students, educators, and IT administrators.

**Author:** Koku Godwin Komla · **Supervisor:** Dr. Eric Opoku Osei
**Institution:** Kwame Nkrumah University of Science and Technology (KNUST)

---

## Overview

Standard classifiers under-detect the small high-risk minority that most warrants
intervention. GK-XGBoost applies two complementary modifications to XGBoost:

1. **Operation F (Focal loss):** the multiclass softmax objective is replaced by a
   focal-loss-weighted objective (custom gradient/Hessian) that focuses learning on
   hard, under-represented minority examples.
2. **Operation M (Reduced min_child_weight):** the minimum child weight is reduced
   from 8 to 3, letting leaf nodes form from smaller groups of minority instances.

A four-cell ablation shows the two modifications are **complementary**: neither helps
alone (focal-only 0.3528; reduced-min-child-weight-only 0.4230; both below the
baseline 0.4366), and only the combination exceeds baseline (0.4466). The gain is
therefore not attributable to the hyperparameter change alone.

## Key results (KNUST field data, n = 1,250, leakage-free coherent-scale design)

| Model | Accuracy | Macro F1 | AUC-ROC | AUC-PR |
|-------|:--------:|:--------:|:-------:|:------:|
| Baseline XGBoost | 0.6912 | 0.4366 | 0.6702 | 0.4802 |
| SMOTE + XGBoost | 0.6882 | 0.4488 | 0.6743 | 0.4812 |
| Class-weighted XGBoost | 0.6139 | 0.4853 | 0.6780 | 0.4851 |
| LightGBM | 0.6334 | 0.4619 | 0.6671 | 0.4676 |
| **GK-XGBoost** | **0.6965** | 0.4466 | 0.6732 | **0.4906** |

Read against the **standard baseline**, GK-XGBoost improves all four metrics; read
against the full five-model field, it leads on Accuracy and AUC-PR, while the
class-weighted baseline leads on Macro F1 and AUC-ROC. On the **pre-declared primary
metric (Macro F1)** the improvement over baseline is +0.0100 and **not statistically
significant** (Wilcoxon W = 108, p = 0.1485, CI crosses zero). The **significant**
result is on **AUC-PR** (+0.0104, W = 68, p = 0.0096), reported as a secondary finding;
it survives Bonferroni correction (0.05/4 = 0.0125). In absolute terms the model detects
about **two additional High-Risk students out of 179** (29 to 31), with ~148 still
missed by both models. Internal consistency of the behaviour scale: Cronbach's
alpha = 0.68 (below the 0.70 convention; split-half correlation 0.53).

## Repository structure

```
.
├── src/
│   ├── gk_xgboost_pipeline.py     # Main pipeline: 5-model comparison, ablation, stats
│   ├── hyperparameter_tuning.py   # Optuna tuning (reproduces the tuned params)
│   ├── shap_analysis.py           # SHAP importance + role-differentiated views
│   ├── transfer_diagnostics.py    # Transfer eval + negative-result diagnostics
│   └── generate_figures.py        # Figure rendering
├── figures/                       # Figures 1-4 (PNG, 200 DPI)
├── results/                       # Machine-readable results + SHAP table
├── docs/                          # Questionnaire + Google Form Apps Script
├── data/README.md                 # Data access instructions (data not distributed)
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducing the results

```bash
pip install -r requirements.txt

# Place the KNUST data file (response.xlsx) in the working directory,
# or a public dataset in data/ (see data/README.md), then:
cd src
python gk_xgboost_pipeline.py       # main comparison + ablation + statistics
python hyperparameter_tuning.py     # reproduce tuned hyperparameters
python shap_analysis.py             # SHAP importance + role views
python transfer_diagnostics.py      # negative-result diagnostics (power, separability)
```

### Configuration (fixed for reproducibility)
- Seeds: 42, 123, 456, 789, 1024 (5 seeds × 5 folds = 25 measurements)
- Baseline `min_child_weight = 8`
- GK-XGBoost `gamma = 3.1577`, `alpha = 0.6355`, `min_child_weight = 3` (all Optuna-tuned)
- Label cut points: Secure-score >=22 Low, 12-21 Medium, <12 High (4.94:1)
- Achieved claim tier: **Tier 1 bounded** (scoped to KNUST survey respondents)
- Shared: 300 estimators, depth 6, learning rate 0.05, subsample 0.8, colsample 0.8

## Transfer / boundary conditions

GK-XGBoost was evaluated on two independent public datasets. It improves minority
detection **when meaningful class imbalance and learnable feature signal coexist**
(KNUST). It offers no benefit under diagnosed boundary conditions:

- **WUSTL-EHMS-2020** — trivially separable (a depth-3 tree scores 0.9987); ceiling
  effect leaves no hard examples for focal loss (modality mismatch).
- **Moroccan HTTPS Awareness** — underpowered (n = 440; minimum detectable effect
  d ≈ 0.81 >> observed effect) and low feature signal after leakage control;
  the comparison is inconclusive, not evidence of no effect.

These are reported honestly as diagnosed boundary conditions, delimiting the
operating range of the method.

## Data availability

The KNUST field data are ethics-restricted and not distributed here (see
`data/README.md`). The two public datasets are available from their original
repositories.

## License

MIT License — see `LICENSE`.

## Citation

If you use this code, please cite the associated thesis/paper (details to be added
upon publication).
