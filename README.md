# Project 2 — Supervised Learning: Fraud Detection Pipeline

**DecodeLabs Industrial Training Kit | Batch 2026**

---

## Overview

Project 2 builds a production-grade fraud detection pipeline on the cleaned e-commerce orders dataset produced by Project 1. The core challenge is handling a highly imbalanced target class (11.8% fraud) without falling into the two most common traps in imbalanced classification: the accuracy illusion and data leakage.

---

## Dataset

| Property | Value |
|---|---|
| Input | `cleaned_dataset.csv` (Project 1 output) |
| Rows | 1,200 orders |
| Features used | 33 (8 numeric/engineered + 25 OHE categorical) |
| Target | `IsFraud` (engineered binary label) |

---

## Fraud Target Engineering

There is no pre-existing fraud label in the dataset. A fraud proxy was engineered using e-commerce domain logic:

**Rule:** `IsFraud = 1` if an order is `Returned` AND either:
- Payment method is `Gift Card` or `Cash` (anonymous, hard-to-trace), OR
- `TotalPrice` exceeds the 75th percentile (high-value return risk)

**Business rationale:** This captures return fraud — the most common e-commerce fraud pattern — where customers exploit refund policies using anonymous payment methods or on high-value items.

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 1,059 | 88.2% |
| Fraudulent (1) | 141 | 11.8% |

---

## The Two Traps (and how they are avoided)

### Trap 1 — The Accuracy Illusion

A naive model predicting "all legitimate" achieves **88.2% accuracy** while catching **zero fraud**. Accuracy is completely discarded. All evaluation and hyperparameter tuning uses `roc_auc` as the scoring metric.

| Metric | What it measures |
|---|---|
| **Precision** | When we flag fraud, are we right? Minimizes false declines. |
| **Recall** | Did we catch all the fraud? Missing fraud = direct financial loss. |
| **F1-Score** | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | Overall class separation capability. Target: 0.85+ |

### Trap 2 — Data Leakage

Applying SMOTE before the train/test split means the test set contains synthetic points generated from training patterns — the model is tested on data it has already seen, producing inflated and misleading metrics.

**Correct order enforced:**
```
Stratified Split  →  [SMOTE inside pipeline per CV fold]  →  Train  →  Predict on sealed test
```

`imblearn.pipeline.Pipeline` is used instead of `sklearn.pipeline.Pipeline` because only imblearn's version natively supports the `fit_resample` interface that SMOTE requires. Using `sklearn.pipeline.Pipeline` would silently ignore or crash on SMOTE.

---

## Pipeline Architecture

### Pipeline A — Logistic Regression
```
StandardScaler  →  SMOTE  →  LogisticRegression
```
StandardScaler is mandatory here. LR's regularization penalty is distorted by high-variance features (`TotalPrice` range: 11–3330, `CartUtilizationRate` range: 0.17–1.0) without scaling.

### Pipeline B — Random Forest
```
SMOTE  →  RandomForestClassifier
```
No scaler needed. Tree-based models partition feature space ordinally — scale transformations are mathematically irrelevant to split decisions.

---

## Hyperparameter Tuning

`GridSearchCV` with `StratifiedKFold(5)` and `scoring='roc_auc'`. SMOTE is applied safely inside every fold for every parameter combination — zero leakage during tuning.

**Logistic Regression grid:**
```python
{
    'smote__k_neighbors': [3, 5],
    'classifier__C':      [0.01, 0.1, 1.0]
}
```

**Random Forest grid:**
```python
{
    'smote__k_neighbors':       [3, 5],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth':    [5, 10, None]
}
```

---

## Results

Evaluated on the **sealed, untouched test set** (240 orders, real-world 11.8% imbalance preserved).

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | see output | see output | see output | see output |
| Random Forest | see output | see output | see output | see output |

*Run the script to get exact values — results depend on the random seed and dataset split.*

---

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
python fraud_detection_ecommerce.py
```

Place `cleaned_dataset.csv` (output of Project 1) in the same directory.

---

## Output Files

```
fraud_detection_ecommerce_dashboard.png   ← 12-panel evaluation dashboard
```

---

## Key Decisions

- **Fraud proxy design** — `Returned + (AnonPayment OR HighValue)` captures real return-fraud patterns without requiring labeled data
- **`imblearn.pipeline.Pipeline`** — the only pipeline implementation that correctly handles `fit_resample`, preventing silent leakage
- **SMOTE over random oversampling** — SMOTE interpolates between existing minority samples using `x_new = x_i + λ × (x_nn − x_i)`, creating genuinely new synthetic points rather than duplicating existing ones
- **`StratifiedKFold`** — preserves the 11.8% fraud ratio in every fold, preventing any single fold from being fraud-free
- **Threshold tuning** — the default 0.5 decision threshold is suboptimal for imbalanced data; the dashboard includes threshold vs. Precision/Recall/F1 curves for both models so the optimal threshold can be read directly
