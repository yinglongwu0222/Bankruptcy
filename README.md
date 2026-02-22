# 🏦 Actuarial Corporate Bankruptcy Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?style=flat-square&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Custom_Loss-green?style=flat-square)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTEENN-red?style=flat-square)

## 📊 Executive Summary
This project implements an industrial-grade machine learning pipeline for corporate bankruptcy prediction. The core challenge in this actuarial domain is **extreme class imbalance** (6,819 healthy companies vs. 220 bankruptcies). Traditional classification models optimized for pure accuracy fail catastrophically in this scenario by ignoring the minority class. 

This engine solves the imbalance problem through a rigorous two-pronged approach:
1. **Data Level**: Implementation of `SMOTEENN` (Synthetic Minority Over-sampling Technique combined with Edited Nearest Neighbors), strictly isolated within a Stratified 5-Fold Cross-Validation loop to categorically prevent data leakage.
2. **Algorithmic Level**: Development and injection of a mathematically derived **Custom Asymmetric Objective Function** from scratch, applying actuarial penalty weights directly into the XGBoost Newton-Raphson optimization process.

---

## 🔬 Mathematical Innovation: Asymmetric Log-Loss

Standard logistic loss treats False Positives (misclassifying a healthy company) and False Negatives (missing a bankrupt company) equally. In financial risk management and credit scoring, a False Negative is exponentially more expensive. 

To address this, we derived a custom objective function for the XGBoost architecture. Let $p$ be the predicted probability after sigmoid transformation, and $y \in \{0, 1\}$ be the true label. We apply a penalty scalar $\alpha$ exclusively to the minority positive class ($y=1$).

The 1st-order derivative (Gradient) and 2nd-order derivative (Hessian) are calculated from scratch:

$$ \text{Gradient}_i = \begin{cases} \alpha \cdot (p_i - 1) & \text{if } y_i = 1 \\ p_i & \text{if } y_i = 0 \end{cases} $$

$$ \text{Hessian}_i = \begin{cases} \alpha \cdot p_i(1 - p_i) & \text{if } y_i = 1 \\ p_i(1 - p_i) & \text{if } y_i = 0 \end{cases} $$

By feeding these modified gradients into the tree-boosting engine, the model is mathematically forced to prioritize the identification of failing companies, aligning the algorithm's objective with real-world actuarial loss functions.

---

## 🏗️ Architecture & Leakage Prevention (Zero Data Leakage)

A common and critical pitfall in handling imbalanced data is applying sampling techniques (like SMOTE) or scaling to the entire dataset prior to Cross-Validation. This leads to severe data leakage and artificially inflated performance metrics. 

This project completely eradicates this flaw by utilizing `imblearn.pipeline.Pipeline` to bind the transformers into a strict **Directed Acyclic Graph (DAG)**:
1. **Pearson & VIF Filter**: Stateful removal of multicollinear features (VIF > 10.0), with thresholds learned *strictly* from the training fold.
2. **SMOTEENN Resampling**: Synthetic minority generation applied *only* to the training matrix.
3. **Estimator**: XGBoost or Logistic Regression fitting.

This architecture ensures the Validation Fold remains absolutely untouched, providing a mathematically sound and reliable estimate of out-of-sample performance.

---

## 📈 Key Results (5-Fold Stratified CV)

The integration of rigorous sampling and custom loss functions drastically transformed the models' ability to detect defaults:

| Architecture | Resampling Strategy | Mean Recall (Sensitivity) | Mean AUC | Mean F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline (None) | 0.193 | 0.912 | 0.274 |
| **Logistic Regression** | **SMOTEENN** | **0.818** | 0.907 | 0.259 |
| **XGBoost (Custom Loss)** | Baseline (None) | 0.284 | 0.936 | 0.359 |
| **XGBoost (Custom Loss)** | **SMOTEENN** | **0.699** | **0.927** | 0.379 |

*Insight: Logistic Regression + SMOTEENN achieves the highest Recall (0.818), making it highly sensitive to defaults. XGBoost + SMOTEENN offers the best balance with a strong Recall (0.699) while maintaining an exceptional AUC (0.927).*

---

## 🗂️ Repository Structure

```text
Bankruptcy/
├── data/
│   └── COMPANY BANKRUPTCY PREDICTION.csv
├── results/                  # Auto-generated visual reports
│   ├── confusion_matrices/
│   ├── feature_importance/
│   └── roc_curves/
├── src/
│   ├── config/               # Factory pattern for immutable hyperparameters
│   ├── data/                 # Ingestion and hold-out splitting engine
│   ├── evaluation/           # Automated and categorized plotting engine
│   ├── features/             # Stateful VIF preprocessor & Sampler factory
│   ├── models/               # Base estimators and Custom Math Engines
│   └── pipeline/             # Strict CV and Imblearn DAG orchestration
├── main.py                   # System entry point
└── README.md
```

## 🚀 How to Run

1. Ensure Python environment is properly configured.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn statsmodels matplotlib seaborn
   ```
3. Execute the full orchestrator to run K-Fold CV, production fitting, and automatic generation of all evaluation plots:
   ```bash
   python main.py
   ```
*Note: Depending on CPU architecture (e.g., Apple Silicon), the VIF recursive feature elimination step may take 1-3 minutes. Real-time progress logs are printed to the console.*