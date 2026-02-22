# 🏦 Actuarial Corporate Bankruptcy Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?style=flat-square&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Custom_Loss-green?style=flat-square)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTEENN-red?style=flat-square)

## 📊 Executive Summary
This project implements an industrial-grade machine learning pipeline for corporate bankruptcy prediction. The core challenge in this actuarial domain is **extreme class imbalance** (6,819 healthy companies vs. 220 bankruptcies). Traditional classification models optimized for pure accuracy fail catastrophically in this scenario by ignoring the minority class. 

This engine solves the imbalance problem through a rigorous two-pronged approach:
1. **Data Level**: Implementation of `SMOTEENN`, strictly isolated within a Stratified 5-Fold Cross-Validation loop to categorically prevent data leakage.
2. **Algorithmic Level**: Development of a **Custom Asymmetric Objective Function**, applying actuarial penalty weights directly into the XGBoost Newton-Raphson optimization process.

---

## 📂 Dataset Information

The model is trained and validated using the **Company Bankruptcy Prediction** dataset, originally sourced from the **Taiwan Economic Journal (1999-2009)**. 

* **Source**: [Kaggle - Company Bankruptcy Prediction](https://www.kaggle.com/datasets/pimishra/company-bankruptcy-prediction)
* **Instance Count**: 6,819 companies
* **Features**: 96 financial ratios (Net Value Per Share, Debt Ratio, etc.)
* **Target**: `Bankrupt?` (Binary: 0 for healthy, 1 for bankrupt)

> **Note**: To run this project locally, please download the `data.csv` from the link above and place it in the `/data` directory as `COMPANY BANKRUPTCY PREDICTION.csv`.

---

## 🔬 Mathematical Innovation: Asymmetric Log-Loss

In financial risk management, a False Negative (missing a bankruptcy) is exponentially more expensive than a False Positive. To address this, we derived a custom objective function for the XGBoost architecture. Let $p$ be the predicted probability after sigmoid transformation, and $y \in \{0, 1\}$ be the true label. We apply a penalty scalar $\alpha$ exclusively to the minority positive class ($y=1$).



The 1st-order derivative (Gradient) and 2nd-order derivative (Hessian) are calculated from scratch to guide the tree-boosting process:

$$ \text{Gradient}_i = \begin{cases} \alpha \cdot (p_i - 1) & \text{if } y_i = 1 \\ p_i & \text{if } y_i = 0 \end{cases} $$

$$ \text{Hessian}_i = \begin{cases} \alpha \cdot p_i(1 - p_i) & \text{if } y_i = 1 \\ p_i(1 - p_i) & \text{if } y_i = 0 \end{cases} $$

By injecting these derivatives, the model is mathematically forced to prioritize the identification of failing companies, aligning the algorithm's objective with real-world actuarial loss functions.

---

## 🏗️ Architecture & Leakage Prevention

This project utilizes `imblearn.pipeline.Pipeline` to bind transformers into a strict **Directed Acyclic Graph (DAG)**, ensuring the validation fold remains absolutely untouched during feature selection and resampling:

1.  **Pearson & Recursive VIF Filter**: Stateful removal of multicollinear features (threshold > 0.7, VIF > 10.0).
2.  **SMOTEENN Resampling**: Synthetic minority generation and noise cleaning applied *only* to the training matrix.
3.  **Estimator**: XGBoost (with Custom Loss) or Logistic Regression fitting.

---

## 📈 Key Results (5-Fold Stratified CV)

The integration of rigorous sampling and custom loss functions drastically transformed the model's ability to detect defaults:

| Architecture | Resampling Strategy | Mean Recall (Sensitivity) | Mean AUC | Mean F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline (None) | 0.193 | 0.912 | 0.274 |
| **Logistic Regression** | **SMOTEENN** | **0.818** | 0.907 | 0.259 |
| **XGBoost (Custom Loss)** | Baseline (None) | 0.284 | 0.936 | 0.359 |
| **XGBoost (Custom Loss)** | **SMOTEENN** | **0.699** | **0.927** | 0.379 |

*Note: Visual reports including ROC Curves, Confusion Matrices, and Feature Importances are automatically generated in the `/results` directory.*

---

## 🗂️ Repository Structure

```text
Bankruptcy/
├── data/                 # Dataset placeholder (.gitkeep)
├── results/              # Auto-generated categorized visual reports
│   ├── confusion_matrices/
│   ├── feature_importance/
│   └── roc_curves/
├── src/                  # Modular source code
│   ├── config/           # Model hyperparameters
│   ├── data/             # Ingestion and hold-out splitting
│   ├── evaluation/       # Visualizer engine
│   ├── features/         # VIF preprocessor & Sampler factory
│   ├── models/           # Custom Asymmetric Engine
│   └── pipeline/         # Training orchestrator
├── main.py               # System entry point
├── .gitignore            # Industrial-grade exclusion rules
└── requirements.txt      # Pinned dependencies
```

## 🚀 How to Run

1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the full orchestrator:
   ```bash
   python main.py
   ```