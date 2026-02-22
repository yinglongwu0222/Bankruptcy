# Corporate Default Risk Modeling: An Actuarial Machine Learning Pipeline 📉🤖

**Domain**: Quantitative Finance & Actuarial Science | **Focus**: Extreme Class Imbalance & Model Explainability (XAI)

## 📊 Executive Summary
In financial risk management, predicting corporate bankruptcy is fundamentally a "rare event" modeling problem. This project provides an end-to-end, highly automated machine learning pipeline to assess corporate default risk based on financial indicators. 

Developed with rigorous quantitative standards, the pipeline successfully isolates the signal from the noise in highly skewed financial datasets (**6,819 total companies, but only 220 actual defaults/bankruptcies**).

## 🗄️ Data Provenance & Structure
The dataset used in this pipeline is sourced from public records on **Kaggle** (originally compiled from the Taiwan Economic Journal). It encompasses 6,819 corporate entities and an initial set of 95 distinct financial features, targeting the binary indicator `Bankrupt?`.

*Note: Adhering to enterprise data-sharing best practices, the raw dataset is excluded from version control via `.gitignore`.*
**Data Setup Instruction**: To replicate this project locally, source the dataset from Kaggle and place it precisely at `data/COMPANY BANKRUPTCY PREDICTION.csv` to match the pipeline configuration.

## 🧠 Key Quantitative Discoveries
Extensive benchmarking across multiple algorithms and sampling strategies yielded the following actionable insights for credit risk assessment:

1. **Optimal Risk Identification**: The **Random Forest** model paired with **SMOTEENN** sampling achieved an exceptional **Recall of 0.77**. In a real-world actuarial context, this implies the model successfully flagged 77% of actual defaulting companies, making it the most conservative and protective configuration for portfolio risk management.
2. **Best Balanced Performance**: **XGBoost** consistently outperformed other models across most sampling methods, offering the optimal trade-off between Precision and Recall (captured via highest AUC and MCC metrics).
3. **Feature Dimensionality Reduction**: Utilizing Variance Inflation Factor (VIF < 10) filtering to eliminate multicollinearity, combined with **SHAP** (SHapley Additive exPlanations) values and LightGBM analysis, the initial 95 financial indicators were aggressively pruned to the **top 30 most predictive features**, drastically reducing computational overhead while enhancing model transparency.

## 🗂 Project Architecture
The codebase is structured for scalability and decoupled configuration:

    project_root/
    ├── config/          # config.yaml (Pipeline toggles, exact paths, top_k=30)
    ├── data/            # Local directory for COMPANY BANKRUPTCY PREDICTION.csv
    ├── processing/      # Core logic: feature_selection, sampler, data_loader
    ├── models/          # ML algorithms and model_configs
    ├── evaluation/      # Metrics calculation and visualizer (ROC/PR Curves)
    ├── results/         # Auto-generates /roc_curves, /pr_curves, /confusion_matrices
    ├── requirements.txt # Environment lock file (Strictly versioned, Py 3.12+)
    └── run.py           # Main execution script

## 🚀 Reproduction & Execution

This pipeline has been fully tested and optimized for modern architecture (e.g., Apple Silicon / M-Series, Python 3.12).

### Step 1: Environment Setup
Ensure you have your quantitative environment ready. The precise dependencies are cleanly locked in `requirements.txt`.

    # Activate your data science environment
    conda activate ds_quant

    # Install dependencies
    pip install -r requirements.txt

### Step 2: Run the Pipeline

    python run.py

*Note: A complete pipeline execution—evaluating 5 sampling methods across 5 models with full feature engineering—completes in approximately 3.5 minutes on standard M-Series hardware.*

## 📈 Outputs & Visual Analytics
All experiment artifacts are automatically generated in the `results/` directory, providing risk managers with:
- **Financial Metrics**: F1-Score, Recall, Precision, MCC, ROC-AUC (summarized in `results_summary.csv`).
- **Visual Analytics**: Individual ROC/PR curves for all model combinations, Confusion Matrices, and Correlation Heatmaps.