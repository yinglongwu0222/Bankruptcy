"""
Industrial Entry Point for Actuarial Default Prediction.
Orchestrates the DAG pipeline, ensuring zero data leakage, strict CV evaluation,
and automated generation of visual performance reports.
"""

import logging
import numpy as np
from src.config.model_configs import ModelConfigFactory
from src.data.data_loader import DataIngestionEngine
from src.features.preprocessor import FinancialFeatureSelector
from src.features.sampler import SamplerFactory
from src.models.classifier import ActuarialModelEngine
from src.pipeline.trainer import ActuarialPipelineEngine
from src.evaluation.visualizer import ActuarialVisualizer

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Bankruptcy_Engine")

def main():
    logger.info("Initializing Actuarial Default Prediction Engine...")

    # 1. Ingest Data
    DATA_PATH = "data/COMPANY BANKRUPTCY PREDICTION.csv"
    data_engine = DataIngestionEngine(file_path=DATA_PATH)
    df_raw = data_engine.load_raw_data()

    # 2. Strict Hold-out Split (20% isolated for final production test)
    X_train, X_test, y_train, y_test = data_engine.create_stratified_split(df_raw)

    # 3. Initialize Visualizer
    visualizer = ActuarialVisualizer(base_dir="results")

    # 4. Define the experiment grid
    models_to_test = ["xgboost", "logistic"] 
    samplers_to_test = ["SMOTEENN", "None"]

    # 5. Execute Pipeline
    for model_name in models_to_test:
        for sampler_name in samplers_to_test:
            logger.info(f"\n{'='*40}\nEvaluating: {model_name.upper()} with {sampler_name}\n{'='*40}")

            preprocessor = FinancialFeatureSelector(corr_threshold=0.7, vif_threshold=10.0)
            sampler = SamplerFactory.get_sampler(method_name=sampler_name)
            params = ModelConfigFactory.get_params(model_name)
            
            # 【外科手术修复 1】：逻辑互斥设计 (A/B Test)
            use_custom_loss = (model_name == "xgboost" and sampler_name.lower() == "none")
            
            classifier = ActuarialModelEngine(
                model_name=model_name, 
                params=params, 
                custom_loss=use_custom_loss
            )

            pipeline_engine = ActuarialPipelineEngine(
                preprocessor=preprocessor,
                sampler=sampler,
                classifier=classifier
            )

            # --- Stage 1: Rigorous Cross-Validation ---
            cv_results = pipeline_engine.cross_validate(X_train, y_train, n_splits=5)
            logger.info(f"CV Results for {model_name.upper()} + {sampler_name}: {cv_results}")

            # --- Stage 2: Production Training & Visual Reporting ---
            logger.info("Executing Production Fit on full training data...")
            pipeline_engine.fit_production(X_train, y_train)

            logger.info("Executing inference on untouched Hold-out Test Set...")
            y_pred = pipeline_engine.predict_production(X_test)
            
            # 【架构修复】：必须通过 pipeline.predict_proba 调用！
            # 确保 X_test 先经过 preprocessor 降维 (96 -> 58 特征)，再进入分类器，防止维度灾难
            try:
                y_proba = pipeline_engine.pipeline.predict_proba(X_test)[:, 1]
            except (NotImplementedError, AttributeError):
                y_proba = y_pred
                logger.warning("Fallback to hard labels for ROC generation.")

            # Generate Plots
            plot_identifier = f"{model_name}_{sampler_name}"
            visualizer.plot_roc_curve(y_test, y_proba, plot_identifier)
            visualizer.plot_confusion_matrix(y_test, y_pred, plot_identifier)

            # Plot Feature Importances
            try:
                fitted_preprocessor = pipeline_engine.pipeline.named_steps['preprocessor']
                feature_names = fitted_preprocessor.selected_features_
                model_step = pipeline_engine.pipeline.named_steps['model']
                
                if model_name in ["xgboost", "random_forest"]:
                    importances = model_step.model.feature_importances_
                    visualizer.plot_feature_importance(feature_names, importances, plot_identifier)
                elif model_name == "logistic":
                    importances = np.abs(model_step.model.coef_[0])
                    visualizer.plot_feature_importance(feature_names, importances, plot_identifier)
            except Exception as e:
                logger.warning(f"Could not extract feature importance for {plot_identifier}: {e}")

    logger.info("All pipeline executions and visual exports completed successfully.")

if __name__ == "__main__":
    main()