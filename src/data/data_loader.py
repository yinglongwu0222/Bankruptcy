"""
Data ingestion and partitioning module.
Ensures strict stratified splitting for highly imbalanced actuarial datasets.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataIngestionEngine:
    """
    Handles secure data loading, schema validation, and hold-out partitioning.
    """
    
    def __init__(self, file_path: str, target_col: str = "Bankrupt?"):
        """
        Args:
            file_path (str): Path to the dataset.
            target_col (str): The target variable column name.
        """
        self.file_path = Path(file_path)
        self.target_col = target_col

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads dataset from disk and validates the presence of the target column.
        
        Returns:
            pd.DataFrame: The ingested raw dataframe.
            
        Raises:
            FileNotFoundError: If the data file does not exist.
            KeyError: If the target column is missing from the schema.
        """
        if not self.file_path.exists():
            error_msg = f"Critical Error: Data file not found at {self.file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Ingesting raw data from {self.file_path}")
        df = pd.read_csv(self.file_path)
        
        if self.target_col not in df.columns:
            error_msg = f"Target column '{self.target_col}' missing from dataset schema."
            logger.error(error_msg)
            raise KeyError(error_msg)
            
        logger.info(f"Data ingested successfully. Shape: {df.shape}")
        return df

    def create_stratified_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executes a stratified train-test split to preserve minority class distribution.
        
        Args:
            df (pd.DataFrame): The full dataset.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Executing stratified split (test_size={test_size}, seed={random_state})")
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )

        logger.info(f"Training matrix: {X_train.shape}. Hold-out matrix: {X_test.shape}.")
        return X_train, X_test, y_train, y_test