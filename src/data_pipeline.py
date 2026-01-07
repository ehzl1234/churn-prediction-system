"""
Data Pipeline for Customer Churn Prediction.
Handles ETL, feature engineering, and data preparation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from .utils import load_config, setup_logging, get_project_root

logger = setup_logging()


class DataPipeline:
    """ETL pipeline for customer churn data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.project_root = get_project_root()
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load raw customer data from CSV."""
        if file_path is None:
            file_path = self.project_root / self.config["data"]["raw_path"]
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset."""
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col != "customer_id":
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=["customer_id"])
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")
        
        logger.info("Data cleaning complete")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data."""
        df = df.copy()
        
        # Tenure groups
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72, np.inf],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"]
        )
        
        # Monthly charge categories
        df["charge_category"] = pd.cut(
            df["monthly_charges"],
            bins=[0, 35, 70, 100, np.inf],
            labels=["Low", "Medium", "High", "Premium"]
        )
        
        # Average monthly spend (total_charges / tenure)
        df["avg_monthly_spend"] = df.apply(
            lambda x: x["total_charges"] / max(x["tenure"], 1), axis=1
        )
        
        # Spending ratio (actual vs expected)
        df["spend_ratio"] = df["monthly_charges"] / df["avg_monthly_spend"].replace(0, 1)
        df["spend_ratio"] = df["spend_ratio"].clip(0, 5)  # Cap outliers
        
        # Service count
        service_cols = ["tech_support", "online_security", "streaming_tv", "streaming_movies"]
        df["service_count"] = df[service_cols].apply(
            lambda x: sum(1 for val in x if val == "Yes"), axis=1
        )
        
        # High value customer flag
        df["is_high_value"] = (
            (df["monthly_charges"] > 70) & (df["tenure"] > 24)
        ).astype(int)
        
        logger.info(f"Created {6} new features")
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        categorical_cols = [
            "contract_type", "payment_method", "internet_service",
            "tech_support", "online_security", "streaming_tv", "streaming_movies",
            "gender", "partner", "dependents", "tenure_group", "charge_category"
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_cols: list, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        logger.info(f"Scaled {len(feature_cols)} numerical features")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix X and target vector y."""
        # Define feature columns
        feature_cols = [
            "tenure", "monthly_charges", "total_charges",
            "contract_type", "payment_method", "internet_service",
            "tech_support", "online_security", "streaming_tv", "streaming_movies",
            "gender", "senior_citizen", "partner", "dependents",
            "tenure_group", "charge_category", "avg_monthly_spend",
            "spend_ratio", "service_count", "is_high_value"
        ]
        
        X = df[feature_cols].copy()
        y = df["churn"].copy()
        
        return X, y, feature_cols
    
    def run_pipeline(self, save_processed: bool = True) -> tuple:
        """Execute the full data pipeline."""
        logger.info("=" * 50)
        logger.info("Starting Data Pipeline")
        logger.info("=" * 50)
        
        # Load
        df = self.load_data()
        
        # Clean
        df = self.clean_data(df)
        
        # Feature Engineering
        df = self.engineer_features(df)
        
        # Encode
        df = self.encode_features(df, fit=True)
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df)
        
        # Scale numerical features
        numeric_features = ["tenure", "monthly_charges", "total_charges", 
                           "avg_monthly_spend", "spend_ratio", "service_count"]
        X = self.scale_features(X, numeric_features, fit=True)
        
        # Train/test split
        test_size = self.config["data"]["test_size"]
        random_state = self.config["data"]["random_state"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        # Save processed data
        if save_processed:
            processed_path = self.project_root / self.config["data"]["processed_path"]
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=False)
            logger.info(f"Saved processed data to {processed_path}")
            
            # Save encoders and scaler
            artifacts_path = self.project_root / "models"
            joblib.dump(self.label_encoders, artifacts_path / "label_encoders.pkl")
            joblib.dump(self.scaler, artifacts_path / "scaler.pkl")
            logger.info("Saved preprocessing artifacts")
        
        logger.info("=" * 50)
        logger.info("Data Pipeline Complete")
        logger.info("=" * 50)
        
        return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test, features = pipeline.run_pipeline()
    print(f"\nFeatures used ({len(features)}): {features}")
