"""
Unit tests for the Churn Prediction System.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDataPipeline:
    """Tests for the data pipeline module."""
    
    def test_load_data(self):
        """Test that data loads correctly."""
        from src.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        df = pipeline.load_data()
        
        assert df is not None
        assert len(df) > 0
        assert "customer_id" in df.columns
        assert "churn" in df.columns
    
    def test_clean_data(self):
        """Test data cleaning removes duplicates."""
        from src.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        df = pipeline.load_data()
        cleaned = pipeline.clean_data(df)
        
        # Check no duplicates
        assert cleaned["customer_id"].duplicated().sum() == 0
    
    def test_feature_engineering(self):
        """Test that new features are created."""
        from src.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        df = pipeline.load_data()
        df = pipeline.clean_data(df)
        df = pipeline.engineer_features(df)
        
        # Check new features exist
        expected_features = ["tenure_group", "charge_category", 
                           "avg_monthly_spend", "service_count", "is_high_value"]
        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"
    
    def test_prepare_features(self):
        """Test feature preparation returns correct shapes."""
        from src.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        df = pipeline.load_data()
        df = pipeline.clean_data(df)
        df = pipeline.engineer_features(df)
        df = pipeline.encode_features(df)
        
        X, y, features = pipeline.prepare_features(df)
        
        assert len(X) == len(y)
        assert len(features) > 0
        assert X.shape[1] == len(features)


class TestModelTrainer:
    """Tests for the model training module."""
    
    def test_cross_validation(self):
        """Test that cross-validation runs without errors."""
        from src.data_pipeline import DataPipeline
        from src.train_model import ModelTrainer
        
        # Get small sample of data
        pipeline = DataPipeline()
        X_train, X_test, y_train, y_test, features = pipeline.run_pipeline(save_processed=False)
        
        trainer = ModelTrainer()
        cv_results = trainer.cross_validate(X_train, y_train, cv_folds=3)
        
        assert len(cv_results) > 0
        for model_name, results in cv_results.items():
            assert "mean_auc" in results
            assert results["mean_auc"] >= 0 and results["mean_auc"] <= 1


class TestPredictor:
    """Tests for the prediction module."""
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for testing."""
        return {
            "tenure": 12,
            "monthly_charges": 65.50,
            "total_charges": 786.00,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "tech_support": "No",
            "online_security": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "No",
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "No",
            "dependents": "No"
        }
    
    def test_prediction_format(self, sample_customer):
        """Test that prediction returns expected format."""
        try:
            from src.predict import ChurnPredictor
            predictor = ChurnPredictor()
            
            result = predictor.predict(sample_customer)
            
            assert "churn_prediction" in result
            assert "churn_probability" in result
            assert "risk_level" in result
            assert result["churn_prediction"] in [0, 1]
            assert 0 <= result["churn_probability"] <= 1
        except FileNotFoundError:
            pytest.skip("Model not trained yet")
    
    def test_explain_prediction(self, sample_customer):
        """Test that explanation includes risk factors."""
        try:
            from src.predict import ChurnPredictor
            predictor = ChurnPredictor()
            
            result = predictor.explain_prediction(sample_customer)
            
            assert "risk_factors" in result
            assert isinstance(result["risk_factors"], list)
        except FileNotFoundError:
            pytest.skip("Model not trained yet")


class TestUtils:
    """Tests for utility functions."""
    
    def test_load_config(self):
        """Test configuration loading."""
        from src.utils import load_config
        
        config = load_config()
        
        assert "data" in config
        assert "model" in config
        assert "api" in config
    
    def test_get_project_root(self):
        """Test project root detection."""
        from src.utils import get_project_root
        
        root = get_project_root()
        
        assert root.exists()
        assert (root / "config.yaml").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
