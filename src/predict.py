"""
Prediction module for Customer Churn Prediction System.
Loads trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Union, List, Dict

from .utils import load_config, setup_logging, get_project_root

logger = setup_logging()


class ChurnPredictor:
    """Handles predictions using the trained churn model."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.project_root = get_project_root()
        self.model = None
        self.model_name = None
        self.feature_names = None
        self.label_encoders = None
        self.scaler = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessing artifacts."""
        model_path = self.project_root / self.config["model"]["path"]
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run train_model.py first to train the model."
            )
        
        # Load model package
        model_package = joblib.load(model_path)
        self.model = model_package["model"]
        self.model_name = model_package["model_name"]
        self.feature_names = model_package["feature_names"]
        
        # Load preprocessors
        self.label_encoders = joblib.load(
            self.project_root / "models" / "label_encoders.pkl"
        )
        self.scaler = joblib.load(
            self.project_root / "models" / "scaler.pkl"
        )
        
        logger.info(f"Loaded {self.model_name} model with {len(self.feature_names)} features")
    
    def _preprocess_input(self, data: Dict) -> pd.DataFrame:
        """Preprocess a single customer record for prediction."""
        df = pd.DataFrame([data])
        
        # Feature engineering
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72, np.inf],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"]
        )
        
        df["charge_category"] = pd.cut(
            df["monthly_charges"],
            bins=[0, 35, 70, 100, np.inf],
            labels=["Low", "Medium", "High", "Premium"]
        )
        
        df["avg_monthly_spend"] = df["total_charges"] / df["tenure"].clip(lower=1)
        df["spend_ratio"] = (df["monthly_charges"] / df["avg_monthly_spend"].replace(0, 1)).clip(0, 5)
        
        service_cols = ["tech_support", "online_security", "streaming_tv", "streaming_movies"]
        df["service_count"] = df[service_cols].apply(
            lambda x: sum(1 for val in x if val == "Yes"), axis=1
        )
        
        df["is_high_value"] = ((df["monthly_charges"] > 70) & (df["tenure"] > 24)).astype(int)
        
        # Encode categorical features
        categorical_cols = [
            "contract_type", "payment_method", "internet_service",
            "tech_support", "online_security", "streaming_tv", "streaming_movies",
            "gender", "partner", "dependents", "tenure_group", "charge_category"
        ]
        
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Select and order features
        X = df[self.feature_names].copy()
        
        # Scale numerical features
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                          "avg_monthly_spend", "spend_ratio", "service_count"]
        X[numeric_features] = self.scaler.transform(X[numeric_features])
        
        return X
    
    def predict(self, data: Dict) -> Dict:
        """
        Predict churn probability for a single customer.
        
        Args:
            data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction results
        """
        X = self._preprocess_input(data)
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        result = {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability[1]),
            "retention_probability": float(probability[0]),
            "risk_level": self._get_risk_level(probability[1]),
            "model_used": self.model_name
        }
        
        return result
    
    def predict_batch(self, data_list: List[Dict]) -> List[Dict]:
        """Predict churn for multiple customers."""
        return [self.predict(data) for data in data_list]
    
    def _get_risk_level(self, churn_prob: float) -> str:
        """Categorize churn probability into risk levels."""
        if churn_prob < 0.3:
            return "Low"
        elif churn_prob < 0.6:
            return "Medium"
        elif churn_prob < 0.8:
            return "High"
        else:
            return "Critical"
    
    def explain_prediction(self, data: Dict) -> Dict:
        """Get prediction with feature contributions."""
        prediction = self.predict(data)
        
        # Add top risk factors
        risk_factors = []
        
        if data.get("contract_type") == "Month-to-month":
            risk_factors.append("Month-to-month contract (higher churn risk)")
        
        if data.get("tenure", 0) < 12:
            risk_factors.append(f"Low tenure ({data.get('tenure')} months)")
        
        if data.get("payment_method") == "Electronic check":
            risk_factors.append("Electronic check payment method")
        
        if data.get("tech_support") == "No" and data.get("internet_service") != "No":
            risk_factors.append("No tech support with internet service")
        
        if data.get("monthly_charges", 0) > 70:
            risk_factors.append(f"High monthly charges (${data.get('monthly_charges')})")
        
        prediction["risk_factors"] = risk_factors[:5]  # Top 5 factors
        
        return prediction


# Example usage
if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    # Sample customer
    sample_customer = {
        "tenure": 5,
        "monthly_charges": 85.50,
        "total_charges": 427.50,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "online_security": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No"
    }
    
    result = predictor.explain_prediction(sample_customer)
    
    print("\n" + "=" * 50)
    print("CHURN PREDICTION RESULT")
    print("=" * 50)
    print(f"Prediction: {'WILL CHURN' if result['churn_prediction'] else 'WILL STAY'}")
    print(f"Churn Probability: {result['churn_probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"\nRisk Factors:")
    for factor in result['risk_factors']:
        print(f"  • {factor}")
