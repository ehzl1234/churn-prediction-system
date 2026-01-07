"""
Model Training for Customer Churn Prediction.
Trains multiple models, performs cross-validation, and selects the best model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .data_pipeline import DataPipeline
from .utils import load_config, setup_logging, get_project_root

logger = setup_logging()


class ModelTrainer:
    """Trains and evaluates multiple ML models for churn prediction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.project_root = get_project_root()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
    
    def _get_models(self) -> dict:
        """Initialize candidate models with hyperparameters."""
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        }
        
        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=2,  # Handle imbalanced classes
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        else:
            logger.warning("XGBoost not available, skipping")
        
        return models
    
    def cross_validate(self, X_train, y_train, cv_folds: int = 5) -> dict:
        """Perform cross-validation on all models."""
        logger.info("\n" + "=" * 50)
        logger.info("Cross-Validation Results")
        logger.info("=" * 50)
        
        cv_results = {}
        models = self._get_models()
        
        for name, model in models.items():
            logger.info(f"\nEvaluating {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, scoring="roc_auc"
            )
            
            cv_results[name] = {
                "mean_auc": cv_scores.mean(),
                "std_auc": cv_scores.std(),
                "scores": cv_scores
            }
            
            logger.info(f"  ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def train_models(self, X_train, y_train, X_test, y_test) -> dict:
        """Train all models and evaluate on test set."""
        logger.info("\n" + "=" * 50)
        logger.info("Training Models")
        logger.info("=" * 50)
        
        models = self._get_models()
        results = {}
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
            
            results[name] = metrics
            
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        self.results = results
        return results
    
    def select_best_model(self, metric: str = "roc_auc") -> tuple:
        """Select the best model based on specified metric."""
        if not self.results:
            raise ValueError("No models trained yet. Call train_models first.")
        
        best_score = -1
        best_name = None
        
        for name, metrics in self.results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_name = name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Best Model: {best_name}")
        logger.info(f"Best {metric}: {best_score:.4f}")
        logger.info("=" * 50)
        
        return self.best_model, best_name
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the best model."""
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not set.")
        
        # Get importance based on model type
        if hasattr(self.best_model, "feature_importances_"):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importance = np.abs(self.best_model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    def save_model(self, model_path: str = None) -> str:
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model to save. Call select_best_model first.")
        
        if model_path is None:
            model_path = self.project_root / self.config["model"]["path"]
        else:
            model_path = Path(model_path)
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model with metadata
        model_package = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "metrics": self.results[self.best_model_name],
            "trained_at": datetime.now().isoformat()
        }
        
        joblib.dump(model_package, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def run_training(self) -> dict:
        """Execute the full training pipeline."""
        logger.info("\n" + "=" * 60)
        logger.info("CHURN PREDICTION MODEL TRAINING")
        logger.info("=" * 60)
        
        # Run data pipeline
        pipeline = DataPipeline()
        X_train, X_test, y_train, y_test, feature_cols = pipeline.run_pipeline()
        self.feature_names = feature_cols
        
        # Cross-validation
        cv_results = self.cross_validate(
            X_train, y_train, 
            cv_folds=self.config["model"]["cv_folds"]
        )
        
        # Train models
        results = self.train_models(X_train, y_train, X_test, y_test)
        
        # Select best model
        best_model, best_name = self.select_best_model()
        
        # Get feature importance
        importance = self.get_feature_importance()
        if importance is not None:
            logger.info("\nTop 10 Feature Importances:")
            for _, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        model_path = self.save_model()
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        return {
            "cv_results": cv_results,
            "test_results": results,
            "best_model": best_name,
            "model_path": model_path,
            "feature_importance": importance
        }


if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_training()
