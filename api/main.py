"""
FastAPI REST API for Customer Churn Prediction.
Provides endpoints for real-time and batch predictions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from src.predict import ChurnPredictor
from src.utils import load_config

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load predictor
predictor = None


def get_predictor():
    """Lazy load the predictor to avoid loading model on import."""
    global predictor
    if predictor is None:
        predictor = ChurnPredictor()
    return predictor


# Request/Response Models
class CustomerData(BaseModel):
    """Customer data for churn prediction."""
    tenure: int = Field(..., ge=0, le=100, description="Months with company")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges in $")
    total_charges: float = Field(..., ge=0, description="Total charges to date")
    contract_type: str = Field(..., description="Month-to-month, One year, or Two year")
    payment_method: str = Field(..., description="Payment method")
    internet_service: str = Field(..., description="DSL, Fiber optic, or No")
    tech_support: str = Field("No", description="Yes or No")
    online_security: str = Field("No", description="Yes or No")
    streaming_tv: str = Field("No", description="Yes or No")
    streaming_movies: str = Field("No", description="Yes or No")
    gender: str = Field("Male", description="Male or Female")
    senior_citizen: int = Field(0, ge=0, le=1, description="1 if senior, 0 otherwise")
    partner: str = Field("No", description="Yes or No")
    dependents: str = Field("No", description="Yes or No")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for churn prediction."""
    churn_prediction: int
    churn_probability: float
    retention_probability: float
    risk_level: str
    model_used: str
    risk_factors: Optional[List[str]] = None


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    try:
        pred = get_predictor()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=pred.model_name
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerData):
    """
    Predict churn probability for a single customer.
    
    Returns prediction, probability, risk level, and contributing factors.
    """
    try:
        pred = get_predictor()
        result = pred.explain_prediction(customer.model_dump())
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers at once.
    
    Accepts a list of customers and returns predictions for each.
    """
    try:
        pred = get_predictor()
        predictions = []
        high_risk_count = 0
        
        for customer in request.customers:
            result = pred.explain_prediction(customer.model_dump())
            predictions.append(PredictionResponse(**result))
            if result["risk_level"] in ["High", "Critical"]:
                high_risk_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    try:
        pred = get_predictor()
        return {
            "model_name": pred.model_name,
            "features": pred.feature_names,
            "feature_count": len(pred.feature_names)
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )


if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    )
