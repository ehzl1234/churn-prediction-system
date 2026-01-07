# 🔮 Customer Churn Prediction System

A complete **end-to-end machine learning system** for predicting customer churn, featuring data pipeline, model training, REST API, and interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🎯 Project Overview

This project demonstrates production-ready ML skills:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Pipeline** | pandas, scikit-learn | ETL, feature engineering, preprocessing |
| **Model Training** | scikit-learn, XGBoost | Multi-model comparison, cross-validation |
| **REST API** | FastAPI, Pydantic | Real-time predictions, batch processing |
| **Dashboard** | Streamlit, Plotly | Interactive customer analysis |
| **Testing** | pytest | Unit tests for all components |

## 📁 Project Structure

```
churn-prediction-system/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Feature-engineered data
├── models/                     # Trained model artifacts
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        # ETL & feature engineering
│   ├── train_model.py          # Model training & evaluation
│   ├── predict.py              # Prediction utilities
│   └── utils.py                # Helper functions
├── api/
│   └── main.py                 # FastAPI REST endpoint
├── app/
│   └── streamlit_app.py        # Interactive dashboard
├── notebooks/
│   └── exploration.ipynb       # Exploratory Data Analysis
├── tests/
│   └── test_pipeline.py        # Unit tests
├── config.yaml                 # Configuration settings
├── requirements.txt            # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ehzl1234/churn-prediction-system.git
cd churn-prediction-system
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m src.train_model
```

This will:
- Load and preprocess the customer data
- Engineer features (tenure groups, spend ratios, etc.)
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Perform cross-validation and select the best model
- Save the model to `models/churn_model.pkl`

### 3. Run the API

```bash
cd api
uvicorn main:app --reload
```

API available at: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### 4. Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard available at: `http://localhost:8501`

## 📊 Features

### Data Pipeline
- Automated data cleaning and validation
- Feature engineering (6 new features)
- Stratified train/test split
- Preprocessing artifact persistence

### Model Training
- **Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Selection**: Automatic best model selection
- **Feature Importance**: Ranked feature contributions

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model details |

### Dashboard Features
- Customer input form
- Real-time churn probability
- Risk level gauge chart
- Risk factor analysis
- Retention recommendations

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📈 Model Performance

After training, expect results similar to:

| Model | ROC-AUC | Accuracy | F1-Score |
|-------|---------|----------|----------|
| Random Forest | ~0.85 | ~0.80 | ~0.75 |
| XGBoost | ~0.86 | ~0.82 | ~0.77 |
| Logistic Regression | ~0.83 | ~0.78 | ~0.72 |

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data paths
- Feature lists
- Model hyperparameters
- API settings

## 📝 License

MIT License - feel free to use for learning and portfolio purposes.

## 👤 Author

**Firdaus** - Data Analyst / ML Engineer

- GitHub: [@ehzl1234](https://github.com/ehzl1234)
