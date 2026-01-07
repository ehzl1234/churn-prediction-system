"""
Streamlit Interactive Dashboard for Customer Churn Prediction.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-low { color: #00c853; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #ff5722; font-weight: bold; }
    .risk-critical { color: #d32f2f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the churn predictor (cached)."""
    try:
        from src.predict import ChurnPredictor
        return ChurnPredictor()
    except FileNotFoundError:
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Customer Churn Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("⚠️ Model not found! Please train the model first by running:")
        st.code("python -m src.train_model", language="bash")
        st.stop()
    
    # Sidebar - Customer Input
    st.sidebar.header("📋 Customer Information")
    
    # Demographics
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    
    # Account Info
    st.sidebar.subheader("Account Information")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract_type = st.sidebar.selectbox(
        "Contract Type", 
        ["Month-to-month", "One year", "Two year"]
    )
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )
    
    # Charges
    st.sidebar.subheader("Billing")
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
    total_charges = st.sidebar.number_input(
        "Total Charges ($)", 
        min_value=0.0, 
        value=monthly_charges * tenure,
        step=50.0
    )
    
    # Services
    st.sidebar.subheader("Services")
    internet_service = st.sidebar.selectbox(
        "Internet Service", 
        ["DSL", "Fiber optic", "No"]
    )
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    
    # Prepare customer data
    customer_data = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "gender": gender,
        "senior_citizen": 1 if senior_citizen == "Yes" else 0,
        "partner": partner,
        "dependents": dependents
    }
    
    # Main content
    col1, col2, col3 = st.columns([2, 2, 2])
    
    # Get prediction
    if st.sidebar.button("🔮 Predict Churn", type="primary", use_container_width=True):
        with st.spinner("Analyzing customer..."):
            result = predictor.explain_prediction(customer_data)
        
        # Store result in session state
        st.session_state.prediction = result
        st.session_state.customer = customer_data
    
    # Display results if available
    if "prediction" in st.session_state:
        result = st.session_state.prediction
        customer = st.session_state.customer
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            churn_status = "WILL CHURN" if result["churn_prediction"] else "WILL STAY"
            color = "#ff5722" if result["churn_prediction"] else "#00c853"
            st.metric(
                label="Prediction",
                value=churn_status,
                delta=None
            )
        
        with col2:
            st.metric(
                label="Churn Probability",
                value=f"{result['churn_probability']:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Risk Level",
                value=result["risk_level"]
            )
        
        with col4:
            st.metric(
                label="Model Used",
                value=result["model_used"].replace("_", " ").title()
            )
        
        st.markdown("---")
        
        # Visualization row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Churn Probability")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["churn_probability"] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk %", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#c8e6c9"},
                        {'range': [30, 60], 'color': "#fff9c4"},
                        {'range': [60, 80], 'color': "#ffccbc"},
                        {'range': [80, 100], 'color': "#ffcdd2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Risk Factors")
            
            if result["risk_factors"]:
                for factor in result["risk_factors"]:
                    st.warning(f"• {factor}")
            else:
                st.success("No significant risk factors identified!")
        
        # Customer Profile
        st.markdown("---")
        st.subheader("👤 Customer Profile")
        
        profile_col1, profile_col2, profile_col3 = st.columns(3)
        
        with profile_col1:
            st.markdown("**Demographics**")
            st.write(f"Gender: {customer['gender']}")
            st.write(f"Senior: {'Yes' if customer['senior_citizen'] else 'No'}")
            st.write(f"Partner: {customer['partner']}")
            st.write(f"Dependents: {customer['dependents']}")
        
        with profile_col2:
            st.markdown("**Account**")
            st.write(f"Tenure: {customer['tenure']} months")
            st.write(f"Contract: {customer['contract_type']}")
            st.write(f"Payment: {customer['payment_method']}")
        
        with profile_col3:
            st.markdown("**Services**")
            st.write(f"Internet: {customer['internet_service']}")
            st.write(f"Tech Support: {customer['tech_support']}")
            st.write(f"Online Security: {customer['online_security']}")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Retention Recommendations")
        
        recommendations = []
        
        if result["churn_probability"] > 0.5:
            if customer["contract_type"] == "Month-to-month":
                recommendations.append("🎁 Offer a discounted annual contract upgrade")
            if customer["tech_support"] == "No":
                recommendations.append("🛠️ Provide complimentary tech support trial")
            if customer["tenure"] < 12:
                recommendations.append("🌟 Enroll in loyalty rewards program")
            if customer["payment_method"] == "Electronic check":
                recommendations.append("💳 Incentivize automatic payment setup")
            if customer["monthly_charges"] > 80:
                recommendations.append("📦 Review plan for cost optimization")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("Customer has low churn risk - maintain current engagement!")
    
    else:
        # Default state
        st.info("👈 Enter customer information in the sidebar and click **Predict Churn** to get started!")
        
        # Sample dashboard view
        st.subheader("📈 System Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", "Random Forest" if predictor else "Not Loaded")
        col2.metric("Features", len(predictor.feature_names) if predictor else 0)
        col3.metric("Status", "Ready" if predictor else "Not Ready")


if __name__ == "__main__":
    main()
