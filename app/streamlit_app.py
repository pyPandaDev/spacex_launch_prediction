"""
SpaceX Launch Success Prediction - Streamlit App
Interactive web application for predicting launch success
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="🚀 SpaceX Launch Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success {
        color: #4CAF50;
        font-weight: bold;
    }
    .failure {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model_and_data():
    """Load trained model and metadata"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.pkl')
        feature_names_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.pkl')
        
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        feature_names = joblib.load(feature_names_path)
        
        return model, metadata, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running the notebooks.")
        return None, None, None

# Header
st.markdown('<div class="main-header">🚀 SpaceX Launch Success Predictor</div>', unsafe_allow_html=True)
st.markdown("### Predict the success probability of a SpaceX Falcon 9 launch using Machine Learning")

# Load model
model, metadata, feature_names = load_model_and_data()

if model is None:
    st.stop()

# Sidebar - Model Information
st.sidebar.header("📊 Model Information")
st.sidebar.markdown(f"**Model:** {metadata['model_name']}")
st.sidebar.markdown(f"**Accuracy:** {metadata['accuracy']:.2%}")
st.sidebar.markdown(f"**Precision:** {metadata['precision']:.2%}")
st.sidebar.markdown(f"**Recall:** {metadata['recall']:.2%}")
st.sidebar.markdown(f"**F1 Score:** {metadata['f1']:.2%}")
if metadata.get('roc_auc'):
    st.sidebar.markdown(f"**ROC-AUC:** {metadata['roc_auc']:.2%}")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This app predicts whether a SpaceX Falcon 9 launch will be successful based on "
    "historical data and machine learning. Adjust the parameters on the right to make predictions."
)

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">🎯 Launch Parameters</div>', unsafe_allow_html=True)
    
    # Input fields
    payload_mass = st.number_input(
        "Payload Mass (kg)",
        min_value=0.0,
        max_value=10000.0,
        value=3000.0,
        step=100.0,
        help="Mass of the payload in kilograms"
    )
    
    booster_type = st.selectbox(
        "Booster Type",
        options=['F9 v1.0', 'F9 v1.1', 'F9 Full Thrust', 'F9 Block 4', 'F9 Block 5', 'Falcon Heavy'],
        index=4,
        help="Type of Falcon 9 booster version"
    )
    
    launch_site = st.selectbox(
        "Launch Site",
        options=['CCAFS', 'VAFB', 'KSC'],
        index=0,
        help="Launch site location"
    )
    
    orbit_type = st.selectbox(
        "Target Orbit",
        options=['LEO', 'GTO', 'Polar', 'HEO', 'Other'],
        index=0,
        help="Target orbit type"
    )
    
    booster_reused = st.radio(
        "Booster Reused?",
        options=['No', 'Yes'],
        index=1,
        help="Is this a reused booster?"
    )
    
    booster_flight_number = st.slider(
        "Booster Flight Number",
        min_value=1,
        max_value=10,
        value=1 if booster_reused == 'No' else 2,
        help="How many times has this booster flown?"
    )
    
    year = st.slider(
        "Year",
        min_value=2010,
        max_value=2030,
        value=datetime.now().year,
        help="Launch year"
    )
    
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=datetime.now().month - 1,
        help="Launch month"
    )
    
    is_nasa = st.checkbox("NASA Mission", value=False)
    is_commercial = st.checkbox("Commercial Mission", value=True)

with col2:
    st.markdown('<div class="sub-header">🔮 Prediction Results</div>', unsafe_allow_html=True)
    
    # Calculate derived features
    orbit_difficulty_map = {'LEO': 1, 'Polar': 2, 'HEO': 3, 'GTO': 4, 'Other': 2.5}
    orbit_difficulty = orbit_difficulty_map[orbit_type]
    payload_orbit_ratio = payload_mass / orbit_difficulty
    quarter = (month - 1) // 3 + 1
    
    # Estimate cumulative launches and success rate based on year
    cumulative_launches = max(1, (year - 2010) * 10)
    cumulative_success_rate = 0.95 if year >= 2017 else 0.85
    days_since_first = (year - 2010) * 365 + month * 30
    
    # Create feature dictionary
    features = {
        'Payload Mass (kg)': payload_mass,
        'Year': year,
        'Month': month,
        'Quarter': quarter,
        'Booster_Reused': 1 if booster_reused == 'Yes' else 0,
        'Booster_Flight_Number': booster_flight_number,
        'Cumulative_Launches': cumulative_launches,
        'Cumulative_Success_Rate': cumulative_success_rate,
        'Orbit_Difficulty': orbit_difficulty,
        'Payload_Orbit_Ratio': payload_orbit_ratio,
        'Days_Since_First_Launch': days_since_first,
        'Is_NASA': 1 if is_nasa else 0,
        'Is_Commercial': 1 if is_commercial else 0
    }
    
    # Add one-hot encoded features
    for feat in feature_names:
        if feat not in features:
            if 'Booster_Type_' in feat:
                features[feat] = 1 if feat == f'Booster_Type_{booster_type}' else 0
            elif 'Launch_Site_Simplified_' in feat:
                features[feat] = 1 if feat == f'Launch_Site_Simplified_{launch_site}' else 0
            elif 'Orbit_Simplified_' in feat:
                features[feat] = 1 if feat == f'Orbit_Simplified_{orbit_type}' else 0
            else:
                features[feat] = 0
    
    # Create DataFrame in correct order
    input_df = pd.DataFrame([features])[feature_names]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    # Display results
    st.markdown("---")
    
    if prediction == 1:
        st.markdown(f'<div class="success" style="font-size: 2rem; text-align: center;">✅ SUCCESS PREDICTED</div>', 
                    unsafe_allow_html=True)
        st.success(f"**Success Probability: {probability[1]:.1%}**")
    else:
        st.markdown(f'<div class="failure" style="font-size: 2rem; text-align: center;">❌ FAILURE PREDICTED</div>', 
                    unsafe_allow_html=True)
        st.error(f"**Failure Probability: {probability[0]:.1%}**")
    
    # Probability gauge
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh([0], [probability[1]], color='green' if prediction == 1 else 'red', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Success Probability', fontsize=12)
    ax.set_title('Confidence Level', fontsize=14, fontweight='bold')
    for i in [0.25, 0.5, 0.75]:
        ax.axvline(i, color='gray', linestyle='--', alpha=0.5)
    ax.text(probability[1], 0, f' {probability[1]:.1%}', va='center', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    # Risk factors
    st.markdown("---")
    st.markdown("### 📋 Mission Analysis")
    
    col2a, col2b = st.columns(2)
    
    with col2a:
        st.metric("Payload Mass", f"{payload_mass:,.0f} kg")
        st.metric("Orbit Difficulty", f"{orbit_difficulty}/4")
        st.metric("Booster Experience", f"{booster_flight_number} flights")
    
    with col2b:
        st.metric("Mission Type", "NASA" if is_nasa else ("Commercial" if is_commercial else "Other"))
        st.metric("Booster Status", "Reused" if booster_reused == 'Yes' else "New")
        st.metric("Launch Era", f"{year}")

# Feature Importance Section
st.markdown("---")
st.markdown('<div class="sub-header">📊 Feature Influence on Prediction</div>', unsafe_allow_html=True)

try:
    # Try to load SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # SHAP waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0] if isinstance(shap_values, list) else shap_values[0],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
        data=input_df.iloc[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig)
    
    st.info("**Interpretation:** Red bars push the prediction towards Failure, blue bars push towards Success. "
            "The longer the bar, the stronger the influence.")
    
except Exception as e:
    st.warning(f"SHAP explanations not available: {e}")
    
    # Fallback to feature importance if model has it
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importances['Feature'], importances['Importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        ax.invert_yaxis()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>🚀 Built with Streamlit | Data from SpaceX Historical Launches | Model trained using XGBoost/Random Forest</p>
        <p>Predictions are based on historical data and should be used for educational purposes only.</p>
    </div>
    """,
    unsafe_allow_html=True
)
