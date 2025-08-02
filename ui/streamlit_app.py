"""
Phase 4: Clinical Dashboard
"""
import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

# Add path for inference engine
sys.path.append('../inference')

def main():
    st.set_page_config(
        page_title="👁️ Ophthalmology AI",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("👁️ Ophthalmology AI Clinical Assistant")
    st.markdown("### AI-Powered Retinal Analysis & Clinical Decision Support")
    
    # Sidebar - Patient Information
    st.sidebar.header("📋 Patient Information")
    
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=65)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    diabetes_duration = st.sidebar.number_input("Diabetes Duration (years)", min_value=0, max_value=50, value=10)
    insulin_use = st.sidebar.checkbox("Insulin Use")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Retinal Image Upload")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
            
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("🧠 AI Analysis in progress..."):
                    # Mock analysis
                    result = run_mock_analysis(age, sex, diabetes_duration, insulin_use)
                
                with col2:
                    display_results(result)
    
    with col2:
        if uploaded_file is None:
            st.header("📊 Analysis Results")
            st.info("👆 Please upload a retinal image to begin analysis")

def run_mock_analysis(age, sex, diabetes_duration, insulin_use):
    """Mock analysis function"""
    
    # Generate risk-based predictions
    base_risk = 0.1
    if age > 65: base_risk += 0.2
    if diabetes_duration > 10: base_risk += 0.3
    if insulin_use: base_risk += 0.2
    
    np.random.seed(42)
    predictions = {
        'diabetic_retinopathy': min(base_risk + np.random.random() * 0.3, 0.95),
        'macular_edema': min(base_risk * 0.7 + np.random.random() * 0.2, 0.8),
        'amd': min((age - 50) / 50 + np.random.random() * 0.2, 0.7),
        'hypertensive_retinopathy': min(base_risk * 0.5 + np.random.random() * 0.15, 0.6),
        'retinal_detachment': np.random.random() * 0.1,
    }
    
    max_pred = max(predictions.values())
    risk_level = "🚨 HIGH RISK" if max_pred > 0.8 else "⚡ MODERATE RISK" if max_pred > 0.5 else "✅ LOW RISK"
    
    # Generate recommendations
    recommendations = []
    for condition, prob in predictions.items():
        if prob > 0.7:
            recommendations.append(f"🚨 {condition.replace('_', ' ').title()}: Urgent referral recommended")
        elif prob > 0.5:
            recommendations.append(f"⚡ {condition.replace('_', ' ').title()}: Close monitoring advised")
    
    if not recommendations:
        recommendations.append("✅ No immediate concerns - continue routine care")
    
    return {
        'predictions': predictions,
        'risk_level': risk_level,
        'recommendations': recommendations,
        'max_confidence': max_pred
    }

def display_results(result):
    """Display analysis results"""
    st.header("📊 AI Analysis Results")
    
    # Risk level
    st.markdown(f"### {result['risk_level']}")
    
    # Predictions
    st.subheader("🔍 Condition Analysis")
    for condition, probability in result['predictions'].items():
        if probability > 0.3:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{condition.replace('_', ' ').title()}**")
                st.progress(probability)
            with col2:
                st.write(f"{probability:.1%}")
    
    # Recommendations
    st.subheader("💡 Clinical Recommendations")
    for i, rec in enumerate(result['recommendations'], 1):
        st.write(f"{i}. {rec}")
    
    # Disclaimer
    st.warning("⚠️ **Medical Disclaimer:** This AI assessment requires validation by a qualified ophthalmologist.")

if __name__ == "__main__":
    main()

print("✅ Streamlit Clinical Dashboard created")
