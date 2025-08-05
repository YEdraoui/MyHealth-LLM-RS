"""
PHASE 1 DEMO: Ophthalmology AI System (Fixed Version)
Works with available packages
"""
import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Try to import plotly, use fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available - using basic charts")

st.set_page_config(
    page_title="👁️ Ophthalmology AI Demo",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("👁️ Ophthalmology AI System - Phase 1 Demo")
st.markdown("### Real BRSET Dataset Analysis & Vision Model Demo")

# Sidebar
st.sidebar.header("📊 System Status")

# Check dataset status
labels_path = Path("data/brset_real/labels.csv")
images_path = Path("data/brset_real/images")

if labels_path.exists():
    df = pd.read_csv(labels_path)
    st.sidebar.success(f"✅ Dataset: {len(df):,} samples loaded")
    dataset_available = True
else:
    st.sidebar.error("❌ Dataset not found - run Phase 0 setup")
    dataset_available = False

# Check model status
model_path = Path("models/phase1_best_model.pth")
stats_path = Path("models/training_stats.json")

if model_path.exists():
    st.sidebar.success("✅ Vision model trained")
    model_available = True
else:
    st.sidebar.warning("⚠️ Model not trained yet")
    model_available = False

if stats_path.exists():
    with open(stats_path, 'r') as f:
        training_stats = json.load(f)
    st.sidebar.success("✅ Training stats available")
else:
    training_stats = None

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Real BRSET Dataset Overview")
    
    if dataset_available:
        # Dataset statistics
        st.subheader("📈 Dataset Statistics")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Samples", f"{len(df):,}")
        with col_b:
            st.metric("Unique Patients", f"{df['patient_id'].nunique():,}")
        with col_c:
            st.metric("Age Range", f"{df['patient_age'].min():.0f}-{df['patient_age'].max():.0f}")
        with col_d:
            st.metric("Mean Age", f"{df['patient_age'].mean():.1f} years")
        
        # Eye conditions distribution
        st.subheader("🎯 Eye Conditions Distribution")
        
        condition_cols = [
            'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
            'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
            'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
            'increased_cup_disc', 'other'
        ]
        
        condition_data = []
        for condition in condition_cols:
            if condition in df.columns:
                count = df[condition].sum()
                percentage = (count / len(df)) * 100
                condition_data.append({
                    'Condition': condition.replace('_', ' ').title(),
                    'Cases': count,
                    'Percentage': percentage
                })
        
        condition_df = pd.DataFrame(condition_data)
        
        if PLOTLY_AVAILABLE:
            # Interactive bar chart
            fig = px.bar(
                condition_df, 
                x='Condition', 
                y='Cases',
                title='Eye Condition Distribution in Real BRSET Dataset',
                color='Percentage',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to bar chart
            st.bar_chart(condition_df.set_index('Condition')['Cases'])
        
        # Show top conditions
        st.subheader("📋 Top Eye Conditions")
        top_conditions = condition_df.nlargest(5, 'Cases')
        for _, row in top_conditions.iterrows():
            st.write(f"**{row['Condition']}**: {row['Cases']:,} cases ({row['Percentage']:.1f}%)")
        
        # Patient demographics
        st.subheader("👥 Patient Demographics")
        
        if PLOTLY_AVAILABLE:
            fig_age = px.histogram(
                df, 
                x='patient_age',
                nbins=30,
                title='Patient Age Distribution',
                labels={'patient_age': 'Age (years)', 'count': 'Number of Patients'}
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            # Fallback histogram
            st.write("**Age Distribution:**")
            age_counts = pd.cut(df['patient_age'], bins=20).value_counts().sort_index()
            st.bar_chart(age_counts)
        
    else:
        st.error("❌ Dataset not available. Please run Phase 0 setup first.")

with col2:
    st.header("🧠 Model Training Status")
    
    if training_stats:
        st.success("✅ Model Training Completed!")
        
        # Training metrics
        epochs = [stat['epoch'] for stat in training_stats]
        train_losses = [stat['train_loss'] for stat in training_stats]
        val_losses = [stat['val_loss'] for stat in training_stats]
        aurocs = [stat['auroc'] for stat in training_stats]
        
        if PLOTLY_AVAILABLE:
            # Loss curves
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=train_losses, name='Train Loss', line=dict(color='blue')))
            fig_loss.add_trace(go.Scatter(x=epochs, y=val_losses, name='Val Loss', line=dict(color='red')))
            fig_loss.update_layout(title='Training Progress', xaxis_title='Epoch', yaxis_title='Loss', height=300)
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # AUROC
            fig_auroc = go.Figure()
            fig_auroc.add_trace(go.Scatter(x=epochs, y=aurocs, name='AUROC', line=dict(color='green')))
            fig_auroc.update_layout(title='Model Performance', xaxis_title='Epoch', yaxis_title='AUROC', height=300)
            st.plotly_chart(fig_auroc, use_container_width=True)
        else:
            # Fallback charts
            st.write("**Training Progress:**")
            progress_df = pd.DataFrame({
                'Epoch': epochs,
                'Train Loss': train_losses,
                'Val Loss': val_losses,
                'AUROC': aurocs
            })
            st.line_chart(progress_df.set_index('Epoch')[['Train Loss', 'Val Loss']])
            st.line_chart(progress_df.set_index('Epoch')[['AUROC']])
        
        # Final metrics
        st.subheader("📊 Final Results")
        final_stats = training_stats[-1]
        st.metric("Final AUROC", f"{final_stats['auroc']:.4f}")
        st.metric("Final Val Loss", f"{final_stats['val_loss']:.4f}")
        st.metric("Epochs Trained", len(training_stats))
        
    elif model_available:
        st.info("✅ Model trained but stats not available")
    else:
        st.warning("⚠️ Model training required")
        if st.button("🚀 Start Training"):
            st.info("Run: python train_phase1_simple.py")

# Phase status
st.header("🔄 Project Status")

phase_data = {
    "Phase": ["Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4"],
    "Task": ["Dataset Setup", "Vision Model", "Multimodal Fusion", "Explainability", "Deployment"],
    "Status": [
        "✅ Complete" if dataset_available else "❌ Incomplete",
        "✅ Complete" if model_available else "⚠️ In Progress",
        "⏳ Pending",
        "⏳ Pending", 
        "⏳ Pending"
    ]
}

phase_df = pd.DataFrame(phase_data)
st.table(phase_df)

# Instructions
st.header("🚀 Instructions")

if not dataset_available:
    st.error("❌ **Step 1**: Complete Phase 0 dataset setup")
elif not model_available:
    st.warning("⚠️ **Step 2**: Train the vision model")
    st.code("python train_phase1_simple.py")
else:
    st.success("✅ **Phase 1 Complete!** Ready for Phase 2")

# Footer
st.markdown("---")
st.markdown("**🏥 MyHealth-LLM-RS: Ophthalmology AI System**")
st.markdown("Real BRSET dataset • Vision + LLM • Clinical decision support")
st.markdown(f"**Dataset**: {df['patient_id'].nunique():,} patients, {len(df):,} images" if dataset_available else "Dataset not loaded")
