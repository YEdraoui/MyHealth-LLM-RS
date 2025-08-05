"""
PHASE 1 DEMO: Ophthalmology AI System
Real-time demo of vision model on BRSET dataset
"""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import os

# Add current directory to path
sys.path.append('.')

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
        
        # Bar chart
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
        
        # Age distribution
        st.subheader("👥 Patient Demographics")
        
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
        
        # Final metrics
        st.subheader("📊 Final Results")
        final_stats = training_stats[-1]
        st.metric("Final AUROC", f"{final_stats['auroc']:.4f}")
        st.metric("Final Val Loss", f"{final_stats['val_loss']:.4f}")
        
    elif model_available:
        st.info("✅ Model trained but stats not available")
    else:
        st.warning("⚠️ Model training required")
        if st.button("🚀 Start Training"):
            st.info("Run: python train_phase1_real.py")

# Phase status
st.header("🔄 Project Status")

phases = [
    {"Phase": "Phase 0", "Task": "Dataset Setup", "Status": "✅ Complete" if dataset_available else "❌ Incomplete"},
    {"Phase": "Phase 1", "Task": "Vision Model", "Status": "✅ Complete" if model_available else "⚠️ In Progress"},
    {"Phase": "Phase 2", "Task": "Multimodal Fusion", "Status": "⏳ Pending"},
    {"Phase": "Phase 3", "Task": "Explainability", "Status": "⏳ Pending"},
    {"Phase": "Phase 4", "Task": "Deployment", "Status": "⏳ Pending"}
]

phase_df = pd.DataFrame(phases)
st.table(phase_df)

# Next steps
st.header("🚀 Next Steps")
if dataset_available and model_available:
    st.success("✅ Ready for Phase 2: Multimodal Fusion")
    st.markdown("- Combine vision features with patient clinical data")
    st.markdown("- Create text encoder for patient metadata")
    st.markdown("- Build fusion architecture")
elif dataset_available:
    st.info("⚠️ Run model training: `python train_phase1_real.py`")
else:
    st.error("❌ Complete Phase 0 setup first")

# Footer
st.markdown("---")
st.markdown("**🏥 MyHealth-LLM-RS: Ophthalmology AI System**")
st.markdown("Real BRSET dataset • Vision + LLM • Clinical decision support")
