"""
PHASE 1 DEMO: Ophthalmology AI System (Working Version)
Fixed all errors - ready for demonstration
"""
import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

st.set_page_config(
    page_title="👁️ Ophthalmology AI Demo",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("👁️ Ophthalmology AI System - Phase 1 Demo")
st.markdown("### Real BRSET Dataset Analysis & Vision Model Demo")
st.markdown("**🎯 Using Real Clinical Data - No Synthetic Data!**")

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
    st.sidebar.error("❌ Dataset not found")
    dataset_available = False

# Check model status
model_path = Path("models/phase1_best_model.pth")
stats_path = Path("models/training_stats.json")

if model_path.exists():
    st.sidebar.success("✅ Vision model trained")
    model_available = True
else:
    st.sidebar.warning("⚠️ Model training in progress")
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
        
        # Simple bar chart (working version)
        st.bar_chart(condition_df.set_index('Condition')['Cases'])
        
        # Show detailed statistics
        st.subheader("📋 Detailed Eye Condition Statistics")
        for _, row in condition_df.iterrows():
            st.write(f"**{row['Condition']}**: {row['Cases']:,} cases ({row['Percentage']:.1f}%)")
        
        # Patient demographics - simplified
        st.subheader("👥 Patient Demographics")
        
        # Age statistics (no problematic charts)
        st.write("**Age Statistics:**")
        st.write(f"- **Youngest Patient**: {df['patient_age'].min():.0f} years")
        st.write(f"- **Oldest Patient**: {df['patient_age'].max():.0f} years")
        st.write(f"- **Average Age**: {df['patient_age'].mean():.1f} ± {df['patient_age'].std():.1f} years")
        st.write(f"- **Median Age**: {df['patient_age'].median():.0f} years")
        
        # Age groups
        age_groups = pd.cut(df['patient_age'], 
                           bins=[0, 18, 30, 50, 65, 100], 
                           labels=['<18', '18-30', '30-50', '50-65', '65+'])
        age_group_counts = age_groups.value_counts().sort_index()
        
        st.write("**Age Distribution by Groups:**")
        for group, count in age_group_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"- **{group} years**: {count:,} patients ({percentage:.1f}%)")
        
        # Sex distribution
        if 'patient_sex' in df.columns:
            sex_counts = df['patient_sex'].value_counts()
            st.write("**Sex Distribution:**")
            total = len(df)
            for sex_code, count in sex_counts.items():
                sex_label = 'Female' if sex_code == 2 else 'Male'
                percentage = (count / total) * 100
                st.write(f"- **{sex_label}**: {count:,} patients ({percentage:.1f}%)")
        
        # Dataset quality info
        st.subheader("📸 Dataset Quality")
        if 'quality' in df.columns:
            quality_counts = df['quality'].value_counts()
            st.write("**Image Quality Distribution:**")
            for quality, count in quality_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"- **{quality}**: {count:,} images ({percentage:.1f}%)")
        
    else:
        st.error("❌ Dataset not available. Please run Phase 0 setup first.")

with col2:
    st.header("🧠 Model Training Status")
    
    if training_stats:
        st.success("✅ Model Training Completed!")
        
        # Training metrics (simple display)
        st.subheader("📊 Training Results")
        final_stats = training_stats[-1]
        
        st.metric("Final AUROC", f"{final_stats['auroc']:.4f}")
        st.metric("Final Val Loss", f"{final_stats['val_loss']:.4f}")
        st.metric("Epochs Trained", len(training_stats))
        
        # Show training progress
        st.subheader("📈 Training Progress")
        for i, stats in enumerate(training_stats):
            st.write(f"**Epoch {stats['epoch']}:**")
            st.write(f"- Train Loss: {stats['train_loss']:.4f}")
            st.write(f"- Val Loss: {stats['val_loss']:.4f}")
            st.write(f"- AUROC: {stats['auroc']:.4f}")
            st.write("")
        
    elif model_available:
        st.info("✅ Model trained but stats not available")
    else:
        st.warning("⚠️ Model training required")
        
        st.write("**To train the model:**")
        st.code("python train_phase1_simple.py")
        
        if st.button("🚀 Check Training Status"):
            st.rerun()

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

# Key achievements
st.header("🎉 Key Achievements")

if dataset_available:
    st.success("✅ **Phase 0 Complete**: Real BRSET dataset integrated")
    st.info(f"📊 **{len(df):,} real fundus images** from {df['patient_id'].nunique():,} patients")
    st.info("🏥 **Real clinical data** - no synthetic data used")

if model_available:
    st.success("✅ **Phase 1 Complete**: Vision model trained")
    st.info("🧠 **ResNet50-based classifier** for 13+ eye conditions")

# Next steps
st.header("🚀 Next Steps")

if dataset_available and model_available:
    st.success("✅ **Ready for Phase 2**: Multimodal Fusion")
    st.markdown("- Combine vision features with patient clinical data")
    st.markdown("- Create text encoder for patient metadata")  
    st.markdown("- Build fusion architecture for enhanced predictions")
elif dataset_available:
    st.info("⚠️ **Complete Vision Model Training**")
    st.markdown("Run: `python train_phase1_simple.py`")
else:
    st.error("❌ **Complete Phase 0 Setup First**")

# Footer
st.markdown("---")
st.markdown("**🏥 MyHealth-LLM-RS: Ophthalmology AI System**")
if dataset_available:
    st.markdown(f"✅ **Real BRSET Dataset**: {df['patient_id'].nunique():,} patients • {len(df):,} fundus images")
    st.markdown("🎯 **13+ Eye Conditions** • Vision + LLM • Clinical decision support")
else:
    st.markdown("🔄 **Setup in Progress**")
