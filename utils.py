"""
Utility functions for Ophthalmology LLM
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

def load_brset_data():
    """Load BRSET dataset labels"""
    labels_path = RAW_DATA_DIR / "labels.csv"
    if labels_path.exists():
        return pd.read_csv(labels_path)
    else:
        print(f"❌ Labels file not found at {labels_path}")
        print("Please download BRSET dataset first!")
        return None

def analyze_dataset_distribution(df):
    """Analyze class distribution in BRSET dataset"""
    if df is None:
        return
    
    print("📊 BRSET Dataset Analysis")
    print(f"Total samples: {len(df)}")
    print(f"Image columns: {df.columns.tolist()}")
    
    # Class distribution for pathology labels
    pathology_cols = [col for col in BRSET_LABELS if col in df.columns]
    
    if pathology_cols:
        print(f"\n�� Pathology Distribution (Top 5 Priority):")
        for condition in PRIORITY_CONDITIONS:
            if condition in df.columns:
                count = df[condition].sum()
                percentage = (count / len(df)) * 100
                print(f"  {condition}: {count} ({percentage:.1f}%)")
    
    # Age/Sex distribution if available
    if 'age' in df.columns:
        print(f"\n👥 Demographics:")
        print(f"  Age range: {df['age'].min()}-{df['age'].max()}")
        print(f"  Mean age: {df['age'].mean():.1f}")
    
    if 'sex' in df.columns:
        print(f"  Sex distribution:")
        print(df['sex'].value_counts().to_string())

def create_eda_plots(df, save_path="data/results/"):
    """Create exploratory data analysis plots"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if df is None:
        return
    
    # Class distribution plot
    pathology_cols = [col for col in PRIORITY_CONDITIONS if col in df.columns]
    if pathology_cols:
        plt.figure(figsize=(12, 6))
        counts = [df[col].sum() for col in pathology_cols]
        plt.bar(range(len(pathology_cols)), counts)
        plt.xticks(range(len(pathology_cols)), pathology_cols, rotation=45)
        plt.title('Distribution of Top 5 Eye Conditions (BRSET)')
        plt.ylabel('Number of Cases')
        plt.tight_layout()
        plt.savefig(f"{save_path}/condition_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Quick dataset check
    df = load_brset_data()
    analyze_dataset_distribution(df)
