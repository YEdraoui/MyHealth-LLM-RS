"""
BRSET Dataset Exploratory Data Analysis
Run this after downloading the BRSET dataset
"""

import sys
sys.path.append('..')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_brset_data, analyze_dataset_distribution, create_eda_plots

def main():
    print("🔍 BRSET Dataset Exploratory Analysis")
    print("=" * 50)
    
    # Load dataset
    df = load_brset_data()
    
    if df is not None:
        # Basic analysis
        analyze_dataset_distribution(df)
        
        # Create visualization
        create_eda_plots(df)
        
        # Save processed summary
        summary = {
            'total_samples': len(df),
            'conditions_present': [col for col in df.columns if col in BRSET_LABELS],
            'priority_condition_counts': {
                condition: int(df[condition].sum()) if condition in df.columns else 0
                for condition in PRIORITY_CONDITIONS
            }
        }
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Results saved to: data/results/")
        
    else:
        print("\n📥 Next steps:")
        print("1. Download BRSET dataset")
        print("2. Extract to: data/raw/brset/")
        print("3. Run this analysis again")

if __name__ == "__main__":
    main()
