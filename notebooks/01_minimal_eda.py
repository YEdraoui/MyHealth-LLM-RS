"""
Minimal BRSET Dataset Analysis (works without full package install)
"""
import sys
sys.path.append('..')

import csv
from pathlib import Path

def load_csv_simple(filepath):
    """Load CSV without pandas"""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def analyze_minimal():
    """Basic analysis without advanced packages"""
    print("🔍 Minimal BRSET Dataset Analysis")
    print("=" * 50)
    
    csv_path = Path('data/raw/brset/labels.csv')
    
    if not csv_path.exists():
        print("❌ Dataset not found!")
        return
    
    # Load data
    data = load_csv_simple(csv_path)
    print(f"📊 Total samples: {len(data)}")
    
    if data:
        print(f"📋 Columns: {list(data[0].keys())}")
        
        # Count conditions
        conditions = ['diabetic_retinopathy', 'age_related_macular_degeneration', 
                     'media_haze', 'laser_scar', 'epiretinal_membrane']
        
        print(f"\n🎯 Priority Conditions:")
        for condition in conditions:
            if condition in data[0]:
                count = sum(1 for row in data if row[condition] == '1')
                percentage = (count / len(data)) * 100
                print(f"  {condition}: {count}/{len(data)} ({percentage:.1f}%)")
        
        # Demographics
        if 'age' in data[0]:
            ages = [int(row['age']) for row in data]
            print(f"\n👥 Demographics:")
            print(f"  Age range: {min(ages)}-{max(ages)}")
            print(f"  Mean age: {sum(ages)/len(ages):.1f}")
        
        if 'sex' in data[0]:
            sex_counts = {}
            for row in data:
                sex = row['sex']
                sex_counts[sex] = sex_counts.get(sex, 0) + 1
            print(f"  Sex distribution: {sex_counts}")
    
    print(f"\n✅ Basic analysis complete!")
    print(f"🔄 Ready for Phase 1: Vision Model Development")

if __name__ == "__main__":
    analyze_minimal()
