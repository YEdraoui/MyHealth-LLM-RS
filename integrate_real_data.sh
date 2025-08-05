#!/bin/bash
echo "🔄 INTEGRATING REAL BRSET DATASET"
echo "=================================="

# Check if real data exists
if [ ! -f "data/brset_real/labels.csv" ]; then
    echo "❌ Real BRSET data not found!"
    echo "📁 Expected: data/brset_real/labels.csv"
    echo "📁 Expected: data/brset_real/images/"
    echo ""
    echo "🔗 Download from: https://physionet.org/content/brazilian-ophthalmological/1.0.0/"
    exit 1
fi

echo "✅ Real BRSET dataset found!"

# Analyze real dataset
python << 'PYTHON_EOF'
import pandas as pd
import os
from pathlib import Path

print("📊 ANALYZING REAL BRSET DATASET")
print("=" * 40)

# Load real labels
df = pd.read_csv('data/brset_real/labels.csv')

print(f"📈 Total samples: {len(df)}")
print(f"📋 Columns: {list(df.columns)}")
print(f"👥 Patients: {df['patient_id'].nunique()}")

# Check image directory
images_dir = Path('data/brset_real/images')
if images_dir.exists():
    image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.jpeg')))
    print(f"🖼️  Images found: {image_count}")
else:
    print("❌ Images directory not found!")

# Analyze conditions
condition_cols = [
    'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
    'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
    'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
    'increased_cup_disc', 'other'
]

print(f"\n🎯 REAL CONDITION DISTRIBUTION:")
for condition in condition_cols:
    if condition in df.columns:
        count = df[condition].sum()
        percentage = (count / len(df)) * 100
        print(f"  {condition}: {count} ({percentage:.1f}%)")

# Demographics
print(f"\n👥 REAL DEMOGRAPHICS:")
print(f"  Age range: {df['patient_age'].min()}-{df['patient_age'].max()}")
print(f"  Mean age: {df['patient_age'].mean():.1f}")
if 'patient_sex' in df.columns:
    sex_dist = df['patient_sex'].value_counts()
    print(f"  Sex distribution: {dict(sex_dist)}")

print(f"\n✅ REAL DATASET ANALYSIS COMPLETE")
PYTHON_EOF

echo ""
echo "🔄 UPDATING PROJECT TO USE REAL DATA..."

# Update data paths in all files
find . -name "*.py" -exec sed -i.bak 's/data\/labels\.csv/data\/brset_real\/labels.csv/g' {} \;
find . -name "*.py" -exec sed -i.bak 's/data\/images/data\/brset_real\/images/g' {} \;

# Remove backup files
find . -name "*.bak" -delete

echo "✅ Project updated to use real BRSET data"
echo ""
echo "🚀 READY TO TRAIN WITH REAL DATA:"
echo "  python train_real_brset.py"
