"""
PHASE 0: REAL BRSET Dataset Setup & Analysis
NO synthetic data - only real clinical data from PhysioNet
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def setup_real_brset():
    """Set up real BRSET dataset from PhysioNet download"""
    print("🚀 PHASE 0: REAL BRSET DATASET SETUP")
    print("=" * 50)
    
    # Path to downloaded BRSET data
    brset_download = Path("../BRSET")
    
    if not brset_download.exists():
        print("❌ BRSET download not found at ../BRSET")
        return False
    
    print(f"✅ Found BRSET download at: {brset_download}")
    
    # Check required files
    labels_file = brset_download / "labels_brset.csv"
    images_dir = brset_download / "fundus_photos"
    
    if not labels_file.exists():
        print(f"❌ Labels file missing: {labels_file}")
        return False
    
    if not images_dir.exists():
        print(f"❌ Images directory missing: {images_dir}")
        return False
    
    # Create our data structure
    data_dir = Path("data/brset_real")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 Setting up real data structure...")
    
    # Copy labels file
    target_labels = data_dir / "labels.csv"
    shutil.copy2(labels_file, target_labels)
    print(f"✅ Copied labels: {target_labels}")
    
    # Create symlink to images (saves disk space)
    target_images = data_dir / "images"
    if target_images.exists():
        if target_images.is_symlink():
            target_images.unlink()
        else:
            shutil.rmtree(target_images)
    
    target_images.symlink_to(images_dir.absolute())
    print(f"✅ Linked images: {target_images} -> {images_dir}")
    
    return True

def analyze_real_dataset():
    """Analyze the real BRSET dataset"""
    print("\n📊 ANALYZING REAL BRSET DATASET")
    print("=" * 40)
    
    labels_path = Path("data/brset_real/labels.csv")
    images_path = Path("data/brset_real/images")
    
    if not labels_path.exists():
        print("❌ Labels file not found - run setup first")
        return None
    
    # Load real labels
    df = pd.read_csv(labels_path)
    print(f"📈 Total samples: {len(df):,}")
    print(f"👥 Unique patients: {df['patient_id'].nunique():,}")
    print(f"📋 Total columns: {len(df.columns)}")
    
    # Show column names
    print(f"\n📋 Dataset columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Count real images
    if images_path.exists():
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg"))
        print(f"\n🖼️ Real images found: {len(image_files):,}")
        
        if len(image_files) > 0:
            # Show sample image info
            sample_img = Image.open(image_files[0])
            print(f"📸 Sample image size: {sample_img.size}")
            print(f"📸 Image format: {sample_img.format}")
    
    # Analyze eye conditions (real distribution)
    condition_cols = [
        'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
        'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
        'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
        'increased_cup_disc', 'other'
    ]
    
    print(f"\n🎯 REAL CONDITION DISTRIBUTION:")
    condition_stats = {}
    for condition in condition_cols:
        if condition in df.columns:
            count = df[condition].sum()
            percentage = (count / len(df)) * 100
            condition_stats[condition] = {'count': count, 'percentage': percentage}
            print(f"  {condition:<25}: {count:>6,} ({percentage:>5.1f}%)")
    
    # Real patient demographics
    print(f"\n👥 REAL PATIENT DEMOGRAPHICS:")
    if 'patient_age' in df.columns:
        print(f"  Age range: {df['patient_age'].min()}-{df['patient_age'].max()}")
        print(f"  Mean age: {df['patient_age'].mean():.1f} ± {df['patient_age'].std():.1f} years")
    
    if 'patient_sex' in df.columns:
        sex_counts = df['patient_sex'].value_counts()
        print(f"  Sex distribution:")
        for sex, count in sex_counts.items():
            print(f"    Sex {sex}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Camera types
    if 'camera' in df.columns:
        camera_counts = df['camera'].value_counts()
        print(f"\n📷 CAMERA TYPES:")
        for camera, count in camera_counts.items():
            print(f"  {camera}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Image quality
    if 'quality' in df.columns:
        quality_counts = df['quality'].value_counts()
        print(f"\n📸 IMAGE QUALITY:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Diabetes-related info
    if 'diabetes' in df.columns:
        diabetes_counts = df['diabetes'].value_counts()
        print(f"\n🩺 DIABETES STATUS:")
        for status, count in diabetes_counts.items():
            print(f"  {status}: {count:,} ({count/len(df)*100:.1f}%)")
    
    if 'diabetes_time_y' in df.columns:
        diabetes_time = df['diabetes_time_y'].dropna()
        if len(diabetes_time) > 0:
            print(f"  Diabetes duration: {diabetes_time.mean():.1f} ± {diabetes_time.std():.1f} years")
    
    return df, condition_stats

def create_visualizations(df, condition_stats):
    """Create visualizations of the real dataset"""
    print(f"\n📊 CREATING VISUALIZATIONS...")
    
    # Create results directory
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)
    
    # 1. Condition distribution plot
    plt.figure(figsize=(15, 8))
    conditions = list(condition_stats.keys())
    counts = [condition_stats[c]['count'] for c in conditions]
    
    plt.bar(range(len(conditions)), counts)
    plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')
    plt.title('Real BRSET Dataset: Eye Condition Distribution')
    plt.ylabel('Number of Cases')
    plt.tight_layout()
    plt.savefig(results_dir / 'condition_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Age distribution
    if 'patient_age' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['patient_age'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        plt.title('Real BRSET Dataset: Patient Age Distribution')
        plt.xlabel('Age (years)')
        plt.ylabel('Number of Patients')
        plt.grid(True, alpha=0.3)
        plt.savefig(results_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Diabetes duration vs conditions
    if 'diabetes_time_y' in df.columns and 'diabetic_retinopathy' in df.columns:
        plt.figure(figsize=(10, 6))
        dr_yes = df[df['diabetic_retinopathy'] == 1]['diabetes_time_y'].dropna()
        dr_no = df[df['diabetic_retinopathy'] == 0]['diabetes_time_y'].dropna()
        
        plt.hist(dr_no, bins=20, alpha=0.7, label='No DR', color='blue')
        plt.hist(dr_yes, bins=20, alpha=0.7, label='DR Present', color='red')
        plt.title('Diabetes Duration vs Diabetic Retinopathy')
        plt.xlabel('Diabetes Duration (years)')
        plt.ylabel('Number of Patients')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(results_dir / 'diabetes_duration_vs_dr.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to: {results_dir}")

def create_phase0_summary(df, condition_stats):
    """Create Phase 0 completion summary"""
    summary = f"""# 📊 PHASE 0 COMPLETE: REAL BRSET DATASET SETUP

## ✅ Dataset Successfully Integrated

### 📈 Dataset Overview
- **Total Samples**: {len(df):,}
- **Unique Patients**: {df['patient_id'].nunique():,}
- **Real Fundus Images**: {len(list(Path('data/brset_real/images').glob('*.jpg'))):,}
- **Data Source**: PhysioNet BRSET v1.0.1

### 🎯 Eye Conditions (Real Distribution)
"""
    
    for condition, stats in condition_stats.items():
        summary += f"- **{condition.replace('_', ' ').title()}**: {stats['count']:,} cases ({stats['percentage']:.1f}%)\n"
    
    summary += f"""
### 👥 Patient Demographics
- **Age Range**: {df['patient_age'].min()}-{df['patient_age'].max()} years
- **Mean Age**: {df['patient_age'].mean():.1f} ± {df['patient_age'].std():.1f} years
- **Sex Distribution**: {dict(df['patient_sex'].value_counts())}

### 📷 Technical Details
- **Cameras Used**: {list(df['camera'].value_counts().keys())}
- **Image Quality**: {dict(df['quality'].value_counts())}
- **File Format**: JPEG fundus photographs

### 🔄 Next Steps
- ✅ Phase 0: Dataset setup complete
- 🎯 Phase 1: Vision model training on real data
- 🧠 Phase 2: Multimodal fusion with clinical context

## 🚀 Ready for Real Training!
"""
    
    with open("PHASE0_REAL_COMPLETE.md", "w") as f:
        f.write(summary)
    
    print(f"✅ Summary saved to: PHASE0_REAL_COMPLETE.md")

def main():
    """Main Phase 0 setup function"""
    print("🚀 STARTING PHASE 0: REAL BRSET SETUP")
    print("🚫 NO SYNTHETIC DATA - ONLY REAL CLINICAL DATA")
    print("=" * 60)
    
    # Step 1: Setup real data structure
    if not setup_real_brset():
        print("❌ Failed to set up BRSET data")
        return False
    
    # Step 2: Analyze real dataset
    df, condition_stats = analyze_real_dataset()
    if df is None:
        print("❌ Failed to analyze dataset")
        return False
    
    # Step 3: Create visualizations
    create_visualizations(df, condition_stats)
    
    # Step 4: Create summary
    create_phase0_summary(df, condition_stats)
    
    print(f"\n🎉 PHASE 0 COMPLETED SUCCESSFULLY!")
    print(f"✅ Real BRSET dataset ready for training")
    print(f"📊 {len(df):,} real samples with {df['patient_id'].nunique():,} patients")
    print(f"🔄 Ready for Phase 1: Vision model training")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 EXECUTE NEXT:")
        print("git add . && git commit -m '✅ PHASE 0: Real BRSET Dataset Setup Complete'")
        print("git push origin phase4-inference-ui")
