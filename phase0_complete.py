"""
PHASE 0 COMPLETE: Real BRSET Dataset Integration & Analysis
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
    """Setup real BRSET dataset from downloaded files"""
    print("📁 SETTING UP REAL BRSET DATASET")
    print("=" * 40)
    
    # Check for BRSET download in parent directory
    brset_path = Path("../BRSET")
    
    if not brset_path.exists():
        print("❌ BRSET directory not found at ../BRSET")
        print("Expected structure:")
        print("  ../BRSET/")
        print("    ├── fundus_photos/")
        print("    └── labels_brset.csv")
        return False
    
    print(f"✅ Found BRSET directory: {brset_path}")
    
    # Check required files
    labels_file = brset_path / "labels_brset.csv"
    images_dir = brset_path / "fundus_photos"
    
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        return False
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    # Create our data structure
    data_dir = Path("data/brset_real")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy labels file
    target_labels = data_dir / "labels.csv"
    shutil.copy2(labels_file, target_labels)
    print(f"✅ Copied labels: {target_labels}")
    
    # Create symlink to images (saves space)
    target_images = data_dir / "images"
    if target_images.exists():
        if target_images.is_symlink():
            target_images.unlink()
        else:
            shutil.rmtree(target_images)
    
    # Create symlink
    target_images.symlink_to(images_dir.absolute())
    print(f"✅ Linked images: {target_images}")
    
    return True

def analyze_dataset():
    """Analyze the real BRSET dataset"""
    print("\n📊 ANALYZING REAL BRSET DATASET")
    print("=" * 40)
    
    labels_path = Path("data/brset_real/labels.csv")
    images_path = Path("data/brset_real/images")
    
    if not labels_path.exists():
        print("❌ Labels file not found")
        return None, None
    
    # Load real dataset
    df = pd.read_csv(labels_path)
    print(f"📈 Total samples: {len(df):,}")
    print(f"👥 Unique patients: {df['patient_id'].nunique():,}")
    print(f"📋 Columns: {len(df.columns)}")
    
    # Count images
    if images_path.exists():
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg"))
        print(f"🖼️ Real images: {len(image_files):,}")
        
        if len(image_files) > 0:
            sample_img = Image.open(image_files[0])
            print(f"📸 Sample image size: {sample_img.size}")
    
    # Show first few columns to understand structure
    print(f"\n📋 Dataset structure (first 10 columns):")
    for i, col in enumerate(df.columns[:10]):
        print(f"  {col}")
    
    # Eye conditions analysis
    condition_cols = [
        'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
        'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
        'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
        'increased_cup_disc', 'other'
    ]
    
    print(f"\n🎯 EYE CONDITIONS (Real Distribution):")
    condition_stats = {}
    for condition in condition_cols:
        if condition in df.columns:
            count = df[condition].sum()
            percentage = (count / len(df)) * 100
            condition_stats[condition] = {'count': count, 'percentage': percentage}
            print(f"  {condition:<25}: {count:>6,} ({percentage:>5.1f}%)")
    
    # Demographics
    print(f"\n👥 PATIENT DEMOGRAPHICS:")
    if 'patient_age' in df.columns:
        print(f"  Age range: {df['patient_age'].min()}-{df['patient_age'].max()}")
        print(f"  Mean age: {df['patient_age'].mean():.1f} ± {df['patient_age'].std():.1f}")
    
    if 'patient_sex' in df.columns:
        sex_counts = df['patient_sex'].value_counts()
        print(f"  Sex distribution: {dict(sex_counts)}")
    
    # Diabetes info
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
    """Create visualizations of real dataset"""
    if condition_stats is None:
        return
    
    print(f"\n📊 CREATING VISUALIZATIONS...")
    
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)
    
    # 1. Condition distribution
    plt.figure(figsize=(15, 8))
    conditions = list(condition_stats.keys())
    counts = [condition_stats[c]['count'] for c in conditions]
    
    plt.bar(range(len(conditions)), counts, color='steelblue', alpha=0.8)
    plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')
    plt.title('Real BRSET Dataset: Eye Condition Distribution', fontsize=16, pad=20)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.xlabel('Eye Conditions', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'condition_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Age distribution
    if 'patient_age' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['patient_age'].dropna(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Real BRSET Dataset: Patient Age Distribution', fontsize=16)
        plt.xlabel('Age (years)', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to: {results_dir}")

def test_dataset_integrity():
    """Test dataset integrity and readiness for training"""
    print(f"\n🧪 TESTING DATASET INTEGRITY")
    print("=" * 35)
    
    test_results = {
        'labels_file': False,
        'images_linked': False,
        'sample_loadable': False,
        'conditions_valid': False,
        'demographics_valid': False
    }
    
    # Test 1: Labels file
    labels_path = Path("data/brset_real/labels.csv")
    if labels_path.exists():
        try:
            df = pd.read_csv(labels_path)
            if len(df) > 1000:  # Should have many samples
                test_results['labels_file'] = True
                print("✅ Labels file: PASS")
            else:
                print("❌ Labels file: Too few samples")
        except Exception as e:
            print(f"❌ Labels file: Error loading - {e}")
    else:
        print("❌ Labels file: Not found")
    
    # Test 2: Images linked
    images_path = Path("data/brset_real/images")
    if images_path.exists() and images_path.is_symlink():
        image_files = list(images_path.glob("*.jpg"))
        if len(image_files) > 1000:
            test_results['images_linked'] = True
            print("✅ Images linked: PASS")
        else:
            print("❌ Images linked: Too few images")
    else:
        print("❌ Images linked: Not properly linked")
    
    # Test 3: Sample image loadable
    if test_results['images_linked']:
        try:
            sample_image = list(images_path.glob("*.jpg"))[0]
            img = Image.open(sample_image)
            if img.size[0] > 100 and img.size[1] > 100:  # Reasonable size
                test_results['sample_loadable'] = True
                print("✅ Sample loading: PASS")
            else:
                print("❌ Sample loading: Image too small")
        except Exception as e:
            print(f"❌ Sample loading: Error - {e}")
    
    # Test 4: Eye conditions valid
    if test_results['labels_file']:
        condition_cols = [
            'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd'
        ]
        valid_conditions = sum(1 for col in condition_cols if col in df.columns)
        if valid_conditions >= 3:
            test_results['conditions_valid'] = True
            print("✅ Eye conditions: PASS")
        else:
            print("❌ Eye conditions: Missing key columns")
    
    # Test 5: Demographics valid
    if test_results['labels_file']:
        required_demo = ['patient_age', 'patient_sex', 'patient_id']
        valid_demo = sum(1 for col in required_demo if col in df.columns)
        if valid_demo >= 2:
            test_results['demographics_valid'] = True
            print("✅ Demographics: PASS")
        else:
            print("❌ Demographics: Missing key columns")
    
    # Overall test result
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n📊 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - READY FOR PHASE 1!")
        return True
    else:
        print("⚠️ Some tests failed - review issues above")
        return False

def create_phase0_summary():
    """Create Phase 0 completion summary"""
    labels_path = Path("data/brset_real/labels.csv")
    
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        
        summary = f"""# ✅ PHASE 0 COMPLETE: Real BRSET Dataset Setup

## 🎯 Dataset Successfully Integrated

### 📊 Real Dataset Overview
- **Total Samples**: {len(df):,}
- **Unique Patients**: {df['patient_id'].nunique():,}
- **Data Source**: PhysioNet BRSET v1.0.1
- **Images**: Real fundus photographs

### 👥 Patient Demographics  
- **Age Range**: {df['patient_age'].min()}-{df['patient_age'].max()} years
- **Mean Age**: {df['patient_age'].mean():.1f} ± {df['patient_age'].std():.1f} years

### 🎯 Key Eye Conditions Available
- Diabetic Retinopathy: {df.get('diabetic_retinopathy', pd.Series([0])).sum():,} cases
- Macular Edema: {df.get('macular_edema', pd.Series([0])).sum():,} cases  
- AMD: {df.get('amd', pd.Series([0])).sum():,} cases
- Retinal Detachment: {df.get('retinal_detachment', pd.Series([0])).sum():,} cases

### ✅ Phase 0 Achievements
- [x] Real BRSET dataset downloaded and verified
- [x] Data structure properly organized
- [x] Image directory linked (no duplication)
- [x] Dataset integrity tested and passed
- [x] Visualizations created
- [x] Ready for vision model training

### 🔄 Next Phase
**Phase 1**: Vision model training on real fundus images
- Multi-label classification for 13+ eye conditions
- ResNet/EfficientNet architectures
- Transfer learning from ImageNet

## 🚀 Status: READY FOR PHASE 1 ✅
"""
        
        with open("PHASE0_COMPLETE.md", "w") as f:
            f.write(summary)
        
        print(f"✅ Phase 0 summary saved: PHASE0_COMPLETE.md")

def main():
    """Main Phase 0 execution"""
    print("🚀 EXECUTING PHASE 0: REAL BRSET SETUP")
    
    # Step 1: Setup real dataset
    if not setup_real_brset():
        print("❌ Phase 0 FAILED: Could not setup dataset")
        return False
    
    # Step 2: Analyze dataset
    df, condition_stats = analyze_dataset()
    if df is None:
        print("❌ Phase 0 FAILED: Could not analyze dataset")
        return False
    
    # Step 3: Create visualizations
    create_visualizations(df, condition_stats)
    
    # Step 4: Test dataset integrity
    if not test_dataset_integrity():
        print("❌ Phase 0 FAILED: Dataset integrity tests failed")
        return False
    
    # Step 5: Create summary
    create_phase0_summary()
    
    print(f"\n🎉 PHASE 0 COMPLETED SUCCESSFULLY!")
    print(f"✅ Real BRSET dataset ready: {len(df):,} samples")
    print(f"✅ All integrity tests passed")
    print(f"🔄 Ready for Phase 1: Vision model training")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ PHASE 0 INCOMPLETE - Review errors above")
        exit(1)
