"""
BRSET Dataset Loader (based on official implementation)
"""
import pandas as pd
import os
from pathlib import Path

def get_dataset(dataset_path='data/', download=False):
    """Load BRSET dataset (official structure)"""
    
    if download:
        print("📥 Download BRSET from PhysioNet: https://physionet.org/content/brazilian-ophthalmological/1.0.0/")
        print("⚠️  Requires credentialed access")
        return None, None
    
    # Check for existing data
    labels_path = os.path.join(dataset_path, 'labels.csv')
    images_path = os.path.join(dataset_path, 'images')
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        print("📁 Expected structure:")
        print("  data/")
        print("    ├── labels.csv")
        print("    └── images/")
        print("        ├── image_1.jpg")
        print("        └── ...")
        
        # Create mock structure for development
        os.makedirs(images_path, exist_ok=True)
        create_mock_labels_csv(labels_path)
        print("✅ Created mock structure for development")
    
    # Load labels
    df = pd.read_csv(labels_path)
    return df, images_path

def create_mock_labels_csv(labels_path):
    """Create mock labels.csv matching BRSET structure"""
    import numpy as np
    np.random.seed(42)
    
    # BRSET actual columns (based on PhysioNet documentation)
    mock_data = {
        'image_id': [f'image_{i:05d}.jpg' for i in range(1, 501)],  # 500 mock images
        'patient_id': [f'patient_{i:04d}' for i in range(1, 501)],
        'camera': np.random.choice(['Canon CR', 'NIKON NF5050'], 500),
        'patient_age': np.random.randint(25, 85, 500),
        'patient_sex': np.random.choice([1, 2], 500),  # 1=male, 2=female
        'exam_eye': np.random.choice([1, 2], 500),     # 1=right, 2=left
        'diabetes': np.random.choice([0, 1], 500, p=[0.8, 0.2]),
        'diabetes_time': np.random.randint(0, 25, 500),
        'insulin_use': np.random.choice([0, 1], 500, p=[0.6, 0.4]),
        
        # Anatomical parameters
        'optic_disc': np.random.choice([1, 2], 500, p=[0.85, 0.15]),
        'vessels': np.random.choice([1, 2], 500, p=[0.8, 0.2]),
        'macula': np.random.choice([1, 2], 500, p=[0.9, 0.1]),
        
        # Quality parameters
        'focus': np.random.choice([1, 2], 500, p=[0.9, 0.1]),
        'illumination': np.random.choice([1, 2], 500, p=[0.85, 0.15]),
        'image_field': np.random.choice([1, 2], 500, p=[0.95, 0.05]),
        'artifacts': np.random.choice([1, 2], 500, p=[0.8, 0.2]),
        
        # Disease classifications (13 conditions)
        'diabetic_retinopathy': np.random.choice([0, 1], 500, p=[0.7, 0.3]),
        'macular_edema': np.random.choice([0, 1], 500, p=[0.9, 0.1]),
        'scar': np.random.choice([0, 1], 500, p=[0.95, 0.05]),
        'nevus': np.random.choice([0, 1], 500, p=[0.98, 0.02]),
        'amd': np.random.choice([0, 1], 500, p=[0.85, 0.15]),
        'vascular_occlusion': np.random.choice([0, 1], 500, p=[0.95, 0.05]),
        'hypertensive_retinopathy': np.random.choice([0, 1], 500, p=[0.8, 0.2]),
        'drusens': np.random.choice([0, 1], 500, p=[0.9, 0.1]),
        'hemorrhage': np.random.choice([0, 1], 500, p=[0.92, 0.08]),
        'retinal_detachment': np.random.choice([0, 1], 500, p=[0.98, 0.02]),
        'myopic_fundus': np.random.choice([0, 1], 500, p=[0.85, 0.15]),
        'increased_cup_disc': np.random.choice([0, 1], 500, p=[0.9, 0.1]),
        'other': np.random.choice([0, 1], 500, p=[0.95, 0.05]),
        
        # Diabetic retinopathy grading (ICDR)
        'DR_ICDR': np.random.choice([0, 1, 2, 3, 4], 500, p=[0.7, 0.15, 0.1, 0.03, 0.02]),
        'DR_SDRG': np.random.choice([0, 1, 2, 3, 4], 500, p=[0.7, 0.15, 0.1, 0.03, 0.02]),
    }
    
    df = pd.DataFrame(mock_data)
    df.to_csv(labels_path, index=False)
    print(f"📊 Created mock labels.csv with {len(df)} samples")

def split_data(df, test_size=0.2, val_size=0.1):
    """Split dataset (official BRSET approach)"""
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['diabetic_retinopathy'])
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42, stratify=train_val['diabetic_retinopathy'])
    
    print(f"📊 Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test
