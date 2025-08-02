"""
Configuration for Ophthalmology LLM System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "brset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Dataset configuration
BRSET_LABELS = [
    'diabetic_retinopathy',
    'age_related_macular_degeneration', 
    'media_haze',
    'drusens',
    'myopia',
    'branch_retinal_vein_occlusion',
    'tessellation',
    'epiretinal_membrane',
    'laser_scar',
    'macular_scar',
    'silver_wire_arteriosclerosis',
    'arteriovenous_nicking',
    'central_retinal_vein_occlusion',
    'tortuous_vessels'
]

# Top 5 priority conditions for initial focus
PRIORITY_CONDITIONS = [
    'diabetic_retinopathy',
    'age_related_macular_degeneration',
    'media_haze',
    'laser_scar',
    'epiretinal_membrane'
]

# Model configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Clinical context
PATIENT_METADATA_FIELDS = ['age', 'sex']
