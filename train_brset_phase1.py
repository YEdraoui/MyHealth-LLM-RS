"""
BRSET Phase 1 Training (Official Implementation Style)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import sys
import os

sys.path.append('src')
sys.path.append('vision_model')
sys.path.append('llm')

from get_dataset import get_dataset, split_data
from data_loader import BRSETDataset, process_transforms
from brset_models import create_brset_model, BRSET_TASKS
from clinical_llm import ClinicalOphthalmologyLLM

def train_brset_model(task='diabetes', epochs=3, batch_size=16):
    """Train BRSET model for specific task"""
    
    print(f"🚀 Training BRSET Model - Task: {task}")
    print("=" * 50)
    
    # Load dataset
    df, images_path = get_dataset(dataset_path='data/', download=False)
    if df is None:
        return None
    
    # Split data
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1)
    
    # Create datasets
    train_transform = process_transforms(shape=(224, 224), augment=True)
    val_transform = process_transforms(shape=(224, 224), augment=False)
    
    train_dataset = BRSETDataset(train_df, images_path, train_transform, task=task)
    val_dataset = BRSETDataset(val_df, images_path, val_transform, task=task)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = create_brset_model(task=task)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🔬 Model: {model.__class__.__name__}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🚀 Training on: {device}")
    
    # Training setup
    task_config = BRSET_TASKS[task]
    if task_config['loss'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (images, labels, metadata) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            if task_config['loss'] == 'bce':
                loss = criterion(outputs.squeeze(), labels)
            else:
                loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels, metadata in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                if task_config['loss'] == 'bce':
                    loss = criterion(outputs.squeeze(), labels)
                    preds = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                else:
                    loss = criterion(outputs, labels.long())
                    preds = torch.softmax(outputs, dim=1).cpu().numpy()
                
                val_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if task_config['metric'] == 'auroc' and len(np.unique(all_labels)) > 1:
            try:
                metric = roc_auc_score(all_labels, all_preds)
                metric_name = 'AUROC'
            except:
                metric = f1_score(all_labels, np.array(all_preds) > 0.5)
                metric_name = 'F1'
        elif task_config['metric'] == 'accuracy':
            metric = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
            metric_name = 'Accuracy'
        else:
            metric = f1_score(all_labels, np.array(all_preds) > 0.5, average='weighted')
            metric_name = 'F1'
        
        print(f"✅ Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, {metric_name}={metric:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('data/models', exist_ok=True)
            torch.save(model.state_dict(), f'data/models/brset_{task}_best.pth')
    
    return model, {'task': task, 'metric': metric, 'metric_name': metric_name}

def test_multimodal_integration():
    """Test complete multimodal pipeline"""
    print("🤖 Testing Multimodal Integration")
    print("=" * 40)
    
    # Load data for testing
    df, _ = get_dataset(dataset_path='data/', download=False)
    if df is None:
        return
    
    # Create LLM
    llm = ClinicalOphthalmologyLLM()
    
    # Test with sample data
    sample_row = df.iloc[0]
    
    # Mock vision predictions (would come from trained model)
    vision_predictions = np.random.random(13) * 0.3  # Low random predictions
    vision_predictions[0] = 0.8  # High DR prediction
    
    # Extract patient metadata
    patient_data = {
        'age': sample_row['patient_age'],
        'sex': sample_row['patient_sex'],
        'diabetes_time': sample_row.get('diabetes_time', 0),
        'insulin_use': sample_row.get('insulin_use', 0)
    }
    
    # Generate clinical report
    report = llm.generate_clinical_report(patient_data, vision_predictions)
    
    print("📋 Sample Multimodal Report:")
    print(report['report'])
    
    return report

if __name__ == "__main__":
    print("🧠 BRSET PHASE 1 TRAINING PIPELINE")
    print("=" * 50)
    
    # Train models for key tasks
    tasks_to_train = ['diabetes', 'DR_2class']
    
    results = {}
    for task in tasks_to_train:
        print(f"\n🎯 Training task: {task}")
        model, result = train_brset_model(task=task, epochs=2, batch_size=8)
        results[task] = result
        
    # Test multimodal integration
    print(f"\n🔗 Testing Multimodal Integration:")
    multimodal_result = test_multimodal_integration()
    
    # Save results
    import json
    with open('data/models/phase1_brset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ PHASE 1 BRSET TRAINING COMPLETED!")
    print(f"📊 Results: {results}")
    print(f"💾 Models saved in data/models/")
    print(f"🔄 Ready for Phase 2: Multimodal Fusion")
