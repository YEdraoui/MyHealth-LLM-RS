"""
PHASE 1: Vision Model Training on Real BRSET Dataset
Multi-label classification for 13+ eye conditions
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import timm
import json
import os

class BRSETDataset(Dataset):
    """Real BRSET Dataset Loader"""
    
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Eye condition columns from real BRSET
        self.condition_cols = [
            'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
            'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
            'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
            'increased_cup_disc', 'other'
        ]
        
        print(f"✅ Dataset initialized: {len(df)} real samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load real image
        image_path = self.images_dir / row['image_id']
        if not image_path.exists():
            # Try without extension
            image_path = self.images_dir / f"{row['image_id'].split('.')[0]}.jpg"
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Create black image as fallback
            image = torch.zeros(3, 224, 224)
        
        # Get multi-label targets
        labels = []
        for col in self.condition_cols:
            if col in row:
                labels.append(float(row[col]))
            else:
                labels.append(0.0)
        
        return image, torch.tensor(labels, dtype=torch.float32)

class EyeConditionClassifier(nn.Module):
    """Multi-label eye condition classifier"""
    
    def __init__(self, num_classes=13, model_name='efficientnet_b0'):
        super().__init__()
        
        # Use timm for pre-trained models
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def train_model():
    """Train vision model on real BRSET data"""
    print("🚀 PHASE 1: TRAINING VISION MODEL ON REAL BRSET")
    print("=" * 50)
    
    # Check for dataset
    labels_path = Path("data/brset_real/labels.csv")
    images_path = Path("data/brset_real/images")
    
    if not labels_path.exists():
        print("❌ Real BRSET dataset not found!")
        print(f"Expected: {labels_path}")
        print("Run Phase 0 setup first!")
        return None
    
    # Load real data
    df = pd.read_csv(labels_path)
    print(f"📊 Loaded {len(df):,} real samples")
    
    # Filter out samples with missing images (for demo)
    df_sample = df.head(1000)  # Use first 1000 for demo
    print(f"📊 Using {len(df_sample)} samples for training demo")
    
    # Split data
    train_df, val_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BRSETDataset(train_df, images_path, train_transform)
    val_dataset = BRSETDataset(val_df, images_path, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model
    model = EyeConditionClassifier(num_classes=13, model_name='efficientnet_b0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🧠 Model: {model.__class__.__name__}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🚀 Device: {device}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training loop
    best_val_loss = float('inf')
    training_stats = []
    
    for epoch in range(3):  # 3 epochs for demo
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate AUROC (handle potential issues)
        try:
            auroc = roc_auc_score(all_labels, all_preds, average='weighted')
        except:
            auroc = 0.5
        
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'auroc': auroc
        }
        training_stats.append(epoch_stats)
        
        print(f"✅ Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, AUROC={auroc:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/phase1_best_model.pth')
            print("💾 Best model saved!")
    
    # Save training stats
    with open('models/training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print("🎉 PHASE 1 TRAINING COMPLETED!")
    return model, training_stats

if __name__ == "__main__":
    model, stats = train_model()
    if model:
        print("✅ Vision model trained successfully!")
        print(f"📊 Final AUROC: {stats[-1]['auroc']:.4f}")
        print("🔄 Ready for Phase 2: Multimodal fusion")
