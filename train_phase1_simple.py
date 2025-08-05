"""
PHASE 1: Simplified Vision Model (works without timm)
Uses torchvision ResNet on real BRSET dataset
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # Create fallback image if loading fails
            print(f"Warning: Could not load {image_path}, using fallback")
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
    """ResNet-based eye condition classifier"""
    
    def __init__(self, num_classes=13):
        super().__init__()
        
        # Use ResNet50 from torchvision
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        
        # Replace final layer for multi-label classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

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
        return None
    
    # Load real data
    df = pd.read_csv(labels_path)
    print(f"📊 Loaded {len(df):,} real samples")
    
    # Use subset for demo (first 500 samples)
    df_sample = df.head(500)
    print(f"📊 Using {len(df_sample)} samples for training demo")
    
    # Split data
    train_df, val_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
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
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Create model
    model = EyeConditionClassifier(num_classes=13)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🧠 Model: ResNet50 + Multi-label Head")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🚀 Device: {device}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    training_stats = []
    
    for epoch in range(2):  # 2 epochs for demo
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
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
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
        
        # Calculate AUROC
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
    
    # Save model and stats
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/phase1_best_model.pth')
    
    with open('models/training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print("🎉 PHASE 1 TRAINING COMPLETED!")
    print(f"📊 Final AUROC: {training_stats[-1]['auroc']:.4f}")
    
    return model, training_stats

if __name__ == "__main__":
    model, stats = train_model()
    if model:
        print("✅ Vision model trained successfully!")
        print("🚀 Ready for Streamlit demo!")
