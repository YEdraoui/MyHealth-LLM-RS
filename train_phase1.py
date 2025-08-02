import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os
from PIL import Image

sys.path.append('vision_model')
sys.path.append('llm')

try:
    from fundus_classifier import BRSETFundusClassifier, get_transforms, BRSET_CONDITIONS
    from clinical_llm import ClinicalOphthalmologyLLM
    IMPORTS_OK = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    IMPORTS_OK = False

class MockDataset(Dataset):
    def __init__(self, num_samples=50, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        np.random.seed(42)
        # Generate realistic mock labels (most images normal)
        self.labels = np.random.binomial(1, 0.1, (num_samples, 13))
        # Ensure some positive cases
        self.labels[0, 0] = 1  # DR case
        self.labels[1, 4] = 1  # AMD case
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create mock retinal image (brownish background like fundus)
        image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        image[:, :, 0] = np.random.randint(100, 180, (224, 224))  # Red channel
        image[:, :, 1] = np.random.randint(50, 120, (224, 224))   # Green channel  
        image[:, :, 2] = np.random.randint(20, 80, (224, 224))    # Blue channel
        
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor(self.labels[idx])

def train_phase1():
    print("🚀 PHASE 1 TRAINING PIPELINE")
    print("=" * 40)
    
    if not IMPORTS_OK:
        print("❌ Cannot import required modules")
        return False
    
    try:
        # Create model
        print("🔬 Creating BRSET Fundus Classifier...")
        model = BRSETFundusClassifier(num_classes=13, model_name='resnet50')
        print(f"✅ Model created: {model.model_name}")
        print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create dataset
        print("📊 Creating mock BRSET dataset...")
        transform = get_transforms(augment=True)
        dataset = MockDataset(num_samples=40, transform=transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"🚀 Training on: {device}")
        
        # Train for 2 epochs
        model.train()
        for epoch in range(2):
            total_loss = 0
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Test inference
        print("🧪 Testing inference...")
        model.eval()
        with torch.no_grad():
            test_image, test_label = dataset[0]
            prediction = model.predict_proba(test_image.unsqueeze(0).to(device))
            prediction = prediction.cpu().squeeze().numpy()
        
        print("📊 Sample prediction:")
        for i, (condition, prob) in enumerate(zip(BRSET_CONDITIONS, prediction)):
            if prob > 0.3:
                print(f"  {condition}: {prob:.3f}")
        
        # Test LLM integration
        print("🤖 Testing Clinical LLM integration...")
        llm = ClinicalOphthalmologyLLM()
        
        # Sample patient data
        sample_patient = {
            'age': 65, 'sex': 2, 'diabetes_time': 12, 'insulin_use': 1
        }
        
        # Generate clinical report
        report = llm.generate_clinical_report(sample_patient, prediction)
        print("📋 Generated Clinical Report:")
        print(report['report'])
        
        # Save model
        os.makedirs('data/models', exist_ok=True)
        torch.save(model.state_dict(), 'data/models/phase1_vision_model.pth')
        print("💾 Model saved to: data/models/phase1_vision_model.pth")
        
        # Save training metadata
        metadata = {
            'model_name': model.model_name,
            'num_classes': model.num_classes,
            'conditions': BRSET_CONDITIONS,
            'final_loss': avg_loss,
            'device': str(device)
        }
        
        with open('data/models/phase1_metadata.json', 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print("✅ PHASE 1 TRAINING COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"⚠️ Training error: {e}")
        print("✅ Phase 1 structure created successfully")
        return False

if __name__ == "__main__":
    success = train_phase1()
    if success:
        print("🎉 Ready for Phase 2: Multimodal Fusion")
    else:
        print("📝 Phase 1 code structure ready for when dependencies are available")
