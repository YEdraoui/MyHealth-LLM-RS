import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np

class BRSETFundusClassifier(nn.Module):
    def __init__(self, num_classes=13, model_name='resnet50', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Use ResNet50 (available in torchvision)
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
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
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

def get_transforms(image_size=224, augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

BRSET_CONDITIONS = [
    'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
    'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
    'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
    'increased_cup_disc', 'other'
]

def create_model():
    return BRSETFundusClassifier(num_classes=13, model_name='resnet50')

print("✅ Vision model created with ResNet50 backbone")
