"""
BRSET Model Architectures (based on official implementation)
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class BRSETResNet(nn.Module):
    """ResNet-based model (official BRSET approach)"""
    
    def __init__(self, num_classes=1, model_name='resnet50', pretrained=True, task='diabetes'):
        super().__init__()
        self.num_classes = num_classes
        self.task = task
        
        # Load backbone
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Task-specific heads
        if task == 'sex' or task == 'DR_3class':
            # Classification tasks
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:
            # Binary or multi-label tasks
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, num_classes)
            )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            if self.task == 'sex' or self.task == 'DR_3class':
                probs = F.softmax(logits, dim=1)
            else:
                probs = torch.sigmoid(logits)
        return probs

def create_brset_model(task='diabetes', num_classes=None):
    """Create BRSET model based on task"""
    
    if task == 'diabetes':
        return BRSETResNet(num_classes=1, task='diabetes')
    elif task == 'DR_2class':
        return BRSETResNet(num_classes=1, task='DR_2class')
    elif task == 'DR_3class':
        return BRSETResNet(num_classes=5, task='DR_3class')  # 0-4 grades
    elif task == 'sex':
        return BRSETResNet(num_classes=2, task='sex')
    elif task == 'multi_label':
        return BRSETResNet(num_classes=13, task='multi_label')
    else:
        raise ValueError(f"Unknown task: {task}")

# Define BRSET tasks and their configurations
BRSET_TASKS = {
    'diabetes': {'num_classes': 1, 'loss': 'bce', 'metric': 'auroc'},
    'DR_2class': {'num_classes': 1, 'loss': 'bce', 'metric': 'auroc'},
    'DR_3class': {'num_classes': 5, 'loss': 'ce', 'metric': 'accuracy'},
    'sex': {'num_classes': 2, 'loss': 'ce', 'metric': 'accuracy'},
    'multi_label': {'num_classes': 13, 'loss': 'bce', 'metric': 'f1'}
}

BRSET_CONDITIONS = [
    'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
    'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
    'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
    'increased_cup_disc', 'other'
]
