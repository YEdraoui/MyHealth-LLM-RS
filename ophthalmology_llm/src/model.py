import torch
import torch.nn as nn
from torchvision import models

NUM_LABELS = 6
# Expected labels for the BRSET dataset
# "diabetic_retinopathy", "macular_edema", "amd", "retinal_detachment", "increased_cup_disc", "other"

class FundusClassifier(nn.Module):
    def __init__(self, num_labels: int = NUM_LABELS):
        super().__init__()
        # ResNet50 backbone (ImageNet)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        # Replace head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_labels)
        )

    def forward(self, x):
        return self.backbone(x)

def create_model():
    return FundusClassifier(NUM_LABELS)
