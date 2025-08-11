import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalClassifier(nn.Module):
    def __init__(self, img_model_name='efficientnet_b0', num_tab_features=4, num_classes=6):
        super().__init__()
        backbone = getattr(models, img_model_name)(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.img_model = backbone
        self.tab_model = nn.Sequential(
            nn.Linear(num_tab_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, tab):
        img_feat = self.img_model(img)
        tab_feat = self.tab_model(tab)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.fc(combined)
