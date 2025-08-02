"""
Phase 2: Multimodal Fusion Architecture
Combines vision + patient text for enhanced predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionModel(nn.Module):
    """Fusion of vision features + patient clinical text"""
    
    def __init__(self, vision_dim=2048, text_dim=768, num_classes=13):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        
        # Vision pathway (from Phase 1)
        self.vision_projection = nn.Linear(vision_dim, 512)
        
        # Text pathway (patient metadata)
        self.text_projection = nn.Linear(text_dim, 512)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),  # vision + text
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Cross-attention for explainability
        self.cross_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
    
    def forward(self, vision_features, text_features):
        # Project to common space
        vision_proj = self.vision_projection(vision_features)  # [B, 512]
        text_proj = self.text_projection(text_features)        # [B, 512]
        
        # Concatenate features
        fused = torch.cat([vision_proj, text_proj], dim=1)     # [B, 1024]
        
        # Final prediction
        logits = self.fusion(fused)                            # [B, num_classes]
        return logits
    
    def forward_with_attention(self, vision_features, text_features):
        """Forward pass with cross-attention for explainability"""
        vision_proj = self.vision_projection(vision_features).unsqueeze(1)  # [B, 1, 512]
        text_proj = self.text_projection(text_features).unsqueeze(1)        # [B, 1, 512]
        
        # Cross attention: vision queries text
        attended, attention_weights = self.cross_attention(
            query=vision_proj, key=text_proj, value=text_proj
        )
        
        # Combine attended vision with original text
        combined = torch.cat([attended.squeeze(1), text_proj.squeeze(1)], dim=1)
        logits = self.fusion(combined)
        
        return logits, attention_weights

def create_multimodal_model():
    """Create multimodal fusion model"""
    model = MultimodalFusionModel(vision_dim=2048, text_dim=768, num_classes=13)
    return model

print("✅ Phase 2: Multimodal Fusion Architecture created")
