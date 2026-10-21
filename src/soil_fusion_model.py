"""
Soil Fusion Model — Feature Fusion Architecture
Combines EfficientNet-B0 image features with soil properties for improved maturity classification.
The original CNN backbone is kept frozen; soil features are fused before the final classifier.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SoilFusionModel(nn.Module):
    """
    Feature Fusion Model that combines:
    - EfficientNet-B0 image features (1280-dim)
    - Soil property features (5-dim → projected to 16-dim via MLP)
    
    Architecture:
        Image → EfficientNet backbone → 1280-dim
        Soil  → FC(5→16) + ReLU + Dropout → 16-dim
        Concat(1280, 16) → FC(1296→256) + ReLU + Dropout → FC(256→num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        num_soil_features: int = 5,
        soil_hidden_dim: int = 16,
        fusion_hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        pretrained_effnet_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_soil_features = num_soil_features
        
        # ===== Image Branch: EfficientNet-B0 Feature Extractor =====
        # Load pretrained EfficientNet-B0
        if pretrained_effnet_path:
            # Load from our existing trained weights
            self.effnet = models.efficientnet_b0(weights=None)
            num_ftrs = self.effnet.classifier[1].in_features
            self.effnet.classifier[1] = nn.Linear(num_ftrs, num_classes)
            self.effnet.load_state_dict(
                torch.load(pretrained_effnet_path, map_location="cpu")
            )
            print(f"[FUSION] Loaded pretrained EfficientNet from {pretrained_effnet_path}")
        else:
            self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the original classifier — we only want features
        self.image_feature_dim = self.effnet.classifier[1].in_features  # 1280
        self.effnet.classifier = nn.Identity()
        
        # ===== Soil Branch: Small MLP =====
        self.soil_branch = nn.Sequential(
            nn.Linear(num_soil_features, soil_hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(soil_hidden_dim),
            nn.Dropout(dropout_rate),
        )
        
        # ===== Fusion Classifier =====
        fusion_input_dim = self.image_feature_dim + soil_hidden_dim  # 1280 + 16 = 1296
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes),
        )
        
        print(f"[FUSION] Model created: image_dim={self.image_feature_dim}, "
              f"soil_dim={soil_hidden_dim}, fusion_dim={fusion_input_dim}")
    
    def extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features using the EfficientNet backbone only."""
        return self.effnet(x)  # (batch, 1280) since classifier is Identity
    
    def forward(self, image: torch.Tensor, soil: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with both image and soil inputs.
        
        Args:
            image: Batch of images (B, 3, 224, 224)
            soil:  Batch of soil features (B, 5) — already normalized
        
        Returns:
            Class logits (B, num_classes)
        """
        # Image branch
        img_features = self.extract_image_features(image)  # (B, 1280)
        
        # Soil branch
        soil_features = self.soil_branch(soil)  # (B, 16)
        
        # Fusion
        fused = torch.cat([img_features, soil_features], dim=1)  # (B, 1296)
        
        # Classification
        output = self.fusion_classifier(fused)  # (B, num_classes)
        return output
    
    def predict_image_only(self, image: torch.Tensor) -> torch.Tensor:
        """
        Fallback: predict using image only (zero soil features).
        Useful when soil data is unavailable.
        """
        batch_size = image.size(0)
        device = image.device
        zero_soil = torch.zeros(batch_size, self.num_soil_features, device=device)
        return self.forward(image, zero_soil)
    
    def freeze_backbone(self):
        """Freeze the EfficientNet backbone for fine-tuning only the fusion layers."""
        for param in self.effnet.parameters():
            param.requires_grad = False
        print("[FUSION] EfficientNet backbone FROZEN")
    
    def unfreeze_backbone(self):
        """Unfreeze the EfficientNet backbone for end-to-end fine-tuning."""
        for param in self.effnet.parameters():
            param.requires_grad = True
        print("[FUSION] EfficientNet backbone UNFROZEN")
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class ImageOnlyWrapper(nn.Module):
    """
    Wrapper to use SoilFusionModel for image-only inference.
    Automatically provides zero soil features.
    """
    
    def __init__(self, fusion_model: SoilFusionModel):
        super().__init__()
        self.fusion_model = fusion_model
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fusion_model.predict_image_only(image)
