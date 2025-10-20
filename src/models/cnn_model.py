"""CNN-based model implementations for tree species classification."""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models

from .base_model import BaseModel
from ..utils.logger import setup_logger


logger = setup_logger(__name__)


class CNNModel(BaseModel):
    """
    CNN-based model using transfer learning from pretrained models.

    Supports various architectures: ResNet, EfficientNet, DenseNet, etc.
    """

    def __init__(
        self,
        num_classes: int,
        architecture: str = 'resnet50',
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CNN model.

        Args:
            num_classes: Number of tree species classes
            architecture: Backbone architecture name
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            config: Optional configuration dictionary
        """
        super().__init__(num_classes, config)

        self.architecture = architecture
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        # Load backbone
        self.backbone = self._load_backbone()

        # Get feature dimension
        self.feature_dim = self._get_feature_dim()

        # Build classifier head
        self.classifier = self._build_classifier()

        logger.info(
            f"CNNModel initialized: {architecture}, "
            f"pretrained={pretrained}, classes={num_classes}"
        )

    def _load_backbone(self) -> nn.Module:
        """Load pretrained backbone."""
        weights = 'DEFAULT' if self.pretrained else None

        if self.architecture == 'resnet50':
            model = models.resnet50(weights=weights)
            # Remove final FC layer
            backbone = nn.Sequential(*list(model.children())[:-1])

        elif self.architecture == 'resnet101':
            model = models.resnet101(weights=weights)
            backbone = nn.Sequential(*list(model.children())[:-1])

        elif self.architecture == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=weights)
            backbone = model.features

        elif self.architecture == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=weights)
            backbone = model.features

        elif self.architecture == 'densenet121':
            model = models.densenet121(weights=weights)
            backbone = model.features

        elif self.architecture == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights=weights)
            backbone = model.features

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        return backbone

    def _get_feature_dim(self) -> int:
        """Get feature dimension of the backbone."""
        # Create dummy input to infer output size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)

            # Flatten if needed
            if len(features.shape) > 2:
                features = nn.AdaptiveAvgPool2d(1)(features)
                features = features.flatten(1)

            feature_dim = features.shape[1]

        return feature_dim

    def _build_classifier(self) -> nn.Module:
        """Build classifier head."""
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Global average pooling if needed
        if len(features.shape) > 2:
            features = nn.AdaptiveAvgPool2d(1)(features)

        # Flatten
        features = features.flatten(1)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extraction backbone."""
        return self.backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Feature tensor (B, feature_dim)
        """
        features = self.backbone(x)

        if len(features.shape) > 2:
            features = nn.AdaptiveAvgPool2d(1)(features)

        features = features.flatten(1)

        return features


class MultiInputCNNModel(BaseModel):
    """
    Multi-input CNN model for processing different tree parts separately.

    Processes leaf, bark, and seed images through separate or shared backbones.
    """

    def __init__(
        self,
        num_classes: int,
        architecture: str = 'resnet50',
        pretrained: bool = True,
        shared_backbone: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-input model.

        Args:
            num_classes: Number of tree species classes
            architecture: Backbone architecture
            pretrained: Whether to use pretrained weights
            shared_backbone: Whether to share backbone across inputs
            config: Optional configuration
        """
        super().__init__(num_classes, config)

        self.architecture = architecture
        self.shared_backbone = shared_backbone

        if shared_backbone:
            # Single shared backbone
            self.backbone = CNNModel(
                num_classes=num_classes,
                architecture=architecture,
                pretrained=pretrained,
                config=config
            )
        else:
            # Separate backbones for each input type
            self.leaf_backbone = CNNModel(
                num_classes=num_classes,
                architecture=architecture,
                pretrained=pretrained,
                config=config
            )
            self.bark_backbone = CNNModel(
                num_classes=num_classes,
                architecture=architecture,
                pretrained=pretrained,
                config=config
            )
            self.seed_backbone = CNNModel(
                num_classes=num_classes,
                architecture=architecture,
                pretrained=pretrained,
                config=config
            )

        logger.info(f"MultiInputCNNModel initialized (shared={shared_backbone})")

    def forward(
        self,
        x: torch.Tensor,
        input_type: str = 'leaf'
    ) -> torch.Tensor:
        """
        Forward pass for a specific input type.

        Args:
            x: Input tensor (B, C, H, W)
            input_type: Type of input ('leaf', 'bark', 'seed')

        Returns:
            Logits (B, num_classes)
        """
        if self.shared_backbone:
            return self.backbone(x)
        else:
            if input_type == 'leaf':
                return self.leaf_backbone(x)
            elif input_type == 'bark':
                return self.bark_backbone(x)
            elif input_type == 'seed':
                return self.seed_backbone(x)
            else:
                raise ValueError(f"Unknown input type: {input_type}")

    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor (returns shared backbone if available)."""
        if self.shared_backbone:
            return self.backbone.get_feature_extractor()
        else:
            return self.leaf_backbone.get_feature_extractor()
