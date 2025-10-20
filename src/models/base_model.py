"""Base model interface for tree species classification."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn

from ..utils.logger import setup_logger


logger = setup_logger(__name__)


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all tree species identification models.

    All model implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(self, num_classes: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.

        Args:
            num_classes: Number of tree species to classify
            config: Optional configuration dictionary
        """
        super().__init__()
        self.num_classes = num_classes
        self.config = config or {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        pass

    @abstractmethod
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extraction backbone.

        Returns:
            Feature extractor module
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'config': self.config,
            'model_type': self.__class__.__name__
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path, strict: bool = True) -> None:
        """
        Load model weights.

        Args:
            path: Path to the saved model
            strict: Whether to strictly enforce key matching
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logger.info(f"Model loaded from {path}")

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning."""
        backbone = self.get_feature_extractor()
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        backbone = self.get_feature_extractor()
        for param in backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with total and trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        params = self.count_parameters()

        return {
            'model_type': self.__class__.__name__,
            'num_classes': self.num_classes,
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'config': self.config
        }
