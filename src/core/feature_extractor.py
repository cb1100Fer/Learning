"""Feature extraction from different tree parts (leaves, bark, seeds)."""

from typing import Dict, List, Optional
from enum import Enum

import torch
import torch.nn as nn
import numpy as np

from ..utils.logger import setup_logger


logger = setup_logger(__name__)


class TreePartType(Enum):
    """Enumeration of tree part types."""
    LEAF = "leaf"
    BARK = "bark"
    SEED = "seed"
    UNKNOWN = "unknown"


class FeatureExtractor:
    """
    Extracts features from images of different tree parts.

    This class manages feature extraction for different tree components
    and can combine features from multiple sources.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 2048,
        use_multi_scale: bool = False
    ):
        """
        Initialize feature extractor.

        Args:
            backbone: Neural network backbone for feature extraction
            feature_dim: Dimension of extracted features
            use_multi_scale: Whether to extract multi-scale features
        """
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.use_multi_scale = use_multi_scale

        # Feature weights for different tree parts
        self.part_weights = {
            TreePartType.LEAF: 0.5,
            TreePartType.BARK: 0.3,
            TreePartType.SEED: 0.2,
        }

        logger.info(f"FeatureExtractor initialized with feature_dim={feature_dim}")

    def extract_features(
        self,
        image_tensor: torch.Tensor,
        part_type: Optional[TreePartType] = None
    ) -> torch.Tensor:
        """
        Extract features from image tensor.

        Args:
            image_tensor: Input image tensor (B, C, H, W)
            part_type: Type of tree part (leaf, bark, seed)

        Returns:
            Feature tensor (B, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(image_tensor)

        # If multi-scale, extract features at different resolutions
        if self.use_multi_scale:
            features = self._extract_multi_scale(image_tensor)

        return features

    def _extract_multi_scale(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features from different layers.

        Args:
            image_tensor: Input image tensor

        Returns:
            Combined multi-scale features
        """
        # This would extract features from multiple layers of the backbone
        # For now, returns single-scale features
        # In practice, you'd hook into intermediate layers
        return self.backbone(image_tensor)

    def combine_features(
        self,
        features_dict: Dict[TreePartType, torch.Tensor],
        weighted: bool = True
    ) -> torch.Tensor:
        """
        Combine features from multiple tree parts.

        Args:
            features_dict: Dictionary mapping part types to feature tensors
            weighted: Whether to use weighted combination

        Returns:
            Combined feature tensor
        """
        if not features_dict:
            raise ValueError("features_dict cannot be empty")

        if len(features_dict) == 1:
            return list(features_dict.values())[0]

        # Weighted or average combination
        combined = None
        total_weight = 0.0

        for part_type, features in features_dict.items():
            weight = self.part_weights.get(part_type, 0.33) if weighted else 1.0
            total_weight += weight

            if combined is None:
                combined = features * weight
            else:
                combined += features * weight

        # Normalize
        if weighted and total_weight > 0:
            combined = combined / total_weight

        return combined

    def set_part_weights(self, weights: Dict[TreePartType, float]) -> None:
        """
        Update weights for different tree parts.

        Args:
            weights: Dictionary mapping part types to weights
        """
        self.part_weights.update(weights)
        logger.info(f"Updated part weights: {self.part_weights}")

    def extract_attention_maps(
        self,
        image_tensor: torch.Tensor
    ) -> Optional[np.ndarray]:
        """
        Extract attention maps to visualize what the model focuses on.

        Args:
            image_tensor: Input image tensor

        Returns:
            Attention map as numpy array, or None if not available
        """
        # This would be implemented with specific attention mechanisms
        # Placeholder for now
        logger.warning("Attention map extraction not yet implemented")
        return None

    def get_feature_statistics(
        self,
        features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate statistics of extracted features.

        Args:
            features: Feature tensor

        Returns:
            Dictionary of statistics
        """
        features_np = features.cpu().numpy()

        stats = {
            'mean': float(np.mean(features_np)),
            'std': float(np.std(features_np)),
            'min': float(np.min(features_np)),
            'max': float(np.max(features_np)),
            'sparsity': float(np.sum(features_np == 0) / features_np.size),
        }

        return stats
