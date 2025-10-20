"""Advanced data augmentation strategies for tree images."""

from typing import List, Optional
import random

import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

from ..utils.logger import setup_logger


logger = setup_logger(__name__)


class AugmentationPipeline:
    """
    Advanced augmentation pipeline with tree-specific transformations.
    """

    def __init__(
        self,
        image_size: int = 224,
        use_advanced: bool = False
    ):
        """
        Initialize augmentation pipeline.

        Args:
            image_size: Target image size
            use_advanced: Whether to use advanced augmentations
        """
        self.image_size = image_size
        self.use_advanced = use_advanced

        # Basic augmentation
        self.basic_aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ])

        # Advanced augmentation
        if use_advanced:
            self.advanced_aug = self._build_advanced_pipeline()

    def _build_advanced_pipeline(self) -> transforms.Compose:
        """Build advanced augmentation pipeline."""
        return transforms.Compose([
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.1),
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2)
            ], p=0.1),
            transforms.RandomApply([
                transforms.RandomAutocontrast()
            ], p=0.1),
        ])

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation to image.

        Args:
            image: Input PIL Image

        Returns:
            Augmented PIL Image
        """
        # Apply basic augmentation
        image = self.basic_aug(image)

        # Apply advanced if enabled
        if self.use_advanced:
            image = self.advanced_aug(image)

        return image


class LeafSpecificAugmentation:
    """Augmentation specifically designed for leaf images."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply leaf-specific augmentation."""
        # Random perspective transform (simulates viewing angle)
        if random.random() < 0.3:
            image = transforms.RandomPerspective(
                distortion_scale=0.2,
                p=1.0
            )(image)

        # Simulate lighting conditions
        if random.random() < 0.2:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Add slight blur to simulate depth of field
        if random.random() < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

        return image


class BarkSpecificAugmentation:
    """Augmentation specifically designed for bark images."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply bark-specific augmentation."""
        # Enhance texture details
        if random.random() < 0.3:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(1.0, 1.5))

        # Adjust contrast to emphasize texture
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.3))

        return image


class SeedSpecificAugmentation:
    """Augmentation specifically designed for seed images."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply seed-specific augmentation."""
        # Seeds might have different orientations
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            image = image.rotate(angle)

        # Vary scale more as seeds can be photographed at different distances
        if random.random() < 0.4:
            scale = random.uniform(0.6, 1.0)
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return image


class MixupAugmentation:
    """
    Mixup augmentation: blend two images and their labels.

    Reference: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize Mixup.

        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        label1: int,
        label2: int
    ) -> tuple:
        """
        Apply mixup to two images.

        Args:
            image1: First image tensor
            image2: Second image tensor
            label1: First image label
            label2: Second image label

        Returns:
            Tuple of (mixed_image, lambda_value) for label mixing
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0

        mixed_image = lam * image1 + (1 - lam) * image2

        return mixed_image, lam


class CutoutAugmentation:
    """
    Cutout augmentation: randomly mask out square regions.

    Reference: Improved Regularization of CNNs with Cutout (DeVries & Taylor, 2017)
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Initialize Cutout.

        Args:
            n_holes: Number of holes to cut
            length: Side length of the square holes
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout to image tensor.

        Args:
            image: Image tensor (C, H, W)

        Returns:
            Image with cutout applied
        """
        h, w = image.shape[1], image.shape[2]
        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.0

        mask = mask.unsqueeze(0)
        image = image * mask

        return image
