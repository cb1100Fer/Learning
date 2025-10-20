"""Image preprocessing and augmentation for tree identification."""

from typing import Tuple, Optional, Union
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from ..utils.config import config
from ..utils.logger import setup_logger


logger = setup_logger(__name__)


class ImageProcessor:
    """Handles image preprocessing, normalization, and augmentation."""

    def __init__(self, config_override: Optional[dict] = None):
        """
        Initialize image processor with configuration.

        Args:
            config_override: Optional dict to override default config
        """
        self.config = config_override or config.model_config.get('image', {})

        self.input_size = tuple(self.config.get('input_size', [224, 224]))
        self.mean = self.config.get('mean', [0.485, 0.456, 0.406])
        self.std = self.config.get('std', [0.229, 0.224, 0.225])

        # Define preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Define augmentation transforms for training
        aug_config = config.model_config.get('augmentation', {})
        if aug_config.get('enabled', True):
            self.augment = self._build_augmentation_pipeline(aug_config)
        else:
            self.augment = None

        logger.info(f"ImageProcessor initialized with input size: {self.input_size}")

    def _build_augmentation_pipeline(self, aug_config: dict) -> transforms.Compose:
        """Build augmentation pipeline from config."""
        transforms_list = [
            transforms.Resize(256),
            transforms.RandomResizedCrop(self.input_size[0]),
        ]

        techniques = aug_config.get('techniques', [])
        for technique in techniques:
            if isinstance(technique, dict):
                for name, params in technique.items():
                    if name == 'horizontal_flip':
                        transforms_list.append(transforms.RandomHorizontalFlip(p=params))
                    elif name == 'vertical_flip':
                        transforms_list.append(transforms.RandomVerticalFlip(p=params))
                    elif name == 'rotation':
                        transforms_list.append(transforms.RandomRotation(degrees=params))
                    elif name == 'color_jitter' and isinstance(params, dict):
                        transforms_list.append(transforms.ColorJitter(**params))
                    elif name == 'gaussian_blur':
                        transforms_list.append(
                            transforms.RandomApply([transforms.GaussianBlur(3)], p=params)
                        )

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        return transforms.Compose(transforms_list)

    def load_image(
        self,
        image_path: Union[str, Path],
        mode: str = 'RGB'
    ) -> Image.Image:
        """
        Load image from file path.

        Args:
            image_path: Path to image file
            mode: Color mode (RGB, L for grayscale)

        Returns:
            PIL Image
        """
        try:
            image = Image.open(image_path).convert(mode)
            logger.debug(f"Loaded image: {image_path}, size: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        augment: bool = False
    ) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Input image (PIL Image, numpy array, or path)
            augment: Whether to apply augmentation

        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Apply transforms
        if augment and self.augment is not None:
            tensor = self.augment(image)
        else:
            tensor = self.preprocess(image)

        return tensor

    def preprocess_batch(
        self,
        images: list,
        augment: bool = False
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of images
            augment: Whether to apply augmentation

        Returns:
            Batched tensor of shape (N, C, H, W)
        """
        tensors = [self.preprocess_image(img, augment) for img in images]
        return torch.stack(tensors)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize tensor back to image.

        Args:
            tensor: Normalized tensor

        Returns:
            Denormalized numpy array (H, W, C)
        """
        # Denormalize
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)

        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to numpy
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

        return image

    def detect_image_type(self, image: Image.Image) -> str:
        """
        Attempt to classify image as leaf, bark, or seed based on characteristics.

        This is a simple heuristic-based approach. For production, use a separate
        classifier model.

        Args:
            image: Input PIL Image

        Returns:
            Predicted type: 'leaf', 'bark', or 'seed'
        """
        # Convert to numpy for analysis
        img_array = np.array(image)

        # Simple heuristics (these would be refined with ML)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Calculate color statistics
        green_mask = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
        green_ratio = np.sum(green_mask > 0) / green_mask.size

        # Texture analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Simple classification based on heuristics
        if green_ratio > 0.3:
            return 'leaf'
        elif edge_density > 0.15:
            return 'bark'
        else:
            return 'seed'

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply enhancement techniques to improve image quality.

        Args:
            image: Input PIL Image

        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy
        img_array = np.array(image)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced)
