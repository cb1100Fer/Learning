"""Tree species classifier orchestrating the identification pipeline."""

from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np

from .image_processor import ImageProcessor
from .feature_extractor import FeatureExtractor, TreePartType
from ..models.base_model import BaseModel
from ..utils.config import config
from ..utils.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    species: str
    confidence: float
    all_predictions: List[Tuple[str, float]]
    part_type: TreePartType
    metadata: Optional[Dict] = None


class TreeClassifier:
    """
    Main classifier for identifying tree species from images.

    Orchestrates the entire pipeline: image processing, feature extraction,
    and classification.
    """

    def __init__(
        self,
        model: BaseModel,
        species_names: List[str],
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ):
        """
        Initialize tree classifier.

        Args:
            model: Trained model for classification
            species_names: List of species names (ordered by class index)
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for valid prediction
        """
        self.model = model
        self.species_names = species_names
        self.num_classes = len(species_names)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Initialize components
        self.image_processor = ImageProcessor()

        # Get confidence threshold from config
        if confidence_threshold is None:
            self.confidence_threshold = config.get(
                'confidence_threshold',
                0.7,
                config_type='model'
            )
        else:
            self.confidence_threshold = confidence_threshold

        self.top_k = config.get('inference.top_k', 5, config_type='model')

        logger.info(
            f"TreeClassifier initialized with {self.num_classes} species "
            f"on device: {self.device}"
        )

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        part_type: Optional[TreePartType] = None,
        return_top_k: Optional[int] = None
    ) -> PredictionResult:
        """
        Predict tree species from an image.

        Args:
            image: Input image (path or numpy array)
            part_type: Type of tree part (auto-detected if None)
            return_top_k: Number of top predictions to return

        Returns:
            PredictionResult with species and confidence
        """
        # Preprocess image
        image_tensor = self.image_processor.preprocess_image(image, augment=False)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Auto-detect part type if not provided
        if part_type is None:
            pil_image = self.image_processor.load_image(image) if isinstance(
                image, (str, Path)
            ) else image
            detected_type = self.image_processor.detect_image_type(pil_image)
            part_type = TreePartType(detected_type)

        # Get predictions
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

        # Get top-k predictions
        k = return_top_k or self.top_k
        top_probs, top_indices = torch.topk(probabilities[0], k=min(k, self.num_classes))

        # Convert to species names
        all_predictions = [
            (self.species_names[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        # Get top prediction
        species, confidence = all_predictions[0]

        # Create result
        result = PredictionResult(
            species=species,
            confidence=confidence,
            all_predictions=all_predictions,
            part_type=part_type,
            metadata={
                'device': str(self.device),
                'threshold': self.confidence_threshold,
                'is_confident': confidence >= self.confidence_threshold
            }
        )

        logger.info(
            f"Prediction: {species} ({confidence:.2%}) - Part type: {part_type.value}"
        )

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        part_types: Optional[List[TreePartType]] = None
    ) -> List[PredictionResult]:
        """
        Predict tree species for a batch of images.

        Args:
            images: List of input images
            part_types: Optional list of part types for each image

        Returns:
            List of PredictionResults
        """
        results = []

        for i, image in enumerate(images):
            part_type = part_types[i] if part_types else None
            result = self.predict(image, part_type=part_type)
            results.append(result)

        return results

    def predict_with_ensemble(
        self,
        images: Dict[TreePartType, Union[str, Path, np.ndarray]],
        use_weighted: bool = True
    ) -> PredictionResult:
        """
        Predict using ensemble of multiple tree parts.

        Args:
            images: Dictionary mapping part types to images
            use_weighted: Whether to use weighted ensemble

        Returns:
            Combined PredictionResult
        """
        all_logits = []
        weights = []

        # Get predictions for each part
        for part_type, image in images.items():
            image_tensor = self.image_processor.preprocess_image(image, augment=False)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(image_tensor)
                all_logits.append(logits)

                if use_weighted:
                    # Get weight from config
                    weight = config.get(
                        f'feature_weights.{part_type.value}',
                        0.33,
                        config_type='model'
                    )
                    weights.append(weight)
                else:
                    weights.append(1.0)

        # Combine logits
        weights = torch.tensor(weights).to(self.device)
        weights = weights / weights.sum()

        combined_logits = sum(
            w * logits for w, logits in zip(weights, all_logits)
        )

        probabilities = torch.softmax(combined_logits, dim=1)

        # Get top-k
        top_probs, top_indices = torch.topk(
            probabilities[0],
            k=min(self.top_k, self.num_classes)
        )

        all_predictions = [
            (self.species_names[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        species, confidence = all_predictions[0]

        result = PredictionResult(
            species=species,
            confidence=confidence,
            all_predictions=all_predictions,
            part_type=TreePartType.UNKNOWN,  # Multiple parts
            metadata={
                'ensemble': True,
                'parts_used': [pt.value for pt in images.keys()],
                'weights': weights.tolist()
            }
        )

        logger.info(f"Ensemble prediction: {species} ({confidence:.2%})")

        return result

    def get_species_info(self, species_name: str) -> Optional[Dict]:
        """
        Get information about a species (placeholder for future DB integration).

        Args:
            species_name: Name of the species

        Returns:
            Dictionary with species information, or None
        """
        # Placeholder - would integrate with a species database
        return {
            'name': species_name,
            'scientific_name': 'Unknown',
            'description': 'Species information not available yet.'
        }

    def save_model(self, path: Union[str, Path]) -> None:
        """Save model weights."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        """Load model weights."""
        self.model.load(path)
        logger.info(f"Model loaded from {path}")
