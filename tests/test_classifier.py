"""Tests for tree classifier."""

import pytest
import torch
from PIL import Image

from src.core.classifier import TreeClassifier, PredictionResult
from src.core.feature_extractor import TreePartType
from src.models.cnn_model import CNNModel


@pytest.fixture
def sample_model():
    """Create a sample model."""
    species_names = ['Oak', 'Maple', 'Pine', 'Birch', 'Elm']
    model = CNNModel(
        num_classes=len(species_names),
        architecture='resnet50',
        pretrained=False  # Use random weights for testing
    )
    return model, species_names


@pytest.fixture
def classifier(sample_model):
    """Create classifier instance."""
    model, species_names = sample_model
    return TreeClassifier(
        model=model,
        species_names=species_names,
        device='cpu',
        confidence_threshold=0.5
    )


@pytest.fixture
def sample_image():
    """Create sample image."""
    return Image.new('RGB', (224, 224), color='green')


def test_classifier_initialization(classifier):
    """Test classifier initialization."""
    assert classifier.num_classes == 5
    assert len(classifier.species_names) == 5
    assert classifier.device.type == 'cpu'


def test_predict(classifier, sample_image):
    """Test prediction."""
    result = classifier.predict(sample_image)

    assert isinstance(result, PredictionResult)
    assert result.species in classifier.species_names
    assert 0 <= result.confidence <= 1
    assert len(result.all_predictions) <= classifier.top_k


def test_predict_with_part_type(classifier, sample_image):
    """Test prediction with specified part type."""
    result = classifier.predict(sample_image, part_type=TreePartType.LEAF)

    assert result.part_type == TreePartType.LEAF


def test_predict_batch(classifier, sample_image):
    """Test batch prediction."""
    images = [sample_image, sample_image, sample_image]
    results = classifier.predict_batch(images)

    assert len(results) == 3
    assert all(isinstance(r, PredictionResult) for r in results)


def test_get_species_info(classifier):
    """Test species info retrieval."""
    info = classifier.get_species_info('Oak')
    assert info is not None
    assert 'name' in info
