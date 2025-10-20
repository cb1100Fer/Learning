"""Tests for image processor module."""

import pytest
import torch
import numpy as np
from PIL import Image

from src.core.image_processor import ImageProcessor


@pytest.fixture
def image_processor():
    """Create ImageProcessor instance."""
    return ImageProcessor()


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    return Image.new('RGB', (256, 256), color='red')


def test_image_processor_initialization(image_processor):
    """Test ImageProcessor initialization."""
    assert image_processor.input_size == (224, 224)
    assert len(image_processor.mean) == 3
    assert len(image_processor.std) == 3


def test_load_image(image_processor, tmp_path):
    """Test image loading."""
    # Create temporary image
    img = Image.new('RGB', (100, 100), color='blue')
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    # Load image
    loaded_img = image_processor.load_image(img_path)
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.mode == 'RGB'


def test_preprocess_image(image_processor, sample_image):
    """Test image preprocessing."""
    tensor = image_processor.preprocess_image(sample_image, augment=False)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_batch(image_processor, sample_image):
    """Test batch preprocessing."""
    images = [sample_image, sample_image, sample_image]
    batch = image_processor.preprocess_batch(images, augment=False)

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (3, 3, 224, 224)


def test_denormalize(image_processor, sample_image):
    """Test denormalization."""
    tensor = image_processor.preprocess_image(sample_image)
    denorm = image_processor.denormalize(tensor)

    assert isinstance(denorm, np.ndarray)
    assert denorm.shape == (224, 224, 3)
    assert denorm.dtype == np.uint8


def test_detect_image_type(image_processor, sample_image):
    """Test image type detection."""
    img_type = image_processor.detect_image_type(sample_image)
    assert img_type in ['leaf', 'bark', 'seed']


def test_enhance_image(image_processor, sample_image):
    """Test image enhancement."""
    enhanced = image_processor.enhance_image(sample_image)
    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == sample_image.size
