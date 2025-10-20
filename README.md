# Tree Species Identifier

A comprehensive framework for identifying tree species from images of leaves, bark, and seeds using deep learning.

## Features

- **Multi-Input Support**: Identify trees from images of leaves, bark, or seeds
- **State-of-the-Art Models**: Supports multiple CNN architectures (ResNet, EfficientNet, DenseNet, etc.)
- **Advanced Augmentation**: Tree-specific data augmentation techniques
- **Ensemble Predictions**: Combine predictions from multiple tree parts for higher accuracy
- **REST API**: FastAPI-based service for easy integration
- **Modular Design**: Clean, extensible architecture for research and production

## Architecture

```
tree-species-identifier/
├── src/
│   ├── core/               # Core modules
│   │   ├── image_processor.py    # Image preprocessing
│   │   ├── feature_extractor.py  # Feature extraction
│   │   └── classifier.py         # Classification pipeline
│   ├── models/             # Model implementations
│   │   ├── base_model.py         # Abstract base class
│   │   └── cnn_model.py          # CNN implementations
│   ├── api/                # REST API
│   │   └── app.py
│   ├── data/               # Dataset and augmentation
│   │   ├── dataset.py
│   │   └── augmentation.py
│   └── utils/              # Utilities
│       ├── config.py
│       └── logger.py
├── config/                 # Configuration files
├── tests/                  # Unit tests
└── notebooks/              # Jupyter notebooks
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tree-species-identifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Quick Start

#### 1. Prepare Your Dataset

Organize your data in the following structure:

```
data/
├── train/
│   ├── oak/
│   │   ├── leaf/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   ├── bark/
│   │   └── seed/
│   ├── maple/
│   │   └── ...
│   └── ...
└── val/
    └── ...
```

#### 2. Train a Model

```python
from src.models.cnn_model import CNNModel
from src.data.dataset import TreeDataset
from torch.utils.data import DataLoader
import torch

# Load dataset
train_dataset = TreeDataset(data_dir='data', split='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
model = CNNModel(
    num_classes=len(train_dataset.classes),
    architecture='resnet50',
    pretrained=True
)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for images, labels, _ in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Save model
model.save('models/weights/my_model.pth')
```

#### 3. Make Predictions

```python
from src.core.classifier import TreeClassifier
from src.models.cnn_model import CNNModel

# Load model
species_names = ['Oak', 'Maple', 'Pine', 'Birch']  # Your species
model = CNNModel(num_classes=len(species_names), architecture='resnet50')
model.load('models/weights/my_model.pth')

# Initialize classifier
classifier = TreeClassifier(model=model, species_names=species_names)

# Predict
result = classifier.predict('path/to/leaf_image.jpg')
print(f"Species: {result.species}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Top 5 predictions: {result.all_predictions}")
```

#### 4. Run the API Server

```bash
python -m src.api.app
```

The API will be available at `http://localhost:8000`

**API Endpoints:**

- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Predict tree species from an image
- `POST /predict/batch` - Batch prediction

**Example API Usage:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf_image.jpg" \
  -F "part_type=leaf"
```

### Advanced Features

#### Ensemble Prediction

Combine predictions from multiple tree parts:

```python
from src.core.feature_extractor import TreePartType

result = classifier.predict_with_ensemble({
    TreePartType.LEAF: 'path/to/leaf.jpg',
    TreePartType.BARK: 'path/to/bark.jpg',
    TreePartType.SEED: 'path/to/seed.jpg'
})
```

#### Custom Augmentation

```python
from src.data.augmentation import LeafSpecificAugmentation

leaf_aug = LeafSpecificAugmentation(image_size=224)
augmented_image = leaf_aug(original_image)
```

#### Multi-Input Model

```python
from src.models.cnn_model import MultiInputCNNModel

# Model with separate backbones for each tree part
model = MultiInputCNNModel(
    num_classes=100,
    architecture='resnet50',
    shared_backbone=False
)
```

## Configuration

Edit configuration files in `config/`:

- `model_config.yaml` - Model architecture, training parameters, augmentation
- `app_config.yaml` - API settings, paths, logging

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_classifier.py
```

## Project Structure Details

### Core Modules

- **ImageProcessor** (`src/core/image_processor.py`): Handles image loading, preprocessing, normalization, and augmentation
- **FeatureExtractor** (`src/core/feature_extractor.py`): Extracts features from images and combines multi-part features
- **TreeClassifier** (`src/core/classifier.py`): Main classification pipeline orchestrating the entire prediction process

### Models

- **BaseModel** (`src/models/base_model.py`): Abstract base class defining the model interface
- **CNNModel** (`src/models/cnn_model.py`): CNN-based implementations with transfer learning
- **MultiInputCNNModel**: Specialized model for processing different tree parts

### Data

- **TreeDataset** (`src/data/dataset.py`): PyTorch Dataset for loading tree images
- **Augmentation** (`src/data/augmentation.py`): Various augmentation strategies including tree-specific techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- [ ] Vision Transformer support
- [ ] Mobile-optimized models (MobileNet, EfficientNet-Lite)
- [ ] Active learning pipeline
- [ ] Species database integration
- [ ] Mobile app (iOS/Android)
- [ ] Web interface
- [ ] ONNX export for deployment
- [ ] TensorBoard integration for training visualization

## License

This project is licensed under the MIT License.

## Acknowledgments

- PyTorch team for the deep learning framework
- torchvision for pretrained models
- FastAPI for the API framework

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tree_species_identifier,
  title = {Tree Species Identifier},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/tree-species-identifier}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
