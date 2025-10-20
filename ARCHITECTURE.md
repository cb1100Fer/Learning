# Tree Species Identifier - Architecture Documentation

## Overview

The Tree Species Identifier is a modular deep learning framework designed to classify tree species from images of different tree parts (leaves, bark, and seeds). The architecture follows clean separation of concerns with distinct layers for data processing, model inference, and API serving.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FastAPI Application (REST Endpoints)               │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   Service Layer                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  TreeClassifier (Orchestration)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼────┐  ┌──────▼──────┐  ┌────▼────────┐
│   Image    │  │   Feature   │  │   Model     │
│ Processor  │  │  Extractor  │  │  Interface  │
└────────────┘  └─────────────┘  └─────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │      Data & Utils Layer        │
        │  ┌──────────┐  ┌───────────┐  │
        │  │ Dataset  │  │   Config  │  │
        │  └──────────┘  └───────────┘  │
        └────────────────────────────────┘
```

## Layer Descriptions

### 1. API Layer

**Purpose**: Provides HTTP endpoints for client applications

**Components**:
- `src/api/app.py`: FastAPI application with endpoints for prediction
- `src/api/routes.py`: Route definitions (if separated)

**Key Endpoints**:
- `POST /predict`: Single image prediction
- `POST /predict/batch`: Batch prediction
- `GET /health`: Health check
- `GET /model/info`: Model metadata

**Responsibilities**:
- Request validation
- File upload handling
- Response formatting
- Error handling
- CORS configuration

### 2. Service Layer

**Purpose**: Orchestrates the prediction pipeline

**Components**:
- `src/core/classifier.py`: TreeClassifier class

**Key Classes**:
- `TreeClassifier`: Main orchestrator
- `PredictionResult`: Result container

**Responsibilities**:
- Pipeline coordination
- Ensemble prediction
- Confidence thresholding
- Result aggregation

### 3. Processing Layer

**Purpose**: Handles image and feature processing

**Components**:

#### ImageProcessor (`src/core/image_processor.py`)
- Image loading and validation
- Preprocessing (resize, normalize)
- Data augmentation
- Image type detection (leaf/bark/seed)
- Enhancement techniques

#### FeatureExtractor (`src/core/feature_extractor.py`)
- Feature extraction from CNN backbone
- Multi-scale feature extraction
- Feature combination from multiple tree parts
- Attention map generation

**Responsibilities**:
- Image transformation
- Feature engineering
- Quality enhancement

### 4. Model Layer

**Purpose**: Defines and implements ML models

**Components**:

#### BaseModel (`src/models/base_model.py`)
- Abstract interface for all models
- Common methods (save, load, freeze)
- Parameter counting
- Model information

#### CNNModel (`src/models/cnn_model.py`)
- Transfer learning implementations
- Supported architectures:
  - ResNet (50, 101)
  - EfficientNet (B0, B4)
  - DenseNet (121)
  - MobileNet V3
- Classifier head customization

#### MultiInputCNNModel
- Processes different tree parts
- Shared or separate backbones
- Feature fusion strategies

**Responsibilities**:
- Forward pass implementation
- Weight management
- Architecture definition

### 5. Data Layer

**Purpose**: Handles dataset loading and augmentation

**Components**:

#### TreeDataset (`src/data/dataset.py`)
- PyTorch Dataset implementation
- Directory structure parsing
- Metadata tracking
- Class and part distribution

#### Augmentation (`src/data/augmentation.py`)
- Part-specific augmentation:
  - LeafSpecificAugmentation
  - BarkSpecificAugmentation
  - SeedSpecificAugmentation
- Advanced techniques:
  - Mixup
  - Cutout

**Expected Data Structure**:
```
data/
├── train/
│   ├── species_1/
│   │   ├── leaf/
│   │   ├── bark/
│   │   └── seed/
│   └── species_2/
│       └── ...
└── val/
    └── ...
```

### 6. Utilities Layer

**Purpose**: Cross-cutting concerns

**Components**:

#### Config (`src/utils/config.py`)
- YAML configuration loading
- Environment-based settings
- Configuration validation

#### Logger (`src/utils/logger.py`)
- Structured logging
- File and console output
- Log rotation

## Data Flow

### Training Flow

```
Raw Images → TreeDataset → DataLoader → Model → Loss → Optimizer → Weights
                ↓
          Augmentation
```

### Inference Flow

```
Input Image → ImageProcessor → Preprocessed Tensor
                                      ↓
                              Model (Forward Pass)
                                      ↓
                                   Logits
                                      ↓
                              TreeClassifier
                                      ↓
                              PredictionResult
```

### Ensemble Inference Flow

```
{Leaf, Bark, Seed} Images → ImageProcessor (each)
                                      ↓
                            Feature Extraction (each)
                                      ↓
                            Weighted Combination
                                      ↓
                                Final Logits
                                      ↓
                            PredictionResult
```

## Design Patterns

### 1. Strategy Pattern
- Different augmentation strategies for different tree parts
- Multiple model architectures through common interface

### 2. Template Method Pattern
- `BaseModel` defines the interface
- Subclasses implement specific architectures

### 3. Factory Pattern
- Model creation based on configuration
- Dataset creation with different splits

### 4. Facade Pattern
- `TreeClassifier` provides simple interface to complex subsystems

### 5. Dependency Injection
- Components receive dependencies through constructors
- Easy testing with mocks

## Configuration Management

Two main configuration files:

### model_config.yaml
- Image preprocessing settings
- Model architecture parameters
- Training hyperparameters
- Augmentation techniques
- Feature weights

### app_config.yaml
- API settings (host, port)
- Path configurations
- Logging settings
- Database configuration
- Caching options

## Extension Points

### Adding New Models

1. Create new class inheriting from `BaseModel`
2. Implement `forward()` and `get_feature_extractor()`
3. Register in model factory

```python
class CustomModel(BaseModel):
    def forward(self, x):
        # Implementation
        pass

    def get_feature_extractor(self):
        # Implementation
        pass
```

### Adding New Augmentation

1. Create augmentation class
2. Implement `__call__()` method
3. Register in augmentation pipeline

```python
class CustomAugmentation:
    def __call__(self, image):
        # Implementation
        return augmented_image
```

### Adding New API Endpoints

1. Add route to `app.py`
2. Define Pydantic models for request/response
3. Implement handler logic

```python
@app.post("/custom-endpoint")
async def custom_endpoint(data: CustomRequest):
    # Implementation
    return CustomResponse(...)
```

## Testing Strategy

### Unit Tests
- Individual components in isolation
- Mock dependencies
- Test files in `tests/`

### Integration Tests
- End-to-end pipeline
- API endpoint testing
- Database integration

### Performance Tests
- Inference latency
- Throughput benchmarks
- Memory usage

## Deployment Considerations

### Docker Containerization
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
CMD ["python", "-m", "src.api.app"]
```

### Model Serving Options
1. **FastAPI**: Current implementation
2. **TorchServe**: PyTorch native serving
3. **ONNX Runtime**: Cross-platform inference
4. **TensorFlow Serving**: If converted to TF

### Scaling Strategies
- Horizontal scaling with load balancer
- Model parallelism for large models
- Batch processing for throughput
- Caching frequent predictions

## Security Considerations

- Input validation (file type, size)
- Rate limiting
- API key authentication
- HTTPS encryption
- Model versioning and rollback

## Performance Optimization

### Inference Optimization
- Model quantization (INT8)
- TorchScript compilation
- ONNX export
- Batch inference

### Caching
- Prediction caching
- Feature caching
- Model weight caching

### Resource Management
- GPU memory management
- Connection pooling
- Async processing

## Monitoring and Observability

### Metrics to Track
- Prediction latency
- Throughput (requests/sec)
- Model accuracy
- Error rates
- Resource utilization

### Logging
- Request/response logging
- Error tracking
- Performance metrics
- Audit trails

## Future Enhancements

1. **Vision Transformers**: Add ViT support
2. **Active Learning**: Continuous model improvement
3. **Explainability**: Grad-CAM visualization
4. **Mobile Deployment**: TFLite/CoreML export
5. **Federated Learning**: Privacy-preserving training
6. **Multi-modal**: Combine images with metadata
7. **Real-time Video**: Live tree identification
