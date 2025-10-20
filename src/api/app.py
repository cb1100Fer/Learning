"""FastAPI application for tree species identification service."""

from typing import List, Optional
from pathlib import Path
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image

from ..core.classifier import TreeClassifier, PredictionResult
from ..core.feature_extractor import TreePartType
from ..models.cnn_model import CNNModel
from ..utils.config import config
from ..utils.logger import setup_logger


logger = setup_logger(__name__)


# Pydantic models for API
class PredictionResponse(BaseModel):
    """Response model for prediction."""
    species: str
    confidence: float
    top_predictions: List[dict]
    part_type: str
    is_confident: bool


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


class ModelInfo(BaseModel):
    """Model information response."""
    num_classes: int
    architecture: str
    device: str


# Global classifier instance (loaded on startup)
classifier: Optional[TreeClassifier] = None


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Tree Species Identifier API",
        description="API for identifying tree species from images of leaves, bark, and seeds",
        version="0.1.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        global classifier

        try:
            # Load species names (in production, load from a file)
            species_names = _load_species_names()

            # Initialize model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_config = config.model_config.get('model', {})

            model = CNNModel(
                num_classes=len(species_names),
                architecture=model_config.get('name', 'resnet50'),
                pretrained=model_config.get('pretrained', True)
            )

            # Load weights if available
            model_path = config.get('paths.models_dir', 'models/weights') + '/best_model.pth'
            if Path(model_path).exists():
                model.load(model_path)
                logger.info(f"Loaded model weights from {model_path}")

            # Initialize classifier
            classifier = TreeClassifier(
                model=model,
                species_names=species_names,
                device=device
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Don't fail startup, but classifier will be None

    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint."""
        return {
            "message": "Tree Species Identifier API",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "predict_batch": "/predict/batch",
                "model_info": "/model/info"
            }
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if classifier is not None else "unhealthy",
            model_loaded=classifier is not None,
            device=str(classifier.device) if classifier else "N/A"
        )

    @app.get("/model/info", response_model=ModelInfo)
    async def get_model_info():
        """Get model information."""
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return ModelInfo(
            num_classes=classifier.num_classes,
            architecture="CNN-based",
            device=str(classifier.device)
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        part_type: Optional[str] = Query(None, description="Tree part type: leaf, bark, or seed"),
        top_k: Optional[int] = Query(5, description="Number of top predictions to return")
    ):
        """
        Predict tree species from an uploaded image.

        Args:
            file: Uploaded image file
            part_type: Optional tree part type (auto-detected if not provided)
            top_k: Number of top predictions to return

        Returns:
            Prediction results
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            # Read image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Parse part type
            tree_part = None
            if part_type:
                try:
                    tree_part = TreePartType(part_type.lower())
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid part_type. Must be one of: leaf, bark, seed"
                    )

            # Predict
            result = classifier.predict(
                image=image,
                part_type=tree_part,
                return_top_k=top_k
            )

            # Format response
            return PredictionResponse(
                species=result.species,
                confidence=result.confidence,
                top_predictions=[
                    {"species": species, "confidence": conf}
                    for species, conf in result.all_predictions
                ],
                part_type=result.part_type.value,
                is_confident=result.metadata.get('is_confident', False)
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch", response_model=List[PredictionResponse])
    async def predict_batch(
        files: List[UploadFile] = File(...),
        top_k: Optional[int] = Query(5, description="Number of top predictions")
    ):
        """
        Predict tree species for multiple images.

        Args:
            files: List of uploaded image files
            top_k: Number of top predictions to return

        Returns:
            List of prediction results
        """
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Limit batch size
        max_batch_size = config.get('api.max_batch_size', 10)
        if len(files) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of {max_batch_size}"
            )

        results = []

        for file in files:
            try:
                # Process each image
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                result = classifier.predict(image=image, return_top_k=top_k)

                results.append(PredictionResponse(
                    species=result.species,
                    confidence=result.confidence,
                    top_predictions=[
                        {"species": species, "confidence": conf}
                        for species, conf in result.all_predictions
                    ],
                    part_type=result.part_type.value,
                    is_confident=result.metadata.get('is_confident', False)
                ))

            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                # Continue with other files

        return results

    return app


def _load_species_names() -> List[str]:
    """
    Load species names from file or config.

    Returns:
        List of species names
    """
    # In production, load from a file
    # For now, return a placeholder list
    species_path = Path(config.get('paths.data_dir', 'data')) / 'species_names.txt'

    if species_path.exists():
        with open(species_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    # Default placeholder species
    logger.warning("Species names file not found, using placeholders")
    return [f"Species_{i}" for i in range(100)]


# For direct execution
if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=config.get('api.host', '0.0.0.0'),
        port=config.get('api.port', 8000),
        reload=config.get('api.reload', False)
    )
