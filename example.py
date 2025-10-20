"""
Example script demonstrating basic usage of the tree species identifier.

This script shows how to:
1. Initialize a model
2. Make predictions
3. Use the classifier
"""

from pathlib import Path
from src.core.classifier import TreeClassifier
from src.core.feature_extractor import TreePartType
from src.models.cnn_model import CNNModel


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===\n")

    # Define species (in production, load from file)
    species_names = ['Oak', 'Maple', 'Pine', 'Birch', 'Elm']

    # Initialize model
    print("Initializing model...")
    model = CNNModel(
        num_classes=len(species_names),
        architecture='resnet50',
        pretrained=True  # Use pretrained weights
    )

    # Initialize classifier
    print("Initializing classifier...")
    classifier = TreeClassifier(
        model=model,
        species_names=species_names,
        device='cpu'  # Use 'cuda' if GPU available
    )

    print(f"Classifier ready on device: {classifier.device}")
    print(f"Number of species: {classifier.num_classes}")
    print(f"Confidence threshold: {classifier.confidence_threshold}\n")

    # Example prediction (would need actual image)
    # result = classifier.predict('path/to/image.jpg')
    # print(f"Species: {result.species}")
    # print(f"Confidence: {result.confidence:.2%}")


def example_ensemble_prediction():
    """Ensemble prediction example."""
    print("\n=== Ensemble Prediction Example ===\n")

    species_names = ['Oak', 'Maple', 'Pine']
    model = CNNModel(num_classes=len(species_names), architecture='resnet50')
    classifier = TreeClassifier(model=model, species_names=species_names)

    # In practice, you would have actual images
    # result = classifier.predict_with_ensemble({
    #     TreePartType.LEAF: 'path/to/leaf.jpg',
    #     TreePartType.BARK: 'path/to/bark.jpg',
    #     TreePartType.SEED: 'path/to/seed.jpg'
    # })

    print("Ensemble prediction combines multiple tree parts for higher accuracy")
    print("Example usage:")
    print("""
    result = classifier.predict_with_ensemble({
        TreePartType.LEAF: 'path/to/leaf.jpg',
        TreePartType.BARK: 'path/to/bark.jpg',
        TreePartType.SEED: 'path/to/seed.jpg'
    })
    """)


def example_model_info():
    """Display model information."""
    print("\n=== Model Information Example ===\n")

    model = CNNModel(
        num_classes=10,
        architecture='efficientnet_b0',
        pretrained=True
    )

    info = model.get_model_info()
    print(f"Model Type: {info['model_type']}")
    print(f"Number of Classes: {info['num_classes']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")


if __name__ == "__main__":
    print("Tree Species Identifier - Example Usage\n")
    print("=" * 50)

    try:
        example_basic_usage()
        example_ensemble_prediction()
        example_model_info()

        print("\n" + "=" * 50)
        print("\nFor more examples, see notebooks/exploration.ipynb")
        print("To run the API server: python -m src.api.app")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This example requires PyTorch and other dependencies.")
        print("Install with: pip install -r requirements.txt")
