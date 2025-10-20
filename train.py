"""
Training script template for tree species identification model.

This script demonstrates a complete training pipeline including:
- Data loading
- Model initialization
- Training loop
- Validation
- Model saving
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.cnn_model import CNNModel
from src.data.dataset import TreeDataset
from src.utils.config import config
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def main(args):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = TreeDataset(
        data_dir=args.data_dir,
        split='train',
        part_type=args.part_type
    )

    val_dataset = TreeDataset(
        data_dir=args.data_dir,
        split='val',
        part_type=args.part_type
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    logger.info(f"Initializing model: {args.architecture}")
    model = CNNModel(
        num_classes=len(train_dataset.classes),
        architecture=args.architecture,
        pretrained=args.pretrained,
        dropout_rate=args.dropout
    )
    model = model.to(device)

    # Log model info
    info = model.get_model_info()
    logger.info(f"Total parameters: {info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step()

        # Log results
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = output_dir / "best_model.pth"
            model.save(model_path)
            logger.info(f"Saved best model to {model_path}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            model.save(checkpoint_path)

    logger.info(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tree species identification model")

    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--part-type', type=str, default=None,
                        choices=['leaf', 'bark', 'seed'],
                        help='Specific tree part to train on (None for all)')

    # Model
    parser.add_argument('--architecture', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet_b0',
                                 'efficientnet_b4', 'densenet121', 'mobilenet_v3_large'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer')

    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='models/weights',
                        help='Output directory for saved models')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
