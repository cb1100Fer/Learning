"""Dataset class for tree species identification."""

from typing import Optional, Callable, List, Dict, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
from PIL import Image

from ..core.image_processor import ImageProcessor
from ..utils.logger import setup_logger


logger = setup_logger(__name__)


class TreeDataset(Dataset):
    """
    PyTorch Dataset for tree species images.

    Expected directory structure:
        data/
        ├── train/
        │   ├── species_1/
        │   │   ├── leaf/
        │   │   ├── bark/
        │   │   └── seed/
        │   ├── species_2/
        │   ...
        └── val/
            └── ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        part_type: Optional[str] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform for images
            target_transform: Optional transform for labels
            part_type: Specific part type to load ('leaf', 'bark', 'seed', None for all)
            image_processor: Optional ImageProcessor instance
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.part_type = part_type

        if image_processor is None:
            self.image_processor = ImageProcessor()
        else:
            self.image_processor = image_processor

        # Load dataset
        self.samples, self.classes = self._load_dataset()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        logger.info(
            f"TreeDataset loaded: {len(self.samples)} samples, "
            f"{len(self.classes)} classes, split={split}"
        )

    def _load_dataset(self) -> Tuple[List[Dict], List[str]]:
        """
        Load dataset from directory structure.

        Returns:
            Tuple of (samples list, class names list)
        """
        samples = []
        classes = set()

        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return [], []

        # Iterate through species directories
        for species_dir in sorted(self.data_dir.iterdir()):
            if not species_dir.is_dir():
                continue

            species_name = species_dir.name
            classes.add(species_name)

            # If part_type specified, only load that type
            if self.part_type:
                part_dirs = [species_dir / self.part_type]
            else:
                # Load all parts
                part_dirs = [
                    species_dir / part
                    for part in ['leaf', 'bark', 'seed']
                    if (species_dir / part).exists()
                ]

            # Collect image files
            for part_dir in part_dirs:
                if not part_dir.exists():
                    continue

                part_name = part_dir.name

                for img_path in part_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        samples.append({
                            'path': str(img_path),
                            'species': species_name,
                            'part_type': part_name
                        })

        return samples, sorted(list(classes))

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get dataset item.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, class index, metadata)
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['path']).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Use default image processor
            image = self.image_processor.preprocess_image(
                image,
                augment=(self.split == 'train')
            )

        # Get class index
        class_idx = self.class_to_idx[sample['species']]

        if self.target_transform:
            class_idx = self.target_transform(class_idx)

        # Metadata
        metadata = {
            'species': sample['species'],
            'part_type': sample['part_type'],
            'path': sample['path']
        }

        return image, class_idx, metadata

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution in the dataset.

        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = {}
        for sample in self.samples:
            species = sample['species']
            distribution[species] = distribution.get(species, 0) + 1

        return distribution

    def get_part_distribution(self) -> Dict[str, int]:
        """
        Get distribution of tree parts in the dataset.

        Returns:
            Dictionary mapping part types to counts
        """
        distribution = {}
        for sample in self.samples:
            part = sample['part_type']
            distribution[part] = distribution.get(part, 0) + 1

        return distribution

    def save_metadata(self, output_path: str) -> None:
        """
        Save dataset metadata to JSON file.

        Args:
            output_path: Path to save metadata
        """
        metadata = {
            'split': self.split,
            'num_samples': len(self.samples),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_distribution': self.get_class_distribution(),
            'part_distribution': self.get_part_distribution()
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset metadata saved to {output_path}")
