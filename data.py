"""Modern data loading for DVAE - consolidated from multiple files."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import lightning as L
from typing import Tuple, Optional
import hydra
from omegaconf import DictConfig


class DataModule(L.LightningDataModule):
    """Modern PyTorch Lightning data module for MNIST and CIFAR-10."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.transform = self._get_transforms()

    def _get_transforms(self) -> transforms.Compose:
        """Get appropriate transforms based on dataset."""
        if self.config.data.dataset == "mnist":
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.config.data.dataset == "cifar10":
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError(f"Unknown dataset: {self.config.data.dataset}")

    def prepare_data(self):
        """Download data if needed."""
        if self.config.data.dataset == "mnist":
            MNIST(self.config.data.data_dir, train=True, download=True)
            MNIST(self.config.data.data_dir, train=False, download=True)
        elif self.config.data.dataset == "cifar10":
            CIFAR10(self.config.data.data_dir, train=True, download=True)
            CIFAR10(self.config.data.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        if self.config.data.dataset == "mnist":
            dataset_class = MNIST
        elif self.config.data.dataset == "cifar10":
            dataset_class = CIFAR10
        else:
            raise ValueError(f"Unknown dataset: {self.config.data.dataset}")

        # Load full training set
        full_train_dataset = dataset_class(
            self.config.data.data_dir,
            train=True,
            transform=self.transform
        )

        # Split into train/validation
        val_size = int(len(full_train_dataset) * self.config.data.validation_split)
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Load test set
        self.test_dataset = dataset_class(
            self.config.data.data_dir,
            train=False,
            transform=self.transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )


@hydra.main(config_path=".", config_name="config", version_base=None)
def test_data_loading(cfg: DictConfig):
    """Test function to verify data loading works."""
    dm = DataModule(cfg)
    dm.prepare_data()
    dm.setup()

    # Test a few batches
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    x, y = batch

    print(f"Dataset: {cfg.data.dataset}")
    print(f"Batch shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Data range: [{x.min():.3f}, {x.max():.3f}]")


if __name__ == "__main__":
    test_data_loading()
