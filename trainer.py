"""Modern PyTorch Lightning trainer for DVAE - consolidated training logic."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb
from typing import Dict, Any, Optional
import hydra
from omegaconf import DictConfig

from data import DataModule
from models import create_model


class DVAELightningModule(L.LightningModule):
    """PyTorch Lightning module for VAE/DVAE training."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Create model
        self.model = create_model(config)

        # Loss weights
        self.reconstruction_weight = 1.0
        self.kl_weight = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, samples=self.config.training.samples)[0]

    def _compute_loss(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reconstruction and KL divergence losses."""
        x, _ = batch  # We don't use labels for unsupervised learning

        reconstruction, kl_div, z = self.model(x, samples=self.config.training.samples)

        # Reconstruction loss (MSE for simplicity, could use BCE for binary data)
        if self.config.data.dataset == "mnist":
            # For MNIST, use BCE loss (original paper approach)
            reconstruction_loss = F.binary_cross_entropy_with_logits(
                reconstruction, x, reduction='mean'
            )
        else:
            # For CIFAR-10, use MSE
            reconstruction_loss = F.mse_loss(reconstruction, x, reduction='mean')

        # Total loss
        total_loss = self.reconstruction_weight * reconstruction_loss + self.kl_weight * kl_div

        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_div': kl_div,
            'reconstruction': reconstruction,
            'z': z
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch)

        # Log metrics
        self.log('train_loss', losses['loss'], prog_bar=True)
        self.log('train_reconstruction_loss', losses['reconstruction_loss'])
        self.log('train_kl_div', losses['kl_div'])

        return losses['loss']

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        losses = self._compute_loss(batch)

        # Log metrics
        self.log('val_loss', losses['loss'], prog_bar=True)
        self.log('val_reconstruction_loss', losses['reconstruction_loss'])
        self.log('val_kl_div', losses['kl_div'])

        return losses

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        losses = self._compute_loss(batch)

        # Log metrics
        self.log('test_loss', losses['loss'])
        self.log('test_reconstruction_loss', losses['reconstruction_loss'])
        self.log('test_kl_div', losses['kl_div'])

        return losses

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.config.training.lr_scheduler.decay_factor ** (1.0 / self.config.training.lr_scheduler.decay_step)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_end(self):
        """Generate samples at the end of each epoch for monitoring."""
        if self.current_epoch % 10 == 0:  # Generate samples every 10 epochs
            samples = self.model.sample(num_samples=16, device=self.device)
            samples = torch.sigmoid(samples) if samples.shape[1] == 1 else samples

            # Log samples to wandb if available
            if isinstance(self.logger, WandbLogger):
                # Create a grid of samples
                grid = torch.clamp(samples, 0, 1)
                grid = torch.cat([grid[i:i+4] for i in range(0, 16, 4)], dim=2)
                grid = torch.cat([grid[:, j:j+4] for j in range(0, grid.shape[2], 4)], dim=3)
                grid = grid.squeeze(1) if grid.shape[1] == 1 else grid.permute(0, 2, 3, 1)

                self.logger.experiment.log({
                    f"samples_epoch_{self.current_epoch}": wandb.Image(grid[0].cpu().numpy())
                })


def create_trainer(config: DictConfig) -> L.Trainer:
    """Create PyTorch Lightning trainer with callbacks and logging."""

    # Callbacks
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=config.logging.monitor_metric,
        dirpath="./checkpoints",
        filename=f"{config.logging.project_name}-{{epoch:02d}}-{{{config.logging.monitor_metric}:.2f}}",
        save_top_k=config.logging.save_top_k,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Logger
    logger = WandbLogger(
        project=config.logging.project_name,
        name=config.logging.run_name,
        log_model=True
    ) if config.logging.project_name else None

    # Trainer
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        log_every_n_steps=config.logging.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    return trainer


@hydra.main(config_path=".", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Main training function."""
    # Set random seeds for reproducibility
    L.seed_everything(42)
    torch.manual_seed(42)

    # Create data module
    data_module = DataModule(cfg)

    # Create model
    model = DVAELightningModule(cfg)

    # Create trainer
    trainer = create_trainer(cfg)

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
