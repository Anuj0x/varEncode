"""Clean entry point for the modern DVAE project."""

import hydra
from omegaconf import DictConfig
import argparse
from pathlib import Path

from trainer import train
from data import DataModule
from models import create_model
from utils import set_seed, count_parameters, save_samples


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point with command line argument parsing."""

    # Set random seed for reproducibility
    set_seed(42)

    print("=" * 50)
    print("Modern DVAE - Denoising Variational Autoencoder")
    print("=" * 50)
    print(f"Dataset: {cfg.data.dataset.upper()}")
    print(f"Model: {'DVAE' if cfg.model.denoising else 'VAE'}")
    print(f"Latent dimension: {cfg.model.latent_dim}")
    print(f"Batch size: {cfg.data.batch_size}")
    print(f"Max epochs: {cfg.training.max_epochs}")
    print()

    # Create model to show parameter count
    model = create_model(cfg)
    num_params = count_parameters(model)
    print(",")

    # Run training
    train(cfg)


def sample(cfg: DictConfig):
    """Generate samples from a trained model."""
    import torch

    print("Loading model for sampling...")
    model = create_model(cfg)

    # Load checkpoint if available
    checkpoint_path = Path("./checkpoints") / f"{cfg.logging.project_name}-best.ckpt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found, using randomly initialized model")

    model.eval()

    # Generate samples
    print(f"Generating {cfg.get('num_samples', 16)} samples...")
    samples = model.sample(num_samples=cfg.get('num_samples', 16))

    # Save samples
    output_path = cfg.get('output_path', 'generated_samples.png')
    save_samples(samples, output_path)
    print(f"Samples saved to {output_path}")


def test(cfg: DictConfig):
    """Test a trained model."""
    import lightning as L
    from trainer import DVAELightningModule

    # Create model and data module
    model = DVAELightningModule(cfg)
    data_module = DataModule(cfg)

    # Create trainer
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
    )

    # Test the model
    trainer.test(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modern DVAE Training and Inference")
    parser.add_argument(
        "mode",
        choices=["train", "sample", "test"],
        help="Mode to run: train, sample, or test"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate (for sample mode)"
    )
    parser.add_argument(
        "--output_path",
        default="generated_samples.png",
        help="Output path for generated samples (for sample mode)"
    )

    args = parser.parse_args()

    # Override config with command line args
    @hydra.main(config_path=".", config_name="config", version_base=None)
    def run_with_args(cfg: DictConfig):
        cfg.num_samples = args.num_samples
        cfg.output_path = args.output_path

        if args.mode == "train":
            main(cfg)
        elif args.mode == "sample":
            sample(cfg)
        elif args.mode == "test":
            test(cfg)

    run_with_args()
