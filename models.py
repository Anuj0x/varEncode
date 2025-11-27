"""Modern VAE and DVAE models - consolidated and simplified."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import List, Tuple, Optional
import hydra
from omegaconf import DictConfig


class Encoder(nn.Module):
    """Convolutional encoder with batch normalization."""

    def __init__(self, input_channels: int, channels: List[int], hidden_dims: List[int],
                 latent_dim: int):
        super().__init__()

        # Convolutional layers
        layers = []
        in_ch = input_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # Calculate flattened size after convolutions
        # Assuming 28x28 input for MNIST, 32x32 for CIFAR-10
        test_input = torch.randn(1, input_channels, 28 if len(channels) == 2 else 32, 28 if len(channels) == 2 else 32)
        with torch.no_grad():
            conv_output = self.encoder(test_input)
        self.flattened_size = conv_output.numel() // conv_output.size(0)

        # Fully connected layers
        fc_layers = []
        in_dim = self.flattened_size
        for out_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = out_dim

        # Output layers for mean and log variance
        self.fc_mean = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Get mean and log variance
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    """Convolutional decoder with batch normalization."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], channels: List[int],
                 output_channels: int):
        super().__init__()

        # Fully connected layers
        fc_layers = []
        in_dim = latent_dim
        for out_dim in reversed(hidden_dims):
            fc_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = out_dim

        # Calculate spatial dimensions after deconvolutions
        # Reverse the encoder's convolution operations
        h, w = 7, 7  # After 2 conv layers with stride 2 on 28x28
        if len(channels) > 2:  # CIFAR-10 case
            h, w = 8, 8  # After 2 conv layers with stride 2 on 32x32

        self.fc_out = nn.Linear(in_dim, channels[0] * h * w)
        self.decoder_input_ch = channels[0]
        self.decoder_h, self.decoder_w = h, w

        # Deconvolutional layers
        layers = []
        in_ch = channels[0]
        for out_ch in reversed(channels):
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            in_ch = out_ch

        # Final layer to output channels (1 for MNIST, 3 for CIFAR-10)
        layers.append(nn.ConvTranspose2d(in_ch, output_channels, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_out(z)
        x = x.view(x.size(0), self.decoder_input_ch, self.decoder_h, self.decoder_w)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    """Modern Variational Autoencoder."""

    def __init__(self, input_channels: int, latent_dim: int,
                 channels: List[int], hidden_dims: List[int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_channels, channels, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, channels, input_channels)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor, samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        mean, logvar = self.encoder(x)

        # Sample from posterior
        z_samples = []
        for _ in range(samples):
            z = self.reparameterize(mean, logvar)
            z_samples.append(z)
        z = torch.stack(z_samples, dim=0).mean(dim=0)  # Average samples

        # Decode
        reconstruction = self.decoder(z)

        # Compute KL divergence
        p_z = Normal(0, 1)
        q_z = Normal(mean, torch.exp(0.5 * logvar))
        kl_div = kl_divergence(q_z, p_z).sum(dim=1).mean()

        return reconstruction, kl_div, z

    def sample(self, num_samples: int = 1, device: str = 'cpu') -> torch.Tensor:
        """Sample from prior distribution."""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return torch.sigmoid(samples) if samples.shape[1] == 1 else samples  # Sigmoid for MNIST


class DVAE(VAE):
    """Denoising Variational Autoencoder."""

    def __init__(self, input_channels: int, latent_dim: int,
                 channels: List[int], hidden_dims: List[int]):
        super().__init__(input_channels, latent_dim, channels, hidden_dims)
        # Additional encoder for denoising (same architecture as recognition)
        self.denoising_encoder = Encoder(input_channels, channels, hidden_dims, latent_dim)

    def forward(self, x: torch.Tensor, samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Standard VAE forward pass
        reconstruction, kl_div, z = super().forward(x, samples)

        # Denoising step: treat reconstruction as corrupted input
        mean_denoise, logvar_denoise = self.denoising_encoder(reconstruction.detach())
        z_denoise = self.reparameterize(mean_denoise, logvar_denoise)
        reconstruction_denoise = self.decoder(z_denoise)

        # Additional KL divergence for denoising
        p_z_denoise = Normal(0, 1)
        q_z_denoise = Normal(mean_denoise, torch.exp(0.5 * logvar_denoise))
        kl_div_denoise = kl_divergence(q_z_denoise, p_z_denoise).sum(dim=1).mean()

        # Combine losses
        total_kl_div = kl_div + kl_div_denoise

        return reconstruction_denoise, total_kl_div, z


def create_model(config: DictConfig) -> nn.Module:
    """Factory function to create VAE or DVAE based on config."""
    input_channels = 1 if config.data.dataset == "mnist" else 3

    if config.model.denoising:
        return DVAE(
            input_channels=input_channels,
            latent_dim=config.model.latent_dim,
            channels=config.model.channels,
            hidden_dims=config.model.hidden_dims
        )
    else:
        return VAE(
            input_channels=input_channels,
            latent_dim=config.model.latent_dim,
            channels=config.model.channels,
            hidden_dims=config.model.hidden_dims
        )


@hydra.main(config_path=".", config_name="config", version_base=None)
def test_model(cfg: DictConfig):
    """Test model creation and forward pass."""
    model = create_model(cfg)

    # Create test input
    input_channels = 1 if cfg.data.dataset == "mnist" else 3
    height = 28 if cfg.data.dataset == "mnist" else 32
    test_input = torch.randn(2, input_channels, height, height)

    # Forward pass
    reconstruction, kl_div, z = model(test_input, samples=cfg.training.samples)

    print(f"Model: {'DVAE' if cfg.model.denoising else 'VAE'}")
    print(f"Input shape: {test_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"KL divergence: {kl_div:.4f}")

    # Test sampling
    samples = model.sample(4)
    print(f"Generated samples shape: {samples.shape}")


if __name__ == "__main__":
    test_model()
