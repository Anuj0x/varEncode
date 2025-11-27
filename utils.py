"""Utility functions for DVAE project."""

import torch
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_reconstructions(original: torch.Tensor, reconstruction: torch.Tensor,
                        filename: str, num_images: int = 8):
    """Save comparison of original vs reconstructed images."""
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        # Original
        orig_img = original[i].cpu().detach()
        if orig_img.shape[0] == 1:  # Grayscale
            orig_img = orig_img.squeeze(0)
        else:  # RGB
            orig_img = orig_img.permute(1, 2, 0)

        axes[0, i].imshow(orig_img, cmap='gray' if orig_img.ndim == 2 else None)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        # Reconstruction
        recon_img = reconstruction[i].cpu().detach()
        if recon_img.shape[0] == 1:  # Grayscale
            recon_img = torch.sigmoid(recon_img.squeeze(0))
        else:  # RGB
            recon_img = torch.clamp(recon_img.permute(1, 2, 0), 0, 1)

        axes[1, i].imshow(recon_img, cmap='gray' if recon_img.ndim == 2 else None)
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def save_samples(samples: torch.Tensor, filename: str, num_images: int = 16):
    """Save generated samples in a grid."""
    grid_size = int(np.sqrt(num_images))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

    for i in range(num_images):
        row, col = i // grid_size, i % grid_size
        img = samples[i].cpu().detach()

        if img.shape[0] == 1:  # Grayscale
            img = torch.sigmoid(img.squeeze(0))
            axes[row, col].imshow(img, cmap='gray')
        else:  # RGB
            img = torch.clamp(img.permute(1, 2, 0), 0, 1)
            axes[row, col].imshow(img)

        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def compute_classification_error(model: torch.nn.Module, classifier: torch.nn.Module,
                               data_loader: torch.utils.data.DataLoader,
                               device: str = 'cpu') -> float:
    """Compute classification error rate of generated samples."""
    model.eval()
    classifier.eval()

    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Generate reconstruction using the model
            reconstruction, _, _ = model(x)

            # Classify the reconstructed images
            # Note: This assumes the classifier expects the same input format
            predictions = classifier(reconstruction)
            predicted_labels = torch.argmax(predictions, dim=1)

            correct_predictions += (predicted_labels == y).sum().item()
            total_samples += y.size(0)

    error_rate = 1.0 - (correct_predictions / total_samples)
    return error_rate
