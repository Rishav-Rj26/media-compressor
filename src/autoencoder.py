"""
autoencoder.py – Convolutional Autoencoder for Learned Image Compression

Improvements over baseline:
  - Combined loss: L1 + SSIM-based perceptual loss (not just MSE)
  - Learning rate scheduling (cosine annealing)
  - Deeper architecture option (5-layer encoder/decoder)
  - Better latent quantization estimation

Architecture (standard):
    Encoder: Conv2d→BN→ReLU(64) → Conv2d→BN→ReLU(32) → Conv2d→BN→ReLU(bottleneck)
    Decoder: ConvT2d→BN→ReLU(32) → ConvT2d→BN→ReLU(64) → ConvT2d→Sigmoid(3ch)

Architecture (deep):
    Encoder: 3→64→128→64→32→bottleneck  (spatial: H/2→H/4→H/8→H/16→H/32)
    Decoder: bottleneck→32→64→128→64→3
"""

import os
import io
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Loss Function ────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combined L1 + SSIM loss for perceptual quality.
    
    SSIM loss encourages structural similarity preservation while
    L1 loss ensures pixel-level accuracy. This produces visually
    better results than pure MSE.
    """

    def __init__(self, alpha: float = 0.84, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.window_size = window_size

    def _gaussian_window(self, channels: int, device) -> torch.Tensor:
        """Create a Gaussian window for SSIM computation."""
        sigma = 1.5
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device)
        coords -= self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        return window

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two batches of images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        channels = x.shape[1]
        window = self._gaussian_window(channels, x.device)
        pad = self.window_size // 2

        mu_x = torch.nn.functional.conv2d(x, window, padding=pad, groups=channels)
        mu_y = torch.nn.functional.conv2d(y, window, padding=pad, groups=channels)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = torch.nn.functional.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = torch.nn.functional.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
        sigma_xy = torch.nn.functional.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1 - self._ssim(output, target)
        l1_loss = self.l1(output, target)
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


# ── Model Definitions ────────────────────────────────────────────────────────

class ImageAutoencoder(nn.Module):
    """Standard convolutional autoencoder — 3-layer encoder/decoder."""

    def __init__(self, bottleneck_channels: int = 8):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.architecture = "standard"

        # Encoder: 3 → 64 → 32 → bottleneck  (spatial: H → H/2 → H/4 → H/8)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder: bottleneck → 32 → 64 → 3  (spatial: H/8 → H/4 → H/2 → H)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # pixel values in [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """Return the compressed latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruct from latent representation."""
        return self.decoder(z)


class DeepImageAutoencoder(nn.Module):
    """
    Deeper autoencoder — 5-layer encoder/decoder.
    Provides better quality at the cost of more parameters and training time.
    Spatial reduction: H/32 (vs H/8 for standard).
    """

    def __init__(self, bottleneck_channels: int = 8):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.architecture = "deep"

        self.encoder = nn.Sequential(
            # Layer 1: 3 → 64, H/2
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 64 → 128, H/4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 128 → 64, H/8
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 64 → 32, H/16
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 32 → bottleneck, H/32
            nn.Conv2d(32, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # Layer 1: bottleneck → 32, H/16
            nn.ConvTranspose2d(bottleneck_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: 32 → 64, H/8
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: 64 → 128, H/4
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: 128 → 64, H/2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 5: 64 → 3, H
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ── Helper Functions ─────────────────────────────────────────────────────────

def _pad_to_multiple(img_array: np.ndarray, multiple: int = 8) -> tuple:
    """Pad image so H and W are divisible by `multiple`. Returns (padded, original_h, original_w)."""
    h, w = img_array.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return img_array, h, w


def _image_to_tensor(img: Image.Image, multiple: int = 8) -> tuple:
    """Convert PIL Image to a normalized [0,1] tensor of shape (1, 3, H, W)."""
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32) / 255.0
    arr, orig_h, orig_w = _pad_to_multiple(arr, multiple)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor, orig_h, orig_w


def _tensor_to_image(tensor: torch.Tensor, orig_h: int, orig_w: int) -> Image.Image:
    """Convert (1, 3, H, W) tensor back to PIL Image, cropping to original size."""
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    arr = arr[:orig_h, :orig_w, :]  # remove padding
    return Image.fromarray(arr)


# ── Training ─────────────────────────────────────────────────────────────────

def train_autoencoder(
    image_paths: list[str],
    bottleneck_channels: int = 8,
    epochs: int = 50,
    lr: float = 1e-3,
    patch_size: int = 256,
    patches_per_image: int = 8,
    deep: bool = False,
    progress_callback=None,
) -> ImageAutoencoder | DeepImageAutoencoder:
    """
    Train the autoencoder on a collection of images.

    Uses combined L1 + SSIM loss and cosine annealing LR schedule.
    Extracts random patches from each image and trains for `epochs`.
    Returns the trained model.

    Parameters
    ----------
    deep : bool
        If True, use the deeper 5-layer architecture (requires 32-divisible patches).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the autoencoder. Install with: pip install torch torchvision")

    # Pad multiple depends on architecture depth
    pad_multiple = 32 if deep else 8

    # Extract training patches
    patches = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            h, w, _ = arr.shape
            for _ in range(patches_per_image):
                if h >= patch_size and w >= patch_size:
                    y = np.random.randint(0, h - patch_size)
                    x = np.random.randint(0, w - patch_size)
                    patch = arr[y:y+patch_size, x:x+patch_size, :]
                else:
                    # Resize small images
                    resized = np.array(img.resize((patch_size, patch_size))).astype(np.float32) / 255.0
                    patch = resized
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))  # (3, H, W)
        except Exception:
            continue

    if len(patches) < 2:
        raise ValueError("Need at least 2 valid image patches to train.")

    dataset = TensorDataset(torch.stack(patches))
    loader = DataLoader(dataset, batch_size=min(16, len(patches)), shuffle=True)

    # Select architecture
    if deep:
        model = DeepImageAutoencoder(bottleneck_channels=bottleneck_channels)
    else:
        model = ImageAutoencoder(bottleneck_channels=bottleneck_channels)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = CombinedLoss(alpha=0.84)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_loss)

    model.eval()
    return model


# ── Compression ──────────────────────────────────────────────────────────────

def compress_with_autoencoder(
    model,
    input_path: str,
    output_path: str,
) -> str:
    """
    Compress an image using the trained autoencoder.
    Saves the reconstructed output as a PNG file.

    Returns output_path on success.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    # Determine pad multiple from architecture
    pad_multiple = 32 if hasattr(model, 'architecture') and model.architecture == "deep" else 8

    img = Image.open(input_path).convert("RGB")
    tensor, orig_h, orig_w = _image_to_tensor(img, multiple=pad_multiple)

    with torch.no_grad():
        reconstructed = model(tensor)

    out_img = _tensor_to_image(reconstructed, orig_h, orig_w)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_img.save(output_path, format="PNG")
    return output_path


def get_latent_size(model, input_path: str) -> int:
    """
    Calculate the size of the compressed latent representation in bytes.
    Uses uint8 quantization + numpy compressed storage for realistic estimation.
    """
    if not TORCH_AVAILABLE:
        return 0

    pad_multiple = 32 if hasattr(model, 'architecture') and model.architecture == "deep" else 8

    img = Image.open(input_path).convert("RGB")
    tensor, _, _ = _image_to_tensor(img, multiple=pad_multiple)

    with torch.no_grad():
        latent = model.encode(tensor)

    # Quantize to uint8 for storage estimation
    latent_np = latent.cpu().numpy()
    latent_quantized = np.clip(latent_np * 255, 0, 255).astype(np.uint8)

    # Compress with numpy's built-in compression to estimate real storage
    buf = io.BytesIO()
    np.savez_compressed(buf, latent=latent_quantized)
    return buf.tell()


def save_model(model, path: str):
    """Save trained autoencoder weights."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "bottleneck_channels": model.bottleneck_channels,
        "architecture": getattr(model, "architecture", "standard"),
    }, path)


def load_model(path: str):
    """Load a previously trained autoencoder."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arch = checkpoint.get("architecture", "standard")
    bc = checkpoint["bottleneck_channels"]
    if arch == "deep":
        model = DeepImageAutoencoder(bottleneck_channels=bc)
    else:
        model = ImageAutoencoder(bottleneck_channels=bc)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
