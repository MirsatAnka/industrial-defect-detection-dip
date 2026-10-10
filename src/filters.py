"""Image enhancement and spatial filtering routines for surface defect analysis."""

from typing import Tuple
import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensure image is single-channel grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def enhance_contrast_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    gray = convert_to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def reduce_noise_bilateral(
    image: np.ndarray, diameter: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0
) -> np.ndarray:
    """Edge-preserving noise attenuation using Bilateral Filtering."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def reduce_noise_gaussian(
    image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0
) -> np.ndarray:
    """Standard Gaussian smoothing filter."""
    return cv2.GaussianBlur(image, kernel_size, sigma)
