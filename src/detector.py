"""Defect segmentation, edge detection, and morphological bounding."""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def detect_edges_canny(
    image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150
) -> np.ndarray:
    """Compute gradient edges via hysteresis thresholding in Canny operator."""
    return cv2.Canny(image, low_threshold, high_threshold)


def compute_sobel_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute spatial gradient magnitude using Sobel kernels."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    normalized = np.uint8(np.clip(magnitude / magnitude.max() * 255.0, 0, 255))
    return normalized


def segment_otsu_threshold(image: np.ndarray) -> Tuple[float, np.ndarray]:
    """Binarize surface discontinuities via Otsu's optimal thresholding."""
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


def apply_morphological_cleanup(
    binary_mask: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """Eliminate pinhole noise and bridge fractures via Opening and Closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def extract_defect_regions(
    binary_mask: np.ndarray, min_area: float = 20.0
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Find connected components and return bounding boxes & area metrics."""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    defects = []
    annotated = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            defects.append(
                {
                    "defect_id": idx + 1,
                    "bbox": (x, y, w, h),
                    "area": float(area),
                    "perimeter": float(perimeter),
                }
            )
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"#{idx+1}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    return defects, annotated
