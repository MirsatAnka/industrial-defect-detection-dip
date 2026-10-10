"""Pipeline orchestration script for automated industrial surface inspection."""

import argparse
from pathlib import Path
import cv2

from src.filters import enhance_contrast_clahe, reduce_noise_bilateral
from src.detector import (
    segment_otsu_threshold,
    apply_morphological_cleanup,
    extract_defect_regions,
    detect_edges_canny,
)


def process_surface_sample(image_path: str, output_dir: str) -> None:
    """Run full spatial inspection pipeline on a target sample image."""
    src = cv2.imread(image_path)
    if src is None:
        raise FileNotFoundError(f"Unable to read image at: {image_path}")

    # Step 1: Preprocessing & Contrast Equalization
    enhanced = enhance_contrast_clahe(src, clip_limit=3.0)
    denoised = reduce_noise_bilateral(enhanced)

    # Step 2: Edge & Threshold Segmentation
    edges = detect_edges_canny(denoised, 60, 180)
    _, binary = segment_otsu_threshold(denoised)

    # Step 3: Morphological Cleanup
    cleaned = apply_morphological_cleanup(binary, kernel_size=3)

    # Step 4: Extraction & Annotation
    defects, annotated = extract_defect_regions(cleaned, min_area=35.0)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    cv2.imwrite(str(out_path / f"{stem}_edges.png"), edges)
    cv2.imwrite(str(out_path / f"{stem}_cleaned_mask.png"), cleaned)
    cv2.imwrite(str(out_path / f"{stem}_annotated.png"), annotated)

    print(f"Inspection complete for {image_path}. Identified {len(defects)} defect anomaly regions.")
    for d in defects:
        print(f" - Defect #{d['defect_id']}: Area={d['area']}px, Box={d['bbox']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial Surface Defect Inspection")
    parser.add_argument("--image", type=str, required=True, help="Input surface image path")
    parser.add_argument("--out-dir", type=str, default="samples/output", help="Output directory")
    args = parser.parse_args()

    process_surface_sample(args.image, args.out_dir)


if __name__ == "__main__":
    main()
