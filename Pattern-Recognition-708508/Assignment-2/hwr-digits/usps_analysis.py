"""USPS Digit Dataset Analysis and Visualization.

This script loads the USPS handwritten digit dataset from Hugging Face, processes
each image into a normalized 16x16 grayscale NumPy array, computes the average
representation for each digit (0-9), and generates visualizations including
individual digit averages and a side-by-side montage.

Architecture Overview:
    1. Dataset ingestion: Loading 'flwrlabs/usps' dataset via HF datasets.
    2. Data transformation: Conversion of PIL images to 16x16 NumPy arrays (uint8).
    3. Categorization: Mapping digit labels to raw arrays inside `images_by_label`.
    4. Aggregation: Computing pixel-wise average representations per label inside
       `average_by_label` (nested Python list) and `average_images` (NumPy arrays).
    5. Validation: Validating dimension shapes of computed average arrays (must be 16x16).
    6. Visualization & Export: Writing individual digits and a combined montage
       grid to disk, and displaying them using Matplotlib.

Assumptions:
    - The Hugging Face dataset has 'image' (PIL Image) and 'label' (integer) columns.
    - Standard USPS digit labels are in the range [0, 9].
    - Normalizing averages to [0, 255] helps preserve readability in output images.
"""

import io
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image

# Reconfigure stdout/stderr to UTF-8 so Unicode progress symbols print
# correctly on Windows (which defaults to cp1252 in many terminal setups).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 16  # USPS images are 16 x 16 pixels
NUM_LABELS: int = 10  # digits 0 - 9
TRAINING_DIGITS_DIR: str = (
    "outputs/training_digits"  # directory to write PNG files into
)


# ===========================================================================
# Helper functions
# ===========================================================================


def image_to_array(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL Image to a (16, 16) NumPy array of grayscale uint8 values.

    Args:
        pil_image: Source PIL image (RGB, RGBA, L, etc.).

    Returns:
        A NumPy array of shape (16, 16) and dtype uint8 containing grayscale
        intensity values (0 = black, 255 = white).

    Raises:
        ValueError: If the conversion results in an incorrect array shape.
    """
    # Convert image to grayscale ('L' mode) to discard color channels safely
    gray = pil_image.convert("L")

    # Resize defensively in case dataset contains variant dimensions
    if gray.size != (IMAGE_SIZE, IMAGE_SIZE):
        gray = gray.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.LANCZOS)

    arr = np.array(gray, dtype=np.uint8)

    # Validate array dimensions
    if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected array shape ({IMAGE_SIZE}, {IMAGE_SIZE}), got {arr.shape}"
        )
    return arr


def build_images_by_label(dataset_split: Any) -> Dict[int, List[List[List[int]]]]:
    """Iterates over a dataset split and groups images as nested lists by label.

    Args:
        dataset_split: A Hugging Face Dataset split containing 'image' and 'label'.

    Returns:
        A dictionary mapping each digit label (0-9) to a list of 16x16 nested
        Python lists representing grayscale values.
    """
    # Pre-populate dictionary keys for safety and predictable order
    result: Dict[int, List[List[List[int]]]] = {
        label: [] for label in range(NUM_LABELS)
    }

    total = len(dataset_split)
    print(f"  Processing {total:,} samples...")

    for idx, sample in enumerate(dataset_split):
        label = int(sample["label"])
        pil_img: Image.Image = sample["image"]

        # Validate label range
        if label < 0 or label >= NUM_LABELS:
            raise ValueError(f"Encountered unexpected label value: {label}")

        # Convert PIL -> NumPy -> 2D nested Python list for JSON compatibility
        arr = image_to_array(pil_img)
        result[label].append(arr.tolist())

        # Progress reporting every 1,000 images
        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"    [{idx + 1:,}/{total:,}] images processed", end="\r")

    print()  # Finalize progress line
    return result


def compute_average_images(
    images_by_label: Dict[int, List[List[List[int]]]],
) -> Tuple[Dict[int, List[List[float]]], Dict[int, np.ndarray]]:
    """Computes the pixel-wise mean image for each digit label.

    Args:
        images_by_label: A dictionary mapping digit labels (0-9) to lists of
            16x16 nested Python lists.

    Returns:
        A tuple containing:
            1. Dict[int, List[List[float]]]: A dictionary of nested 16x16 Python lists
               containing pixel averages.
            2. Dict[int, np.ndarray]: A dictionary of NumPy arrays of shape (16, 16)
               containing pixel averages.

    Raises:
        RuntimeError: If no images are available for a given label.
    """
    average_by_label: Dict[int, List[List[float]]] = {}
    average_images: Dict[int, np.ndarray] = {}

    for label in range(NUM_LABELS):
        imgs = images_by_label[label]
        if not imgs:
            raise RuntimeError(f"No images found for label {label}.")

        # Stack nested lists into (N, 16, 16) array and take mean along axis 0
        stack = np.array(imgs, dtype=np.float64)
        mean = stack.mean(axis=0)

        average_images[label] = mean
        average_by_label[label] = mean.tolist()

    return average_by_label, average_images


def validate_average_shapes(average_images: Dict[int, np.ndarray]) -> None:
    """Validates that all average digit representations have shape (16, 16).

    Args:
        average_images: Dictionary mapping labels to their computed average NumPy arrays.

    Raises:
        AssertionError: If any array does not match the (16, 16) dimension.
    """
    print("\nValidating shapes of average arrays...")
    all_valid = True
    for label, arr in average_images.items():
        shape_ok = arr.shape == (IMAGE_SIZE, IMAGE_SIZE)
        status = "[OK]" if shape_ok else "[FAIL]"
        print(f"  Label {label}: shape = {arr.shape}  {status}")
        if not shape_ok:
            all_valid = False

    assert (
        all_valid
    ), f"One or more average arrays do not have the expected ({IMAGE_SIZE}, {IMAGE_SIZE}) shape."
    print("  All average arrays are 16x16.  [OK]")


def save_average_images(
    average_images: Dict[int, np.ndarray],
    output_dir: str = TRAINING_DIGITS_DIR,
) -> List[str]:
    """Normalizes and saves individual average representations as PNG files.

    Min-max normalization is applied to scale values to the standard [0, 255]
    uint8 range for visual clarity.

    Args:
        average_images: Dictionary mapping digit labels to their computed average NumPy arrays.
        output_dir: Directory where the average digit PNGs should be written.

    Returns:
        A list of absolute file paths corresponding to the saved images.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    print(f"\nSaving individual average images to '{output_dir}'...")
    for label, arr in sorted(average_images.items()):
        # Normalize to standard uint8 intensity range [0, 255]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalised = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            # Prevent division by zero if all pixels are identical
            normalised = np.zeros_like(arr, dtype=np.uint8)

        img = Image.fromarray(normalised, mode="L")
        path = os.path.join(output_dir, f"training_digit_{label}.png")
        img.save(path)
        saved_paths.append(os.path.abspath(path))
        print(f"  Saved: {path}")

    return saved_paths


def display_average_images(average_images: Dict[int, np.ndarray]) -> None:
    """Displays computed average digit images side-by-side using Matplotlib.

    Args:
        average_images: Dictionary mapping digit labels to their computed average NumPy arrays.
    """
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(12, 5),
        constrained_layout=True,
    )
    fig.suptitle("USPS - Average Image per Digit (0-9)", fontsize=14, fontweight="bold")

    for label, ax in zip(range(NUM_LABELS), axes.flat):
        arr = average_images[label]
        im = ax.imshow(arr, cmap="gray", interpolation="nearest")
        ax.set_title(f"Digit {label}", fontsize=11)
        ax.axis("off")
        # Attach color bar for precise scaling reference
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


# ===========================================================================
# Main routine
# ===========================================================================


def main() -> None:
    """Main execution routine performing USPS data loading, processing, and visual evaluation."""
    print("=" * 60)
    print("  USPS Digit Dataset Analysis")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load the USPS dataset from Hugging Face
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading USPS dataset from Hugging Face...")
    try:
        dataset = load_dataset("flwrlabs/usps")
    except Exception as exc:
        print(f"ERROR: Could not load dataset.\n  {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Dataset splits available: {list(dataset.keys())}")

    # Combine all splits for an aggregated average analysis
    from datasets import concatenate_datasets

    all_data = concatenate_datasets([dataset[split] for split in dataset.keys()])
    print(f"  Total samples (all splits combined): {len(all_data):,}")

    # ------------------------------------------------------------------
    # 2. Convert images -> 16x16 arrays; group by label
    # ------------------------------------------------------------------
    print("\n[Step 2] Converting images and grouping by label...")
    images_by_label: Dict[int, List[List[List[int]]]] = build_images_by_label(all_data)

    # ------------------------------------------------------------------
    # 3. Report per-label counts
    # ------------------------------------------------------------------
    print("\n[Step 3] Image counts per label:")
    total_images = 0
    for label in range(NUM_LABELS):
        count = len(images_by_label[label])
        total_images += count
        print(f"  Label {label}: {count:,} images")
    print(f"\n  Total images processed: {total_images:,}")

    # ------------------------------------------------------------------
    # 4. Compute average image per label
    # ------------------------------------------------------------------
    print("\n[Step 4] Computing pixel-wise average images...")
    average_by_label, average_images = compute_average_images(images_by_label)
    print("  Done.")

    # ------------------------------------------------------------------
    # 5. Validate shapes
    # ------------------------------------------------------------------
    validate_average_shapes(average_images)

    # ------------------------------------------------------------------
    # 6. Quick access demonstration
    # ------------------------------------------------------------------
    print(
        f"\n  Demo - average_by_label[3][5][8] = "
        f"{average_by_label[3][5][8]:.4f}"
        "  (avg intensity of pixel (5,8) across all '3' images)"
    )

    # ------------------------------------------------------------------
    # 7. Save individual average PNGs
    # ------------------------------------------------------------------
    saved_paths = save_average_images(average_images, output_dir=TRAINING_DIGITS_DIR)

    # ------------------------------------------------------------------
    # 8. Display average images in Matplotlib
    # ------------------------------------------------------------------
    print("\n[Step 5] Displaying average images in Matplotlib...")
    display_average_images(average_images)

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total images processed : {total_images:,}")
    print("  Images per label:")
    for label in range(NUM_LABELS):
        print(f"    Label {label}: {len(images_by_label[label]):,}")
    print("\n  Individual average images saved:")
    for p in saved_paths:
        print(f"    {p}")
    print("\n  All average images were successfully saved.  [OK]")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
