"""
usps_analysis.py
================
Performs analysis on the USPS handwritten digit dataset loaded from Hugging Face.

Tasks performed:
  1. Load the USPS dataset (train + test) via the `datasets` library.
  2. Convert every image to a 16×16 NumPy array of grayscale values.
  3. Group images by label in `images_by_label` (dict[int, list[np.ndarray]]).
  4. Compute a pixel-wise average image per label in `average_by_label`
     and in `average_images` (dict[int, np.ndarray]).
  5. Save each average image as a PNG file (average_digit_0.png … average_digit_9.png).
  6. Save a side-by-side montage of all ten average digits (average_digits_grid.png).
  7. Display each average image with Matplotlib.
  8. Print detailed progress and summary statistics.

Dependencies: datasets, numpy, Pillow, matplotlib
"""

import sys
import os
import numpy as np
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 16  # USPS images are 16 × 16 pixels
NUM_LABELS = 10  # digits 0 – 9
OUTPUT_DIR = "outputs/average_digits"  # directory to write PNG files into


# ===========================================================================
# Helper functions
# ===========================================================================


def image_to_array(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a (16, 16) NumPy array of grayscale uint8 values.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Source image (any mode; will be converted to grayscale).

    Returns
    -------
    np.ndarray
        Shape (16, 16), dtype uint8.  Each element is the grayscale
        intensity of the corresponding pixel (0 = black, 255 = white).

    Raises
    ------
    ValueError
        If the resulting array does not have the expected (16, 16) shape.
    """
    # Convert to grayscale ('L' mode) in case the source is RGB or RGBA
    gray = pil_image.convert("L")

    # Resize defensively – USPS images should already be 16×16
    if gray.size != (IMAGE_SIZE, IMAGE_SIZE):
        gray = gray.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.LANCZOS)

    arr = np.array(gray, dtype=np.uint8)  # shape: (16, 16)

    if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected array shape ({IMAGE_SIZE}, {IMAGE_SIZE}), got {arr.shape}"
        )
    return arr


def build_images_by_label(dataset_split) -> dict[int, list[list]]:
    """
    Iterate over a HuggingFace dataset split and group images by label.

    Parameters
    ----------
    dataset_split :
        A HuggingFace ``Dataset`` object with 'image' (PIL) and 'label' (int)
        columns.

    Returns
    -------
    dict[int, list[list]]
        Keys  : digit labels 0–9.
        Values: lists of 16×16 nested Python lists (converted from NumPy arrays).
    """
    # Pre-populate with empty lists for each label so the order is guaranteed
    result: dict[int, list[list]] = {label: [] for label in range(NUM_LABELS)}

    total = len(dataset_split)
    print(f"  Processing {total:,} samples …")

    for idx, sample in enumerate(dataset_split):
        label: int = int(sample["label"])
        pil_img: Image.Image = sample["image"]

        # Convert PIL → NumPy → nested list
        arr = image_to_array(pil_img)
        result[label].append(arr.tolist())  # store as nested Python list

        # Lightweight progress indicator every 1 000 images
        if (idx + 1) % 1_000 == 0 or (idx + 1) == total:
            print(f"    [{idx + 1:,}/{total:,}] images processed", end="\r")

    print()  # newline after the carriage-return progress line
    return result


def compute_average_images(
    images_by_label: dict[int, list[list]],
) -> tuple[dict[int, list], dict[int, np.ndarray]]:
    """
    Compute the pixel-wise mean image for each digit label.

    Parameters
    ----------
    images_by_label : dict[int, list[list]]
        Output of ``build_images_by_label``.

    Returns
    -------
    average_by_label : dict[int, list]
        Keys  : digit labels 0–9.
        Values: 16×16 nested Python list of float averages.
    average_images : dict[int, np.ndarray]
        Keys  : digit labels 0–9.
        Values: (16, 16) NumPy array of float64 averages.
    """
    average_by_label: dict[int, list] = {}
    average_images: dict[int, np.ndarray] = {}

    for label in range(NUM_LABELS):
        imgs = images_by_label[label]
        if not imgs:
            raise RuntimeError(f"No images found for label {label}.")

        # Stack into a 3-D array (N, 16, 16) then take the mean along axis 0
        stack = np.array(imgs, dtype=np.float64)  # shape: (N, 16, 16)
        mean = stack.mean(axis=0)  # shape: (16, 16)

        average_images[label] = mean
        average_by_label[label] = mean.tolist()  # nested Python list

    return average_by_label, average_images


def validate_average_shapes(average_images: dict[int, np.ndarray]) -> None:
    """
    Assert that every average array has shape (16, 16).

    Raises
    ------
    AssertionError
        If any array has an unexpected shape.
    """
    print("\nValidating shapes of average arrays …")
    all_valid = True
    for label, arr in average_images.items():
        shape_ok = arr.shape == (IMAGE_SIZE, IMAGE_SIZE)
        status = "✓" if shape_ok else "✗"
        print(f"  Label {label}: shape = {arr.shape}  {status}")
        if not shape_ok:
            all_valid = False

    assert all_valid, (
        "One or more average arrays do not have the expected "
        f"({IMAGE_SIZE}, {IMAGE_SIZE}) shape."
    )
    print("  All average arrays are 16×16.  ✓")


def save_average_images(
    average_images: dict[int, np.ndarray],
    output_dir: str = OUTPUT_DIR,
) -> list[str]:
    """
    Save each average image as a grayscale PNG file.

    Values are normalised to [0, 255] before saving so the full dynamic
    range is utilised, making the images easier to interpret visually.

    Parameters
    ----------
    average_images : dict[int, np.ndarray]
    output_dir     : str
        Directory in which to write the PNG files.

    Returns
    -------
    list[str]
        Absolute paths of the saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: list[str] = []

    print(f"\nSaving individual average images to '{output_dir}' …")
    for label, arr in sorted(average_images.items()):
        # Normalise float array to uint8 range [0, 255]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalised = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            # All pixels identical (degenerate case)
            normalised = np.zeros_like(arr, dtype=np.uint8)

        img = Image.fromarray(normalised, mode="L")
        path = os.path.join(output_dir, f"average_digit_{label}.png")
        img.save(path)
        saved_paths.append(os.path.abspath(path))
        print(f"  Saved: {path}")

    return saved_paths


def save_montage(
    average_images: dict[int, np.ndarray],
    output_dir: str = OUTPUT_DIR,
    filename: str = "average_digits_grid.png",
) -> str:
    """
    Create and save a 1×10 montage of all average digit images side-by-side.

    Each digit cell is individually normalised to [0, 255] before placing
    in the montage so that each digit uses its own full dynamic range.

    Parameters
    ----------
    average_images : dict[int, np.ndarray]
    output_dir     : str
    filename       : str

    Returns
    -------
    str
        Absolute path of the saved montage PNG.
    """
    # Build a list of normalised uint8 arrays in order 0–9
    tiles: list[np.ndarray] = []
    for label in range(NUM_LABELS):
        arr = average_images[label]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            norm = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(arr, dtype=np.uint8)
        tiles.append(norm)

    # Add a 2-pixel white separator between cells
    sep_width = 2
    sep_col = np.full((IMAGE_SIZE, sep_width), 255, dtype=np.uint8)

    rows = [tiles[0]]
    for tile in tiles[1:]:
        rows.append(sep_col)
        rows.append(tile)

    montage = np.concatenate(rows, axis=1)  # shape: (16, 10*16 + 9*2)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    Image.fromarray(montage, mode="L").save(path)
    print(f"\nMontage saved: {path}")
    return os.path.abspath(path)


def display_average_images(average_images: dict[int, np.ndarray]) -> None:
    """
    Display all ten average digit images in a single Matplotlib figure.

    Parameters
    ----------
    average_images : dict[int, np.ndarray]
    """
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(12, 5),
        constrained_layout=True,
    )
    fig.suptitle("USPS – Average Image per Digit (0–9)", fontsize=14, fontweight="bold")

    for label, ax in zip(range(NUM_LABELS), axes.flat):
        arr = average_images[label]
        im = ax.imshow(arr, cmap="gray", interpolation="nearest")
        ax.set_title(f"Digit {label}", fontsize=11)
        ax.axis("off")
        # Add a colour bar for each cell to indicate intensity scale
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


# ===========================================================================
# Main routine
# ===========================================================================


def main() -> None:
    print("=" * 60)
    print("  USPS Digit Dataset Analysis")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load the USPS dataset from Hugging Face
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading USPS dataset from Hugging Face …")
    try:
        dataset = load_dataset("flwrlabs/usps")
    except Exception as exc:
        print(f"ERROR: Could not load dataset.\n  {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Dataset splits available: {list(dataset.keys())}")

    # Combine train and test splits for a complete analysis.
    # ``concatenate_datasets`` preserves column names.
    from datasets import concatenate_datasets  # local import for clarity

    all_data = concatenate_datasets([dataset[split] for split in dataset.keys()])
    print(f"  Total samples (all splits combined): {len(all_data):,}")

    # ------------------------------------------------------------------
    # 2. Convert images → 16×16 arrays; group by label
    # ------------------------------------------------------------------
    print("\n[Step 2] Converting images and grouping by label …")
    images_by_label: dict[int, list[list]] = build_images_by_label(all_data)

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
    print("\n[Step 4] Computing pixel-wise average images …")
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
        f"\n  Demo – average_by_label[3][5][8] = "
        f"{average_by_label[3][5][8]:.4f}"
        "  (avg intensity of pixel (5,8) across all '3' images)"
    )

    # ------------------------------------------------------------------
    # 7. Save individual average PNGs
    # ------------------------------------------------------------------
    saved_paths = save_average_images(average_images, output_dir=OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 8. Save montage
    # ------------------------------------------------------------------
    montage_path = save_montage(average_images, output_dir=OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 9. Display average images in Matplotlib
    # ------------------------------------------------------------------
    print("\n[Step 5] Displaying average images in Matplotlib …")
    display_average_images(average_images)

    # ------------------------------------------------------------------
    # 10. Final summary
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
    print(f"\n  Montage saved: {montage_path}")
    print("\n  All average images were successfully saved.  ✓")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
