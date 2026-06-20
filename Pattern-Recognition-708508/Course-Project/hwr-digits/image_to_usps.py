"""image_to_usps.py
==================
Converts a photograph of a handwritten digit (dark ink on light paper) into a
USPS-style 16x16 grayscale representation and outputs its pixel array.

Pipeline Overview:
    1. File picker -> user selects an image (PNG / JPG / BMP)
    2. Grayscale conversion + Gaussian blur
    3. Otsu thresholding  -> binary mask
    4. Connected-component analysis -> noise removal -> union bounding box
    5. Bounding-box crop  -> remove surrounding whitespace
    6. Aspect-preserving pad -> expand shorter side so the crop is square
    7. Resize to 16x16 via Lanczos resampling
    8. Colour inversion   -> dark-digit/light-paper => white-digit/black-background
    9. Pixel-array export + statistics + Matplotlib visualisation
    10. Save processed_digit_16x16.png

Assumptions:
    - The image contains exactly one handwritten digit.
    - The digit is darker than the surrounding paper (standard pen-on-paper photo).
    - Auto-inversion is triggered when the mean pixel value of the resized image
      exceeds 127, indicating a light background.

Dependencies:
    pip install opencv-python-headless numpy pillow matplotlib
    (tkinter ships with CPython's standard library)
"""

import sys
import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE: int = 16  # Final image edge length (pixels)
PADDING_FRACTION: float = 0.05  # Extra margin added around the bounding-box crop
# (expressed as a fraction of the bounding-box size)
OUTPUT_FILENAME: str = "outputs/processed_digit_16x16.png"
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")

# Use a non-interactive backend if no display is available (e.g. CI servers).
# On a desktop workstation this is a no-op.
matplotlib.use("TkAgg")


# ===========================================================================
# Step 1 – File selection
# ===========================================================================


def pick_image_file() -> str:
    """Opens a Tkinter file dialog and returns the chosen image path.

    Returns:
        Absolute path to the selected image file.

    Raises:
        SystemExit: If the user cancels the dialog without selecting a file.
    """
    # Hide the main Tk window; we only need the dialog widget
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Bring dialog to the front on Windows

    file_types = [
        ("Image files", "*.png *.jpg *.jpeg *.bmp"),
        ("PNG", "*.png"),
        ("JPEG", "*.jpg *.jpeg"),
        ("BMP", "*.bmp"),
        ("All files", "*.*"),
    ]

    path: str = filedialog.askopenfilename(
        title="Select a handwritten digit image",
        filetypes=file_types,
    )
    root.destroy()

    if not path:
        print("No file selected. Exiting.")
        sys.exit(0)

    # Validate extension
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        messagebox.showerror(
            "Unsupported Format",
            f"File extension '{ext}' is not supported.\n"
            f"Please choose one of: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
        sys.exit(1)

    return path


# ===========================================================================
# Step 2 – Load image
# ===========================================================================


def load_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads an image from disk and returns both a colour and a grayscale version.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        A 2-tuple:
            - colour_bgr (np.ndarray): Original image in BGR colour space
              (OpenCV convention), shape (H, W, 3) or (H, W, 4).
            - gray (np.ndarray): Grayscale version, shape (H, W), dtype uint8.

    Raises:
        FileNotFoundError: If the path does not point to a readable file.
        ValueError: If OpenCV cannot decode the image at that path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # cv2.IMREAD_COLOR loads RGB channels even for grayscale source images
    colour_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if colour_bgr is None:
        raise ValueError(f"OpenCV could not decode the image: {path}")

    gray: np.ndarray = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2GRAY)
    return colour_bgr, gray


# ===========================================================================
# Step 3 – Robust bounding-box detection using connected-component analysis
# ===========================================================================

# Minimum component area as a fraction of the total image area.
# Components smaller than this threshold are classified as noise and discarded.
# 0.05% keeps even faint strokes of disconnected digits (e.g. "4", "5") while
# reliably removing JPEG/photo sensor noise dots.
NOISE_AREA_THRESHOLD: float = 0.0005  # 0.05 % of image area


def detect_and_crop(
    gray: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray, np.ndarray, int, int
]:
    """Finds the smallest bounding box enclosing ALL significant digit components.

    Rather than relying on a single ``findNonZero`` call (which breaks for
    disconnected digits such as '4' or '5'), this function runs a full
    connected-component analysis and keeps every component whose area exceeds
    a tiny noise threshold.  The union bounding box of all surviving components
    is used to crop the image.

    Algorithm:
        1. Gaussian blur to reduce high-frequency photo noise.
        2. Otsu threshold -> binary foreground mask (digit = 255).
        3. ``cv2.connectedComponentsWithStats`` labels every connected region.
        4. Background label (0) is always skipped.
        5. Components with area < NOISE_AREA_THRESHOLD * image_area are removed.
        6. The union bounding box of all surviving components is computed.
        7. A small margin (PADDING_FRACTION) is added before cropping.

    Safety rule:
        A component is NEVER removed just because it is not the largest one.
        Only components that are clearly noise (area < threshold) are discarded.

    Args:
        gray: Single-channel grayscale image, dtype uint8.

    Returns:
        A 7-tuple:
            - cropped (np.ndarray): Cropped grayscale image.
            - binary (np.ndarray): Thresholded binary mask (uint8, same size as gray).
            - bbox (Tuple[int,int,int,int]): (left,top,right,bottom) tight bbox
              before the margin is added, in original image coordinates.
            - kept_mask (np.ndarray): Binary mask showing only kept components.
            - noise_mask (np.ndarray): Binary mask showing only removed noise components.
            - n_kept (int): Number of significant components retained.
            - n_noise (int): Number of noise components discarded.

    Raises:
        ValueError: If no foreground pixels survive noise filtering.
    """
    h, w = gray.shape
    image_area: int = h * w
    min_area: int = max(1, int(image_area * NOISE_AREA_THRESHOLD))

    # --- 1. Gaussian blur smooths JPEG compression artefacts and sensor noise
    #        before thresholding so small noise blobs merge or disappear.
    blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2. Otsu threshold.  THRESH_BINARY_INV makes the dark digit white (255)
    #        and the light background black (0).
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # --- 3. Connected-component analysis.
    #        Returns: num_labels, label_map, stats array, centroids.
    #        stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA (index constants below).
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Accumulate union bounding box across all kept components
    union_left = w  # start at far right; will shrink inward
    union_top = h
    union_right = 0  # start at far left; will grow outward
    union_bottom = 0

    kept_mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)
    noise_mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)
    n_kept = 0
    n_noise = 0

    # --- 4-7. Iterate labels; skip background (label 0).
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_mask = (label_map == label).astype(np.uint8) * 255

        if area < min_area:
            # Component is clearly noise; record it for debug display but discard.
            noise_mask = cv2.bitwise_or(noise_mask, comp_mask)
            n_noise += 1
            continue

        # Significant component: expand the union bounding box.
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[label, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[label, cv2.CC_STAT_HEIGHT])

        union_left = min(union_left, x0)
        union_top = min(union_top, y0)
        union_right = max(union_right, x1)
        union_bottom = max(union_bottom, y1)

        kept_mask = cv2.bitwise_or(kept_mask, comp_mask)
        n_kept += 1

    if n_kept == 0:
        raise ValueError(
            "No digit pixels survived noise filtering. "
            "Check that the image contains a dark handwritten digit on a light background."
        )

    # Tight bbox before padding (reported in diagnostics)
    bbox: Tuple[int, int, int, int] = (union_left, union_top, union_right, union_bottom)

    bw = union_right - union_left
    bh = union_bottom - union_top

    # Add a small safety margin around the tight bbox
    margin_x = max(1, int(bw * PADDING_FRACTION))
    margin_y = max(1, int(bh * PADDING_FRACTION))

    left_pad = max(0, union_left - margin_x)
    top_pad = max(0, union_top - margin_y)
    right_pad = min(w, union_right + margin_x)
    bottom_pad = min(h, union_bottom + margin_y)

    cropped = gray[top_pad:bottom_pad, left_pad:right_pad]

    return cropped, binary, bbox, kept_mask, noise_mask, n_kept, n_noise


# ===========================================================================
# Step 4 – Aspect-preserving square padding
# ===========================================================================


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Expands the shorter dimension of a cropped image to form a square.

    The digit is centered within the new square canvas which is filled with
    the background colour (255 = white at this stage, before inversion).

    Args:
        image: Grayscale image of the cropped digit, shape (H, W).

    Returns:
        Square grayscale image, shape (S, S) where S = max(H, W).
    """
    h, w = image.shape
    size = max(h, w)

    # Create a white canvas (255) so unfilled regions look like paper background
    square: np.ndarray = np.full((size, size), 255, dtype=np.uint8)

    # Compute offsets to centre the digit within the square
    top_offset = (size - h) // 2
    left_offset = (size - w) // 2

    square[top_offset : top_offset + h, left_offset : left_offset + w] = image
    return square


# ===========================================================================
# Step 5 – Resize to 16x16
# ===========================================================================


def resize_to_target(image: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Resizes the square digit image to (size x size) using Lanczos resampling.

    Lanczos is preferred over nearest-neighbor or bilinear for this use case
    because it preserves fine stroke details while reducing aliasing artifacts
    when downscaling from large photos to a 16x16 grid.

    Args:
        image: Grayscale square image, dtype uint8.
        size:  Target edge length in pixels (default: 16).

    Returns:
        Resized grayscale image of shape (size, size), dtype uint8.
    """
    pil_img = Image.fromarray(image)
    pil_resized = pil_img.resize((size, size), resample=Image.Resampling.LANCZOS)
    return np.array(pil_resized, dtype=np.uint8)


# ===========================================================================
# Step 6 – Colour inversion (USPS convention)
# ===========================================================================


def apply_usps_convention(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Inverts pixel values if the image has a light background.

    USPS convention: background = 0 (black), digit strokes = 255 (white).
    Real photos typically have dark ink on light paper, so inversion is almost
    always required.  The decision is based on the mean pixel value of the
    resized image: mean > 127 => light background => invert.

    Args:
        image: Grayscale image, dtype uint8.

    Returns:
        A 2-tuple:
            - result (np.ndarray): Image in USPS colour convention.
            - was_inverted (bool): True if inversion was applied.
    """
    mean_value: float = float(image.mean())
    if mean_value > 127:
        # Light background detected -> invert
        return (255 - image).astype(np.uint8), True
    return image.copy(), False


# ===========================================================================
# Step 7 – Pixel array generation
# ===========================================================================


def image_to_pixel_array(image: np.ndarray) -> list:
    """Converts a NumPy array to a nested Python list of integer pixel values.

    Args:
        image: 2D NumPy array of shape (H, W), dtype uint8.

    Returns:
        A List[List[int]] of the same dimensions.
    """
    return image.astype(int).tolist()


# ===========================================================================
# Validation helpers
# ===========================================================================


def validate_image(image: np.ndarray, expected_size: int = TARGET_SIZE) -> None:
    """Asserts that the processed image meets all dimension and value constraints.

    Args:
        image: The final 2D grayscale array to validate.
        expected_size: Expected edge length (default: 16).

    Raises:
        AssertionError: If any constraint is violated.
    """
    assert image.ndim == 2, f"Expected 2D array, got {image.ndim}D"
    assert image.shape == (
        expected_size,
        expected_size,
    ), f"Expected ({expected_size}, {expected_size}), got {image.shape}"
    assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"
    assert image.min() >= 0, f"Pixel value below 0: {image.min()}"
    assert image.max() <= 255, f"Pixel value above 255: {image.max()}"


def validate_pixel_array(array: list, expected_size: int = TARGET_SIZE) -> None:
    """Asserts that the nested list has the correct dimensions.

    Args:
        array: Nested list of pixel values.
        expected_size: Expected side length (default: 16).

    Raises:
        AssertionError: If row count or column count is wrong.
    """
    assert (
        len(array) == expected_size
    ), f"Array has {len(array)} rows; expected {expected_size}"
    for i, row in enumerate(array):
        assert (
            len(row) == expected_size
        ), f"Row {i} has {len(row)} columns; expected {expected_size}"


# ===========================================================================
# Terminal output helpers
# ===========================================================================


def print_pixel_array(array: list, size: int = TARGET_SIZE) -> None:
    """Prints the 16x16 pixel array and image statistics to the terminal.

    Args:
        array: 16x16 nested list of integer pixel values.
        size:  Side length (used for header; default 16).
    """
    flat = [v for row in array for v in row]
    black_count = sum(1 for v in flat if v == 0)
    white_count = sum(1 for v in flat if v == 255)

    print(f"\n{'=' * 56}")
    print(f"Image Shape: {size} x {size}")
    print()
    for row in array:
        formatted = ", ".join(f"{v:>3}" for v in row)
        print(f"[{formatted}]")

    print(f"\nWidth              : {size}")
    print(f"Height             : {size}")
    print(f"Minimum Pixel Value: {min(flat)}")
    print(f"Maximum Pixel Value: {max(flat)}")
    print(f"Average Pixel Value: {sum(flat) / len(flat):.2f}")
    print(f"\nBlack Pixel Count  : {black_count}")
    print(f"White Pixel Count  : {white_count}")
    print(f"\nColor Convention:")
    print(f"  Background = Black (0)")
    print(f"  Digit      = White (255)")
    print(f"{'=' * 56}\n")


def print_bbox(bbox: Tuple[int, int, int, int]) -> None:
    """Prints bounding-box diagnostic information.

    Args:
        bbox: (left, top, right, bottom) in original image coordinates.
    """
    left, top, right, bottom = bbox
    print("Cropping Bounding Box:")
    print(f"  Left   : {left}")
    print(f"  Top    : {top}")
    print(f"  Right  : {right}")
    print(f"  Bottom : {bottom}")


def print_final_summary(output_path: str) -> None:
    """Prints the success summary block.

    Args:
        output_path: Path to the saved output PNG.
    """
    print(f"\n{'=' * 56}")
    print("Successfully generated USPS-style 16x16 grayscale digit.")
    print(f"\nImage Dimensions: {TARGET_SIZE} x {TARGET_SIZE}")
    print(f"Array Dimensions: {TARGET_SIZE} x {TARGET_SIZE}")
    print(f"\nBackground = Black (0)")
    print(f"Digit      = White (255)")
    print(f"\nOutput image saved as:")
    print(f"  {os.path.abspath(output_path)}")
    print(f"{'=' * 56}\n")


# ===========================================================================
# Visualisation
# ===========================================================================


def show_debug_components(
    colour_bgr: np.ndarray,
    binary: np.ndarray,
    kept_mask: np.ndarray,
    noise_mask: np.ndarray,
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> None:
    """Displays a debug figure showing connected-component analysis results.

    Panels shown:
        1. Original image.
        2. Thresholded binary mask (after Otsu).
        3. Components classified as noise (removed).
        4. Components classified as digit (kept).
        5. Original image with the final crop rectangle drawn on it.

    Args:
        colour_bgr: Original image in BGR format.
        binary:     Otsu binary mask (digit = 255).
        kept_mask:  Binary mask of all kept (significant) components.
        noise_mask: Binary mask of all discarded (noise) components.
        gray:       Grayscale version of original (for bbox overlay).
        bbox:       (left, top, right, bottom) tight bounding box.
    """
    colour_rgb = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2RGB)

    # Draw the crop rectangle on a copy of the colour image
    bbox_img = colour_rgb.copy()
    left, top, right, bottom = bbox
    cv2.rectangle(
        bbox_img, (left, top), (right, bottom), color=(255, 0, 0), thickness=2
    )

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    fig.suptitle(
        "Debug: Connected-Component Analysis",
        fontsize=13,
        fontweight="bold",
    )

    panels = [
        (colour_rgb, "Original Image", None),
        (binary, "Thresholded (Otsu)", "gray"),
        (noise_mask, "Noise Components (removed)", "Reds"),
        (kept_mask, "Digit Components (kept)", "Greens"),
        (bbox_img, "Final Crop Rectangle", None),
    ]

    for ax, (img, title, cmap) in zip(axes, panels):
        if img.ndim == 3 or cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.show()


def show_processing_stages(
    colour_bgr: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    cropped: np.ndarray,
    padded: np.ndarray,
    final: np.ndarray,
) -> None:
    """Displays each processing stage in a Matplotlib subplot figure.

    Args:
        colour_bgr: Original loaded image (BGR OpenCV format).
        gray:       Grayscale version of the original.
        binary:     Otsu binary threshold mask.
        cropped:    Bounding-box crop of the digit.
        padded:     Aspect-preserving square-padded digit.
        final:      Final 16x16 USPS-style image.
    """
    # Convert BGR to RGB for correct Matplotlib colour rendering
    colour_rgb = cv2.cvtColor(colour_bgr, cv2.COLOR_BGR2RGB)

    stages = [
        (colour_rgb, "Stage 1: Original Image", None),
        (gray, "Stage 2: Grayscale Image", "gray"),
        (binary, "Stage 3: Thresholded Binary", "gray"),
        (cropped, "Stage 4: Cropped Digit", "gray"),
        (padded, "Stage 5: Centered & Padded Digit", "gray"),
        (final, "Stage 6: Final 16x16 USPS Image", "gray"),
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=6,
        figsize=(22, 4),
        constrained_layout=True,
    )
    fig.suptitle(
        "Handwritten Digit -> USPS 16x16 Processing Pipeline",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (img, title, cmap) in zip(axes, stages):
        if img.ndim == 3 or cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.show()


# ===========================================================================
# Save
# ===========================================================================


def save_image(image: np.ndarray, output_path: str = OUTPUT_FILENAME) -> None:
    """Saves the processed 16x16 image as a PNG file.

    Args:
        image:       2D uint8 NumPy array to save.
        output_path: Destination file path (default: processed_digit_16x16.png).
    """
    pil_img = Image.fromarray(image, mode="L")
    pil_img.save(output_path)
    print(f"  Saved -> {os.path.abspath(output_path)}")


# ===========================================================================
# Main pipeline
# ===========================================================================


def process_digit_image(path: str) -> None:
    """Runs the full digit-image preprocessing pipeline.

    Args:
        path: Absolute path to the source image.
    """
    # ---- Load ----------------------------------------------------------------
    print("Loading image...")
    colour_bgr, gray = load_image(path)
    print(f"  Original size: {colour_bgr.shape[1]} x {colour_bgr.shape[0]} px")

    # ---- Grayscale -----------------------------------------------------------
    print("Converting to grayscale...")
    # gray is already computed inside load_image

    # ---- Detect & crop (robust multi-component) ------------------------------
    print("Detecting digit...")
    print("Applying Gaussian blur and Otsu threshold...")
    print("Running connected-component analysis...")
    print("Cropping whitespace...")
    cropped, binary, bbox, kept_mask, noise_mask, n_kept, n_noise = detect_and_crop(
        gray
    )

    # Print component diagnostics
    print(f"  Foreground Components Kept   : {n_kept}")
    print(f"  Noise Components Removed     : {n_noise}")
    left, top, right, bottom = bbox
    print(f"  Final Bounding Box           : ({left}, {top}, {right}, {bottom})")
    print_bbox(bbox)
    print(f"  Cropped size: {cropped.shape[1]} x {cropped.shape[0]} px")

    # ---- Debug component visualisation ---------------------------------------
    show_debug_components(colour_bgr, binary, kept_mask, noise_mask, gray, bbox)

    # ---- Pad to square -------------------------------------------------------
    print("Centering digit...")
    padded = pad_to_square(cropped)
    print(f"  Padded size: {padded.shape[1]} x {padded.shape[0]} px")

    # ---- Resize --------------------------------------------------------------
    print("Resizing to 16x16...")
    resized = resize_to_target(padded)

    # ---- Invert to USPS convention -------------------------------------------
    print("Inverting colors...")
    final, was_inverted = apply_usps_convention(resized)
    if was_inverted:
        print("  Light background detected -> colours inverted (dark digit => white).")
    else:
        print("  Dark background detected -> no inversion needed.")

    # ---- Validate ------------------------------------------------------------
    validate_image(final)

    # ---- Generate pixel array ------------------------------------------------
    print("Generating pixel array...")
    pixel_array = image_to_pixel_array(final)
    validate_pixel_array(pixel_array)

    # ---- Terminal output -----------------------------------------------------
    print_pixel_array(pixel_array)

    # ---- Processing-stages visualisation -------------------------------------
    show_processing_stages(colour_bgr, gray, binary, cropped, padded, final)

    # ---- Save ----------------------------------------------------------------
    print("Saving image...")
    save_image(final, OUTPUT_FILENAME)

    # ---- Final summary -------------------------------------------------------
    print("Done.")
    print_final_summary(OUTPUT_FILENAME)


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Entry point: opens the file picker and runs the processing pipeline."""
    print("=" * 56)
    print("  Handwritten Digit -> USPS 16x16 Converter")
    print("=" * 56)

    # Let the user pick an image via GUI dialog
    image_path = pick_image_file()
    print(f"\nSelected file: {image_path}\n")

    try:
        process_digit_image(image_path)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
