"""app.py
========
End-to-end handwritten digit recognition application.

Orchestrates two existing modules:
  - image_to_usps.py  : preprocesses a photograph into a 16x16 USPS-style image.
  - usps_analysis.py  : builds per-label average templates from the USPS dataset.

Pipeline:
    1. Load or compute average digit templates (0-9) from the USPS dataset.
       On first run this downloads and processes ~9 300 images; subsequent
       runs load a cached NumPy file in under a second.
    2. Open a file-picker dialog so the user can select a photograph.
    3. Run the preprocessing pipeline (grayscale, crop, pad, resize, invert).
    4. Classify the result by computing Euclidean distance to every template.
    5. Print distances, similarity ranking, and the predicted label.
    6. Display a Matplotlib comparison figure.

Architecture:
    load_or_compute_averages()   -- loads cache or runs usps_analysis pipeline
    preprocess_user_image()      -- calls image_to_usps step functions; returns
                                    (final_array, colour_bgr, gray, binary,
                                     cropped, padded, kept_mask, noise_mask, bbox)
    classify_digit()             -- vectorized Euclidean distance computation
    print_distances()            -- formatted per-label distance table
    print_ranking()              -- sorted similarity ranking
    show_classification_figure() -- Matplotlib comparison panel
    print_classification_summary()
    main()                       -- entry point

Assumptions:
    - average_by_label[k] is a 16x16 array of float64 (pre-inversion averages
      computed from the USPS dataset which already uses white-digit convention).
    - The user's photo contains a single dark digit on a light background.

Cache file:
    outputs/usps_averages.npy  -- shape (10, 16, 16), dtype float64
    Stores stacked averages for labels 0-9 so the dataset is not re-downloaded
    on every run.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Re-use step functions from the two existing modules.
# ---------------------------------------------------------------------------

# image_to_usps processing steps
from image_to_usps import (
    pick_image_file,
    load_image,
    detect_and_crop,
    pad_to_square,
    resize_to_target,
    apply_usps_convention,
    validate_image,
    image_to_pixel_array,
    validate_pixel_array,
    print_pixel_array,
    show_debug_components,
    show_processing_stages,
    save_image,
    TEST_DIGIT_FILENAME,
    TARGET_SIZE,
)

# usps_analysis dataset processing steps
from usps_analysis import (
    build_images_by_label,
    compute_average_images,
    NUM_LABELS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR: str = "datasets"
CACHE_FILE: str = os.path.join(CACHE_DIR, "usps_averages.npy")  # shape (10, 16, 16)


# ===========================================================================
# Step A – Average template loading / caching
# ===========================================================================


def load_or_compute_averages() -> Dict[int, np.ndarray]:
    """Returns the per-label average digit arrays, using a disk cache when available.

    On first run the USPS dataset (~9 300 images) is downloaded from Hugging Face
    and processed.  The resulting (10, 16, 16) array is saved to ``CACHE_FILE``
    so subsequent runs skip the download entirely.

    Returns:
        Dict mapping each label (0-9) to a (16, 16) float64 NumPy array
        representing the pixel-wise mean image for that digit class.
    """
    # ---- Fast path: load from cache -----------------------------------------
    if os.path.isfile(CACHE_FILE):
        print(f"[Templates] Loading cached averages from '{CACHE_FILE}'...")
        stacked: np.ndarray = np.load(CACHE_FILE)  # shape (10, 16, 16)
        average_images: Dict[int, np.ndarray] = {
            label: stacked[label] for label in range(NUM_LABELS)
        }
        print(
            f"  Loaded {NUM_LABELS} templates.  Shape per template: {stacked[0].shape}"
        )
        return average_images

    # ---- Slow path: compute from USPS dataset --------------------------------
    print("[Templates] Cache not found.  Loading USPS dataset from Hugging Face...")
    print("  This may take a minute on the first run.")

    from datasets import load_dataset, concatenate_datasets  # type: ignore[import]

    dataset = load_dataset("flwrlabs/usps")
    all_data = concatenate_datasets([dataset[split] for split in dataset.keys()])
    print(f"  Total samples: {len(all_data):,}")

    print("  Processing images...")
    images_by_label = build_images_by_label(all_data)

    print("  Computing pixel-wise averages...")
    _, average_images = compute_average_images(images_by_label)

    # Persist as a (10, 16, 16) array keyed by label index
    os.makedirs(CACHE_DIR, exist_ok=True)
    stacked = np.stack([average_images[label] for label in range(NUM_LABELS)])
    np.save(CACHE_FILE, stacked)
    print(f"  Templates cached to '{CACHE_FILE}'.")

    return average_images


# ===========================================================================
# Step B – User image preprocessing
# ===========================================================================


def preprocess_user_image(
    image_path: str,
) -> Tuple[
    np.ndarray,  # final 16x16 USPS-style image
    np.ndarray,  # colour_bgr (for visualisation)
    np.ndarray,  # gray
    np.ndarray,  # binary (Otsu mask)
    np.ndarray,  # cropped
    np.ndarray,  # padded
    np.ndarray,  # kept_mask
    np.ndarray,  # noise_mask
    Tuple[int, int, int, int],  # bbox
]:
    """Runs the full image_to_usps preprocessing pipeline and returns every stage.

    Args:
        image_path: Absolute path to the user's photograph.

    Returns:
        A 9-tuple of intermediate and final arrays (see type annotation above).

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If digit detection fails or the image cannot be decoded.
        AssertionError: If the final image fails dimension/value validation.
    """
    print("\n[Preprocessing] Loading image...")
    colour_bgr, gray = load_image(image_path)
    print(f"  Original size: {colour_bgr.shape[1]} x {colour_bgr.shape[0]} px")

    print("[Preprocessing] Converting to grayscale...")

    print("[Preprocessing] Detecting digit (connected-component analysis)...")
    cropped, binary, bbox, kept_mask, noise_mask, n_kept, n_noise = detect_and_crop(
        gray
    )
    left, top, right, bottom = bbox
    print(f"  Foreground Components Kept : {n_kept}")
    print(f"  Noise Components Removed   : {n_noise}")
    print(f"  Bounding Box               : ({left}, {top}, {right}, {bottom})")

    # Display debug component figure
    show_debug_components(colour_bgr, binary, kept_mask, noise_mask, gray, bbox)

    print("[Preprocessing] Centering and padding digit...")
    padded = pad_to_square(cropped)

    print("[Preprocessing] Resizing to 16x16...")
    resized = resize_to_target(padded)

    print("[Preprocessing] Applying USPS color convention (invert if needed)...")
    final, was_inverted = apply_usps_convention(resized)
    if was_inverted:
        print("  Light background detected -> inverted (dark digit => white on black).")
    else:
        print("  Dark background detected -> no inversion applied.")

    # Validate all constraints before proceeding
    validate_image(final)

    # Display processing-stages figure
    show_processing_stages(colour_bgr, gray, binary, cropped, padded, final)

    print("[Preprocessing] Saving processed image...")
    save_image(final, TEST_DIGIT_FILENAME)

    return final, colour_bgr, gray, binary, cropped, padded, kept_mask, noise_mask, bbox


# ===========================================================================
# Step C – Euclidean distance classification
# ===========================================================================


def classify_digit(
    user_image: np.ndarray,
    average_images: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """Computes Euclidean distance from the user image to every average template.

    Uses NumPy vectorized operations (no Python loops over pixels).

    Formula:
        distance = sqrt( sum( (user_pixel_i - template_pixel_i)^2 ) )

    Args:
        user_image:    (16, 16) uint8 array of the preprocessed user digit.
        average_images: Dict mapping label -> (16, 16) float64 average array.

    Returns:
        Dict mapping each label (0-9) to the corresponding Euclidean distance.

    Raises:
        ValueError: If any template or the user image does not have shape (16, 16).
    """
    # Validate user image dimensions
    if user_image.shape != (TARGET_SIZE, TARGET_SIZE):
        raise ValueError(
            f"User image shape {user_image.shape} is not ({TARGET_SIZE}, {TARGET_SIZE})."
        )

    # Flatten user image to a 256-element float vector once
    user_vec: np.ndarray = user_image.flatten().astype(np.float64)  # shape: (256,)

    distances: Dict[int, float] = {}

    for label, avg_array in average_images.items():
        # Validate each template
        if np.array(avg_array).shape != (TARGET_SIZE, TARGET_SIZE):
            raise ValueError(
                f"Template for label {label} has shape {np.array(avg_array).shape}; "
                f"expected ({TARGET_SIZE}, {TARGET_SIZE})."
            )

        # Flatten template and compute L2 norm of the difference vector
        template_vec: np.ndarray = np.array(avg_array).flatten()  # shape: (256,)
        dist: float = float(np.linalg.norm(user_vec - template_vec))
        distances[label] = dist

    return distances


# ===========================================================================
# Terminal output helpers
# ===========================================================================


def print_distances(distances: Dict[int, float]) -> None:
    """Prints the Euclidean distance from the user digit to every template.

    Args:
        distances: Dict mapping label -> distance.
    """
    print(f"\n{'=' * 42}")
    print("  Euclidean Distances to Each Template")
    print(f"{'=' * 42}")
    for label in range(NUM_LABELS):
        print(f"  Distance to {label}: {distances[label]:.2f}")
    print(f"{'=' * 42}\n")


def print_ranking(distances: Dict[int, float]) -> None:
    """Prints all labels sorted by increasing (best to worst) distance.

    Args:
        distances: Dict mapping label -> distance.
    """
    sorted_labels: List[int] = sorted(distances, key=lambda k: distances[k])

    print("Similarity Ranking (closest first):")
    print(f"{'=' * 32}")
    for rank, label in enumerate(sorted_labels, start=1):
        print(f"  {rank}. Digit {label}  ->  {distances[label]:.2f}")
    print(f"{'=' * 32}\n")


def print_classification_summary(
    predicted_digit: int,
    min_distance: float,
) -> None:
    """Prints the final classification result block.

    Args:
        predicted_digit: The digit label with the minimum distance.
        min_distance:    The corresponding Euclidean distance value.
    """
    print(f"\n{'=' * 42}")
    print("  Digit Classification Complete")
    print(f"{'=' * 42}")
    print(f"  Predicted Digit            : {predicted_digit}")
    print(f"  Minimum Euclidean Distance : {min_distance:.2f}")
    print(f"  Closest Average Template   : Label {predicted_digit}")
    print(f"{'=' * 42}\n")


# ===========================================================================
# Visualisation
# ===========================================================================


def show_classification_figure(
    user_image: np.ndarray,
    average_images: Dict[int, np.ndarray],
    distances: Dict[int, float],
    predicted_digit: int,
) -> None:
    """Displays a Matplotlib figure comparing the user's digit to all templates.

    Layout:
        - Row 1, left:  User's 16x16 processed digit.
        - Row 1, right: The closest matching average template.
        - Row 2:        All 10 average templates with their distances annotated.

    Args:
        user_image:      (16, 16) uint8 array of the user's processed digit.
        average_images:  Dict mapping label -> (16, 16) average template.
        distances:       Dict mapping label -> Euclidean distance.
        predicted_digit: The label with the minimum distance.
    """
    min_distance = distances[predicted_digit]
    sorted_labels = sorted(distances, key=lambda k: distances[k])

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    fig.suptitle(
        f"Digit Classification via Euclidean Distance  |  "
        f"Prediction: {predicted_digit}  |  Min Distance: {min_distance:.2f}",
        fontsize=13,
        fontweight="bold",
    )

    # ---- Top row: user image vs. closest template ----------------------------
    gs_top = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.45)

    ax_user = fig.add_subplot(gs_top[0, 0])
    ax_user.imshow(user_image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_user.set_title("Your Digit (16x16)", fontsize=11)
    ax_user.axis("off")

    ax_match = fig.add_subplot(gs_top[0, 1])
    ax_match.imshow(
        average_images[predicted_digit],
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    ax_match.set_title(
        f"Closest Template: Digit {predicted_digit}\nDistance: {min_distance:.2f}",
        fontsize=11,
    )
    ax_match.axis("off")

    # ---- Bottom row: all 10 templates ranked ---------------------------------
    gs_bottom = fig.add_gridspec(2, NUM_LABELS, height_ratios=[1.4, 1])

    for rank, label in enumerate(sorted_labels):
        ax = fig.add_subplot(gs_bottom[1, rank])
        ax.imshow(
            average_images[label],
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        # Highlight predicted digit with a coloured border
        border_color = "lime" if label == predicted_digit else "white"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
        ax.set_title(
            f"#{rank + 1}  Digit {label}\n{distances[label]:.0f}",
            fontsize=7.5,
            color="lime" if label == predicted_digit else "white",
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_facecolor("#1a1a1a")

    fig.patch.set_facecolor("#1a1a1a")
    plt.show()


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Entry point: loads templates, preprocesses the user image, and classifies."""
    print("=" * 56)
    print("  Handwritten Digit Recognizer")
    print("  (Euclidean Distance to USPS Average Templates)")
    print("=" * 56)

    # ---- A: Load or compute average templates --------------------------------
    average_images = load_or_compute_averages()

    # ---- B: Pick user image via file dialog ----------------------------------
    print("\n[Input] Please select a photograph of your handwritten digit.")
    image_path = pick_image_file()
    print(f"  Selected: {image_path}\n")

    try:
        # ---- C: Preprocess user image ----------------------------------------
        (
            final,
            colour_bgr,
            gray,
            binary,
            cropped,
            padded,
            kept_mask,
            noise_mask,
            bbox,
        ) = preprocess_user_image(image_path)

        # Print pixel array and basic statistics
        pixel_array = image_to_pixel_array(final)
        validate_pixel_array(pixel_array)
        print_pixel_array(pixel_array)

        # ---- D: Classify by Euclidean distance ------------------------------
        print("[Classification] Computing Euclidean distances...")
        distances = classify_digit(final, average_images)

        # ---- E: Print results -----------------------------------------------
        print_distances(distances)

        predicted_digit: int = min(distances, key=lambda k: distances[k])
        min_distance: float = distances[predicted_digit]

        print(f"  Predicted Digit: {predicted_digit}")
        print(f"  Minimum Distance: {min_distance:.2f}\n")

        print_ranking(distances)
        print_classification_summary(predicted_digit, min_distance)

        # ---- F: Visualize ---------------------------------------------------
        show_classification_figure(final, average_images, distances, predicted_digit)

    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
