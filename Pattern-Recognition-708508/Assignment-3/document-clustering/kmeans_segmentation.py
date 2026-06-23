"""
=============================================================================
Image Segmentation using K-Means Clustering
=============================================================================
Assignment 3 – Pattern Recognition
Institute of Space Technology (IST)

Description:
    This script applies K-Means clustering (k=3) to segment a document image
    into three semantic classes:
        1. Foreground  – dark text/ink pixels
        2. Background  – white/light page pixels
        3. Noise       – everything in between (shadows, stains, artefacts)

Cluster Identification Heuristic (clearly documented):
    After K-Means runs, each cluster centre is a 3-channel (B, G, R) colour
    value in [0, 255].  We convert each centre to a single grayscale
    brightness score using the standard luminance formula:
        L = 0.2126·R + 0.7152·G + 0.0722·B
    Then we rank the three clusters by brightness:
        • Highest L  → Background   (white / light page)
        • Lowest  L  → Foreground   (dark ink / text)
        • Middle  L  → Noise        (intermediate shades)

    This is a well-motivated heuristic for scanned document images where:
        – the page is bright white/cream,
        – text is near-black,
        – noise sits between the two.

Usage:
    python kmeans_segmentation.py                       # uses default image
    python kmeans_segmentation.py --image path/to/img  # custom image
    python kmeans_segmentation.py --image path/to/img --k 3 --no-display

Dependencies:
    pip install opencv-python-headless numpy matplotlib
=============================================================================
"""

import argparse
import os
import sys

# Force UTF-8 output so Unicode characters (arrows, bullets, etc.)
# render correctly on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K = 3  # Number of clusters

# Cluster label names (assigned after brightness ranking)
LABEL_FOREGROUND = "Foreground"
LABEL_BACKGROUND = "Background"
LABEL_NOISE      = "Noise"

# Colours used in the cluster-visualisation image (BGR order for OpenCV)
COLOR_FOREGROUND = (0,   0,   255)   # Red   in BGR
COLOR_BACKGROUND = (0,   255, 0  )   # Green in BGR
COLOR_NOISE      = (255, 0,   0  )   # Blue  in BGR


# ---------------------------------------------------------------------------
# 1. Image I/O helpers
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from *image_path* and return it as a NumPy BGR array.

    Raises:
        FileNotFoundError – if the path does not exist.
        ValueError        – if OpenCV cannot decode the file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"[ERROR] Image file not found: '{image_path}'\n"
            f"        Please check the path and try again."
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"[ERROR] OpenCV could not decode '{image_path}'.\n"
            f"        Supported formats: JPEG, PNG, BMP, TIFF, WebP, …"
        )

    print(f"[INFO]  Loaded image: {image_path}")
    print(f"[INFO]  Shape: {image.shape}  |  dtype: {image.dtype}")
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Write *image* (BGR NumPy array) to *path*, creating directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    success = cv2.imwrite(path, image)
    if success:
        print(f"[INFO]  Saved -> {path}")
    else:
        print(f"[WARN]  Failed to save image to {path}")


# ---------------------------------------------------------------------------
# 2. K-Means clustering
# ---------------------------------------------------------------------------

def run_kmeans(image: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape the image into a (N, 3) float32 matrix of pixels and apply
    cv2.kmeans with k clusters.

    Parameters:
        image – BGR uint8 image of shape (H, W, 3).
        k     – number of clusters (default 3).

    Returns:
        labels   – (H, W) integer array where each value is a cluster index [0, k).
        centers  – (k, 3) float32 array of cluster centre colours in BGR.
    """
    h, w = image.shape[:2]

    # Flatten to a list of pixels: shape (H*W, 3), dtype float32
    # cv2.kmeans requires float32 input
    pixels = image.reshape(-1, 3).astype(np.float32)

    # K-Means configuration
    # Stopping criteria: stop when either
    #   (a) the algorithm has run for MAX_ITER iterations, OR
    #   (b) cluster centres move less than EPSILON between iterations
    MAX_ITER  = 100
    EPSILON   = 1.0
    criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 MAX_ITER, EPSILON)

    # KMEANS_PP_INIT uses the k-means++ initialisation strategy which gives
    # more stable results than random initialisation.
    # attempts=10 -> run 10 independent times, keep the best result.
    _, label_flat, centers = cv2.kmeans(
        data       = pixels,
        K          = k,
        bestLabels = None,
        criteria   = criteria,
        attempts   = 10,
        flags      = cv2.KMEANS_PP_CENTERS,
    )

    # Reshape flat label array back to the original image grid
    labels = label_flat.flatten().reshape(h, w)

    print(f"[INFO]  K-Means complete.  k={k}")
    for i, c in enumerate(centers):
        print(f"        Cluster {i}: centre BGR = ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")

    return labels, centers


# ---------------------------------------------------------------------------
# 3. Semantic cluster assignment
# ---------------------------------------------------------------------------

def assign_cluster_roles(centers: np.ndarray) -> dict[str, int]:
    """
    Map each K-Means cluster index to a semantic role (Foreground / Background
    / Noise) based on the luminance of each cluster centre.

    Heuristic (see module docstring for full justification):
        Highest luminance  → Background
        Lowest  luminance  → Foreground
        Remaining          → Noise

    Parameters:
        centers – (k, 3) float32 array of cluster centres in BGR order.

    Returns:
        role_to_index – dict mapping role name → cluster index, e.g.
                        {"Foreground": 2, "Background": 0, "Noise": 1}
    """
    # Convert BGR centres to luminance (perceptual brightness)
    # Standard Rec.709 coefficients: L = 0.2126·R + 0.7152·G + 0.0722·B
    luminance = np.array([
        0.2126 * c[2] + 0.7152 * c[1] + 0.0722 * c[0]   # c = [B, G, R]
        for c in centers
    ])

    # argsort returns indices that would sort the array in ascending order
    sorted_indices = np.argsort(luminance)   # darkest to brightest

    role_to_index = {
        LABEL_FOREGROUND: int(sorted_indices[0]),   # darkest  → text
        LABEL_NOISE:      int(sorted_indices[1]),   # middle   → noise
        LABEL_BACKGROUND: int(sorted_indices[2]),   # brightest → page
    }

    print("[INFO]  Cluster role assignment (by luminance):")
    for role, idx in role_to_index.items():
        L = luminance[idx]
        c = centers[idx]
        print(f"        {role:12s} -> cluster {idx}  "
              f"(L={L:.1f}, BGR={c[0]:.0f},{c[1]:.0f},{c[2]:.0f})")

    return role_to_index


# ---------------------------------------------------------------------------
# 4. Output image construction
# ---------------------------------------------------------------------------

def build_cluster_visualisation(
    labels: np.ndarray,
    role_to_index: dict[str, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Return a colour-coded segmentation map (BGR).

    Each pixel is painted with:
        Foreground → Red   (0, 0, 255) in BGR
        Background → Green (0, 255, 0) in BGR
        Noise      → Blue  (255, 0, 0) in BGR
    """
    h, w = shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[labels == role_to_index[LABEL_FOREGROUND]] = COLOR_FOREGROUND
    vis[labels == role_to_index[LABEL_BACKGROUND]] = COLOR_BACKGROUND
    vis[labels == role_to_index[LABEL_NOISE]]      = COLOR_NOISE

    return vis


def build_cluster_extraction(
    image: np.ndarray,
    labels: np.ndarray,
    cluster_index: int,
) -> np.ndarray:
    """
    Return a copy of *image* where only pixels belonging to *cluster_index*
    retain their original colour; all other pixels are set to black (0, 0, 0).
    """
    extracted = np.zeros_like(image)
    mask = labels == cluster_index
    extracted[mask] = image[mask]
    return extracted


# ---------------------------------------------------------------------------
# 5. Matplotlib display
# ---------------------------------------------------------------------------

def display_results(
    original:    np.ndarray,
    vis:         np.ndarray,
    fg_image:    np.ndarray,
    bg_image:    np.ndarray,
    noise_image: np.ndarray,
) -> None:
    """
    Display all five images (original + 4 outputs) in a single Matplotlib
    figure with descriptive titles.
    """
    # Helper: convert BGR to RGB for Matplotlib
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images = [
        (bgr2rgb(original),    "Original Image"),
        (bgr2rgb(vis),         "Cluster Visualisation\n(Red=FG · Green=BG · Blue=Noise)"),
        (bgr2rgb(fg_image),    "Foreground Cluster\n(dark text / ink)"),
        (bgr2rgb(bg_image),    "Background Cluster\n(page colour)"),
        (bgr2rgb(noise_image), "Noise Cluster\n(artefacts / shadows)"),
    ]

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle("K-Means Document Image Segmentation  (k=3)",
                 fontsize=16, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

    for col, (img, title) in enumerate(images):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=9, pad=6)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def segment_image(
    image_path:    str,
    k:             int  = K,
    output_dir:    str  = "output",
    display:       bool = True,
) -> dict[str, np.ndarray]:
    """
    Full segmentation pipeline:
        1. Load image
        2. Run K-Means
        3. Assign semantic roles to clusters
        4. Build and save output images
        5. (Optionally) display with Matplotlib

    Parameters:
        image_path – path to the input image file.
        k          – number of K-Means clusters (default 3).
        output_dir – directory in which to save result images.
        display    – whether to call plt.show() at the end.

    Returns:
        dict with keys "visualisation", "foreground", "background", "noise"
        mapping to the corresponding NumPy BGR images.
    """
    # ── Step 1: Load ──────────────────────────────────────────────────────
    image = load_image(image_path)

    # ── Step 2: Cluster ───────────────────────────────────────────────────
    labels, centers = run_kmeans(image, k=k)

    # ── Step 3: Assign roles ──────────────────────────────────────────────
    role_to_index = assign_cluster_roles(centers)

    # ── Step 4a: Cluster visualisation ────────────────────────────────────
    vis = build_cluster_visualisation(labels, role_to_index, image.shape[:2])

    # ── Step 4b: Per-cluster extractions ──────────────────────────────────
    fg_image    = build_cluster_extraction(image, labels,
                                           role_to_index[LABEL_FOREGROUND])
    bg_image    = build_cluster_extraction(image, labels,
                                           role_to_index[LABEL_BACKGROUND])
    noise_image = build_cluster_extraction(image, labels,
                                           role_to_index[LABEL_NOISE])

    # ── Step 4c: Save all outputs ─────────────────────────────────────────
    base = os.path.splitext(os.path.basename(image_path))[0]
    save_image(vis,         os.path.join(output_dir, f"{base}_cluster_vis.png"))
    save_image(fg_image,    os.path.join(output_dir, f"{base}_foreground.png"))
    save_image(bg_image,    os.path.join(output_dir, f"{base}_background.png"))
    save_image(noise_image, os.path.join(output_dir, f"{base}_noise.png"))

    # ── Step 5: Display ───────────────────────────────────────────────────
    if display:
        display_results(image, vis, fg_image, bg_image, noise_image)

    return {
        "visualisation": vis,
        "foreground":    fg_image,
        "background":    bg_image,
        "noise":         noise_image,
    }


# ---------------------------------------------------------------------------
# 7. CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-Means document image segmentation (k=3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        default=os.path.join("uploads", "test-doc-image.jpeg"),
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="Number of K-Means clusters.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the Matplotlib display step (useful in headless environments).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  K-Means Image Segmentation")
    print("=" * 60)
    print(f"  Image      : {args.image}")
    print(f"  k          : {args.k}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 60)

    try:
        segment_image(
            image_path  = args.image,
            k           = args.k,
            output_dir  = args.output_dir,
            display     = not args.no_display,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print("[DONE]  All outputs saved.")


if __name__ == "__main__":
    main()
