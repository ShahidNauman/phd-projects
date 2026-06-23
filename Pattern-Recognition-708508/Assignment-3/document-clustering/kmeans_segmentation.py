"""Image segmentation of scanned documents using K-Means clustering.

This module applies unsupervised K-Means clustering (k=3) to partition a
scanned document image into three semantic regions:

    1. **Foreground** - dark ink / handwritten text pixels.
    2. **Background** - bright page colour (white, cream, or yellowed paper).
    3. **Noise**      - intermediate-brightness artefacts such as shadows,
                        scanner noise, coffee stains, and image compression
                        artefacts (e.g. JPEG blocking).

Architecture
------------
The module is organised into six self-contained layers that call each other
in a strict top-down order:

    1. I/O helpers          - load_image(), save_image()
    2. Clustering           - run_kmeans()
    3. Semantic assignment  - assign_cluster_roles()
    4. Image construction   - build_cluster_visualisation(),
                              build_cluster_extraction()
    5. Display              - _checkerboard_background(),
                              _composite_rgba_on_background(),
                              display_results()
    6. Pipeline / CLI       - segment_image(), parse_args(), main()

Cluster Identification Heuristic
---------------------------------
K-Means assigns arbitrary integer labels {0, 1, 2} to clusters; the labels
carry no inherent semantic meaning.  To map them to Foreground / Background /
Noise, we compute the **perceptual luminance** of each cluster centroid using
the ITU-R BT.709 coefficients:

    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

We then rank the three luminance values:

    +------------------+--------+--------------------------------------+
    | Luminance rank   | Role   | Justification                        |
    +==================+========+======================================+
    | Highest (L_max)  | BG     | White/cream page reflects most light |
    +------------------+--------+--------------------------------------+
    | Lowest  (L_min)  | FG     | Dark ink absorbs most light          |
    +------------------+--------+--------------------------------------+
    | Middle           | Noise  | Artefacts fall between the two       |
    +------------------+--------+--------------------------------------+

Assumption: the input is a scanned document with a predominantly light
background and dark foreground text.  For inverted or highly colourful
documents an alternative heuristic (cluster size, spatial distribution, etc.)
would be needed.

Output Images
-------------
Four PNG files are written to the configured output directory:

    <stem>_cluster_vis.png  - colour-coded segmentation map
                              (Red=FG, Green=BG, Blue=Noise)
    <stem>_foreground.png   - RGBA, foreground pixels opaque, rest transparent
    <stem>_background.png   - RGBA, background pixels opaque, rest transparent
    <stem>_noise.png        - RGBA, noise pixels opaque, rest transparent

Dependencies
------------
    pip install opencv-python numpy matplotlib

Typical Usage
-------------
    # CLI - default image
    python kmeans_segmentation.py

    # CLI - custom image, suppress display window
    python kmeans_segmentation.py --image path/to/scan.jpg --no-display

    # Programmatic import
    from kmeans_segmentation import segment_image
    results = segment_image("scan.jpg", k=3, output_dir="out", display=False)
    foreground_bgra = results["foreground"]  # shape (H, W, 4)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

# ---------------------------------------------------------------------------
# Force UTF-8 output so Unicode characters (arrows, bullets, etc.) render
# correctly on Windows terminals that default to the cp1252 code page.
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default number of K-Means clusters.  Matches the three semantic classes:
#: foreground, background, and noise.
K: int = 3

# Semantic role names used as dictionary keys throughout the module.
LABEL_FOREGROUND: str = "Foreground"
LABEL_BACKGROUND: str = "Background"
LABEL_NOISE: str = "Noise"

# BGR colour tuples used in the cluster-visualisation image.
# NOTE: OpenCV stores images in Blue-Green-Red channel order, not RGB.
COLOR_FOREGROUND: Tuple[int, int, int] = (0, 0, 255)  # appears Red in BGR
COLOR_BACKGROUND: Tuple[int, int, int] = (0, 255, 0)  # appears Green in BGR
COLOR_NOISE: Tuple[int, int, int] = (255, 0, 0)  # appears Blue in BGR


# ---------------------------------------------------------------------------
# 1. Image I/O helpers
# ---------------------------------------------------------------------------
def load_image(image_path: str) -> np.ndarray:
    """Load an image file and return it as a BGR NumPy array.

    Validates that the path exists before attempting to decode it, so the
    caller receives a meaningful error message rather than a silent ``None``
    from OpenCV.

    Args:
        image_path: Absolute or relative path to the input image.  Supported
            formats include JPEG, PNG, BMP, TIFF, and WebP.

    Returns:
        A ``uint8`` NumPy array of shape ``(H, W, 3)`` in **BGR** channel
        order (the native OpenCV format).

    Raises:
        FileNotFoundError: If *image_path* does not exist on the filesystem.
        ValueError: If the file exists but OpenCV cannot decode it (e.g. the
            file is corrupted or in an unsupported format).

    Example:
        >>> img = load_image("uploads/scan.jpeg")
        >>> img.shape
        (784, 500, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"[ERROR] Image file not found: '{image_path}'\n"
            f"        Please verify the path and try again."
        )

    image: np.ndarray | None = cv2.imread(image_path)

    # cv2.imread returns None (not an exception) when decoding fails.
    if image is None:
        raise ValueError(
            f"[ERROR] OpenCV could not decode '{image_path}'.\n"
            f"        Supported formats: JPEG, PNG, BMP, TIFF, WebP."
        )

    print(f"[INFO]  Loaded image : {image_path}")
    print(f"[INFO]  Shape        : {image.shape}  |  dtype: {image.dtype}")
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Write a NumPy image array to disk as a PNG file.

    Creates any missing parent directories automatically.  Supports both
    3-channel BGR images and 4-channel BGRA images; PNG is used throughout
    this module because it is lossless and supports an alpha channel.

    Args:
        image: NumPy array to save.  Must be ``uint8`` and either
            ``(H, W, 3)`` (BGR) or ``(H, W, 4)`` (BGRA).
        path: Destination file path.  The ``.png`` extension should be
            included by the caller.

    Returns:
        None

    Note:
        A warning is printed (but no exception is raised) if ``cv2.imwrite``
        reports a failure.  This keeps the pipeline running even if one
        output cannot be saved (e.g. due to a permissions error).
    """
    # Ensure the output directory exists before writing.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    success: bool = cv2.imwrite(path, image)
    if success:
        print(f"[INFO]  Saved -> {path}")
    else:
        print(f"[WARN]  cv2.imwrite failed for path: {path}")


# ---------------------------------------------------------------------------
# 2. K-Means clustering
# ---------------------------------------------------------------------------
def run_kmeans(
    image: np.ndarray,
    k: int = K,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply K-Means clustering to the pixels of a BGR image.

    The image is treated as a collection of independent colour samples — spatial
    relationships between pixels are deliberately ignored.  This is valid for
    document segmentation because the three target regions (text, page, noise)
    are each characterised by a distinctive colour/brightness independent of
    where they appear on the page.

    Algorithm details:

    * **Initialisation**: k-means++ (``cv2.KMEANS_PP_CENTERS``) spreads the
      initial centroids far apart, which avoids the poor local minima that
      plague random initialisation.
    * **Restarts**: ``attempts=10`` independent runs are executed and the
      result with the lowest inertia (sum of squared distances) is returned.
    * **Stopping criteria**: convergence is declared when either the centroids
      move less than *EPSILON* pixels between iterations, or *MAX_ITER*
      iterations have elapsed.

    Args:
        image: BGR ``uint8`` array of shape ``(H, W, 3)``.
        k: Number of clusters.  Defaults to the module constant :data:`K` (3).

    Returns:
        A tuple ``(labels, centers)`` where:

        * ``labels`` - ``int32`` array of shape ``(H, W)``.  Each element is
          the cluster index ``[0, k)`` assigned to that pixel.
        * ``centers`` - ``float32`` array of shape ``(k, 3)``.  Each row is
          the mean BGR colour of one cluster centroid.

    Example:
        >>> labels, centers = run_kmeans(image, k=3)
        >>> labels.shape
        (784, 500)
        >>> centers.shape
        (3, 3)
    """
    h, w = image.shape[:2]

    # Flatten the (H, W, 3) image to (H*W, 3) so each row is one pixel's BGR
    # triple.  cv2.kmeans requires float32; uint8 would raise an error.
    pixels: np.ndarray = image.reshape(-1, 3).astype(np.float32)

    # Stopping criteria tuple: (type_flags, max_iterations, epsilon).
    # TERM_CRITERIA_EPS  -> stop when centroid shift < epsilon.
    # TERM_CRITERIA_MAX_ITER -> stop after max_iter iterations regardless.
    MAX_ITER: int = 100
    EPSILON: float = 1.0
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        MAX_ITER,
        EPSILON,
    )

    # cv2.kmeans signature:
    #   (data, K, bestLabels, criteria, attempts, flags[, centers])
    # Returns: (compactness, labels, centers)
    # We discard 'compactness' (inertia) as it is only used internally for
    # selecting the best attempt.
    _, label_flat, centers = cv2.kmeans(
        data=pixels,
        K=k,
        bestLabels=None,  # let OpenCV allocate the output label buffer
        criteria=criteria,
        attempts=10,  # run 10 independent initialisations
        flags=cv2.KMEANS_PP_CENTERS,
    )

    # label_flat has shape (H*W, 1); reshape back to the 2-D image grid.
    labels: np.ndarray = label_flat.flatten().reshape(h, w)

    print(f"[INFO]  K-Means complete  (k={k}, attempts=10)")
    for i, c in enumerate(centers):
        # c is [B, G, R] because OpenCV uses BGR order.
        print(
            f"        Cluster {i}: centroid BGR = "
            f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"
        )

    return labels, centers


# ---------------------------------------------------------------------------
# 3. Semantic cluster assignment
# ---------------------------------------------------------------------------
def assign_cluster_roles(centers: np.ndarray) -> dict[str, int]:
    """Map K-Means cluster indices to semantic roles using luminance ranking.

    K-Means assigns arbitrary integer indices to clusters; the same physical
    region (e.g. the white page) may receive index 0 on one run and index 2
    on the next.  This function deterministically resolves the ambiguity by
    computing the **perceptual luminance** of each cluster centroid and
    ranking them.

    Luminance formula (ITU-R BT.709 / sRGB):

        L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    The coefficients reflect the eye's differing sensitivity to each primary:
    green contributes most (~72%), red second (~21%), blue least (~7%).

    Ranking logic:

        * **Highest luminance** -> Background: bright page reflects most light.
        * **Lowest luminance**  -> Foreground: dark ink absorbs most light.
        * **Middle luminance**  -> Noise: intermediate-brightness artefacts.

    Assumption: this heuristic holds for scanned documents with light
    backgrounds and dark text.  It may misclassify clusters for inverted,
    heavily coloured, or very low-contrast documents.

    Args:
        centers: ``float32`` array of shape ``(k, 3)`` containing the BGR
            centroid colour of each cluster, as returned by
            :func:`run_kmeans`.

    Returns:
        A dictionary mapping each semantic role name to its cluster index::

            {
                "Foreground": 2,
                "Background": 0,
                "Noise":      1,
            }

    Example:
        >>> role_map = assign_cluster_roles(centers)
        >>> role_map["Foreground"]
        2
    """
    # Compute scalar luminance for each centroid.
    # centers[:, 0] = B, centers[:, 1] = G, centers[:, 2] = R (BGR order).
    luminance: np.ndarray = np.array(
        [0.2126 * c[2] + 0.7152 * c[1] + 0.0722 * c[0] for c in centers]
    )

    # np.argsort returns the indices that would sort luminance in ascending
    # order, i.e. sorted_indices[0] is the darkest cluster index.
    sorted_indices: np.ndarray = np.argsort(luminance)

    role_to_index: dict[str, int] = {
        LABEL_FOREGROUND: int(sorted_indices[0]),  # darkest  -> ink/text
        LABEL_NOISE: int(sorted_indices[1]),  # middle   -> artefacts
        LABEL_BACKGROUND: int(sorted_indices[2]),  # brightest -> page
    }

    print("[INFO]  Cluster role assignment (by luminance):")
    for role, idx in role_to_index.items():
        L = luminance[idx]
        c = centers[idx]
        print(
            f"        {role:12s} -> cluster {idx}  "
            f"(L={L:.1f}, BGR={c[0]:.0f},{c[1]:.0f},{c[2]:.0f})"
        )

    return role_to_index


# ---------------------------------------------------------------------------
# 4. Output image construction
# ---------------------------------------------------------------------------
def build_cluster_visualisation(
    labels: np.ndarray,
    role_to_index: dict[str, int],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Render a false-colour segmentation map from K-Means labels.

    Each pixel is replaced by a pure diagnostic colour that encodes its
    assigned semantic role:

        * **Red**   ``(R=255, G=0, B=0)`` - Foreground (text/ink).
        * **Green** ``(R=0, G=255, B=0)`` - Background (page colour).
        * **Blue**  ``(R=0, G=0, B=255)`` - Noise (artefacts).

    The resulting image is stored in OpenCV's BGR channel order.

    Args:
        labels: ``int32`` array of shape ``(H, W)`` as returned by
            :func:`run_kmeans`.  Values are cluster indices in ``[0, k)``.
        role_to_index: Mapping from role name to cluster index, as returned
            by :func:`assign_cluster_roles`.
        shape: ``(height, width)`` of the output image in pixels.

    Returns:
        A ``uint8`` BGR array of shape ``(H, W, 3)`` where every pixel
        is one of the three diagnostic colours.
    """
    h, w = shape

    # Allocate an all-black canvas; we will paint each region in turn.
    vis: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)

    # Boolean index assignment - no Python loop over pixels needed.
    # NumPy evaluates the boolean mask and assigns the colour vector in one
    # vectorised operation.
    vis[labels == role_to_index[LABEL_FOREGROUND]] = COLOR_FOREGROUND
    vis[labels == role_to_index[LABEL_BACKGROUND]] = COLOR_BACKGROUND
    vis[labels == role_to_index[LABEL_NOISE]] = COLOR_NOISE

    return vis


def build_cluster_extraction(
    image: np.ndarray,
    labels: np.ndarray,
    cluster_index: int,
) -> np.ndarray:
    """Extract one cluster from the image as an RGBA PNG with transparency.

    Pixels that belong to *cluster_index* keep their original BGR colour and
    receive alpha = 255 (fully opaque).  All other pixels receive alpha = 0
    (fully transparent), so they disappear when the PNG is opened in any
    application that respects the alpha channel (browsers, Photoshop, GIMP,
    etc.).

    Implementation notes:

    * ``cv2.COLOR_BGR2BGRA`` initialises the alpha channel to 255 for every
      pixel, so we only need to zero out the pixels we want to hide — one
      vectorised NumPy assignment rather than a per-pixel loop.
    * The returned array is in **BGRA** order (OpenCV convention).
      ``cv2.imwrite`` handles BGRA correctly when saving PNG files.

    Args:
        image: Source BGR ``uint8`` array of shape ``(H, W, 3)``.
        labels: ``int32`` label array of shape ``(H, W)`` from
            :func:`run_kmeans`.
        cluster_index: The integer cluster index whose pixels should be kept
            opaque.  All other indices become transparent.

    Returns:
        A BGRA ``uint8`` array of shape ``(H, W, 4)``.  The alpha channel
        is either 255 (cluster pixel) or 0 (non-cluster pixel).

    Example:
        >>> fg_bgra = build_cluster_extraction(image, labels, role_map["Foreground"])
        >>> fg_bgra.shape
        (784, 500, 4)
        >>> fg_bgra[..., 3].max()   # highest alpha value
        255
    """
    # Add a fully-opaque alpha channel to every pixel.
    # Output shape: (H, W, 4) with channel order B, G, R, A.
    bgra: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Build a boolean mask that is True wherever the pixel does NOT belong to
    # the requested cluster, then zero out their alpha values in one shot.
    outside_mask: np.ndarray = labels != cluster_index
    bgra[outside_mask, 3] = 0  # alpha = 0 -> fully transparent

    return bgra


# ---------------------------------------------------------------------------
# 5. Matplotlib display helpers
# ---------------------------------------------------------------------------
def _checkerboard_background(
    height: int,
    width: int,
    tile: int = 16,
    light: float = 0.85,
    dark: float = 0.65,
) -> np.ndarray:
    """Generate a grey checkerboard image for use as a transparency indicator.

    Matplotlib's ``imshow`` does not natively render PNG alpha channels.
    Compositing the RGBA extraction images over this checkerboard (before
    calling ``imshow``) makes transparent regions visually obvious — mimicking
    the convention used by image editors such as Photoshop and GIMP.

    Implementation uses vectorised NumPy broadcasting so no Python loops are
    involved, even for large images.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        tile: Side length (in pixels) of each checker square.  Defaults to 16.
        light: Brightness of the light-grey squares, in the range [0, 1].
            Defaults to 0.85.
        dark: Brightness of the dark-grey squares, in the range [0, 1].
            Defaults to 0.65.

    Returns:
        A ``uint8`` RGBA array of shape ``(H, W, 4)``.  The alpha channel is
        255 everywhere (fully opaque background).

    Note:
        This function is module-private (prefixed with ``_``) because it is an
        implementation detail of :func:`display_results` and
        :func:`_composite_rgba_on_background`.
    """
    # Integer division maps pixel rows/columns to tile indices (0, 1, 2, ...).
    # The modulo-2 converts that to an alternating 0/1 pattern per tile band.
    row_parity: np.ndarray = (np.arange(height) // tile) % 2  # shape (H,)
    col_parity: np.ndarray = (np.arange(width) // tile) % 2  # shape (W,)

    # Broadcasting: row_parity[:, None] has shape (H, 1);
    #               col_parity[None, :] has shape (1, W).
    # XOR produces True where parities differ -> alternating light/dark tiles.
    checker: np.ndarray = (row_parity[:, None] ^ col_parity[None, :]).astype(bool)

    # Map True -> light brightness, False -> dark brightness, then scale to [0, 255].
    grey_float: np.ndarray = np.where(checker, light, dark)
    grey_u8: np.ndarray = (grey_float * 255).astype(np.uint8)

    # Stack three identical grey channels plus a fully-opaque alpha channel
    # into shape (H, W, 4).  np.stack concatenates along a new axis=2.
    background: np.ndarray = np.stack(
        [grey_u8, grey_u8, grey_u8, np.full((height, width), 255, dtype=np.uint8)],
        axis=2,
    )
    return background


def _composite_rgba_on_background(
    rgba: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Alpha-composite a BGRA image over a grey checkerboard for display.

    Uses the standard **Porter-Duff "over" compositing operator**:

        out_RGB = fg_RGB * alpha + bg_RGB * (1 - alpha)

    where *alpha* is normalised to [0, 1].  This is equivalent to painting
    the foreground layer on top of the background with per-pixel opacity.

    Args:
        rgba: BGRA ``uint8`` array of shape ``(H, W, 4)``, as returned by
            :func:`build_cluster_extraction`.
        height: Image height in pixels (must match ``rgba.shape[0]``).
        width: Image width in pixels (must match ``rgba.shape[1]``).

    Returns:
        An RGB ``uint8`` array of shape ``(H, W, 3)`` suitable for passing
        directly to ``matplotlib.axes.Axes.imshow``.

    Note:
        This function is module-private (prefixed with ``_``).  External
        callers should use :func:`display_results` instead.
    """
    # Convert BGRA to RGBA so the channel order matches what NumPy and
    # Matplotlib expect: index 0=R, 1=G, 2=B, 3=A.
    rgba_float: np.ndarray = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA).astype(np.float32)

    # Slice the alpha channel and normalise to [0, 1].
    # Shape (H, W, 1) allows broadcasting against the (H, W, 3) colour arrays.
    alpha: np.ndarray = rgba_float[:, :, 3:4] / 255.0

    # Build the checkerboard background and discard its own alpha channel
    # (it is 255 everywhere and is not needed for compositing).
    background: np.ndarray = _checkerboard_background(height, width).astype(np.float32)
    bg_rgb: np.ndarray = background[:, :, :3]  # shape (H, W, 3)

    # Foreground RGB channels from the RGBA layer.
    fg_rgb: np.ndarray = rgba_float[:, :, :3]  # shape (H, W, 3)

    # Porter-Duff "over": opaque pixels show original colour; transparent
    # pixels show the checkerboard.
    composited: np.ndarray = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
    return composited.astype(np.uint8)


def display_results(
    original: np.ndarray,
    vis: np.ndarray,
    fg_image: np.ndarray,
    bg_image: np.ndarray,
    noise_image: np.ndarray,
) -> None:
    """Display all five segmentation outputs in a single Matplotlib figure.

    Renders a 1x5 grid containing:

        1. Original image (BGR converted to RGB for display).
        2. Cluster visualisation (false-colour segmentation map).
        3. Foreground extraction (composited over checkerboard).
        4. Background extraction (composited over checkerboard).
        5. Noise extraction (composited over checkerboard).

    The three extracted cluster images are BGRA arrays with transparent
    non-cluster pixels.  Because ``matplotlib.imshow`` does not natively
    handle PNG transparency, they are first alpha-composited over a grey
    checkerboard via :func:`_composite_rgba_on_background` so that
    transparency is visually obvious.

    Args:
        original: BGR ``uint8`` source image, shape ``(H, W, 3)``.
        vis: BGR ``uint8`` cluster visualisation, shape ``(H, W, 3)``.
        fg_image: BGRA ``uint8`` foreground extraction, shape ``(H, W, 4)``.
        bg_image: BGRA ``uint8`` background extraction, shape ``(H, W, 4)``.
        noise_image: BGRA ``uint8`` noise extraction, shape ``(H, W, 4)``.

    Returns:
        None.  Calls ``plt.show()`` which blocks until the window is closed.
    """

    def bgr2rgb(img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to RGB channel order for Matplotlib."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = original.shape[:2]

    # Build the list of (display_array, title) pairs in left-to-right order.
    panels = [
        (bgr2rgb(original), "Original Image"),
        (bgr2rgb(vis), "Cluster Visualisation\n(Red=FG, Green=BG, Blue=Noise)"),
        (
            _composite_rgba_on_background(fg_image, h, w),
            "Foreground Cluster\n(dark text / ink)",
        ),
        (
            _composite_rgba_on_background(bg_image, h, w),
            "Background Cluster\n(page colour)",
        ),
        (
            _composite_rgba_on_background(noise_image, h, w),
            "Noise Cluster\n(artefacts / shadows)",
        ),
    ]

    fig = plt.figure(figsize=(25, 9))
    fig.suptitle(
        "K-Means Document Image Segmentation  (k=3)",
        fontsize=16,
        fontweight="bold",
    )

    # GridSpec gives finer control over inter-panel spacing than plt.subplots.
    gs = gridspec.GridSpec(1, len(panels), figure=fig)

    for col, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=9, pad=6)
        ax.axis("off")  # hide axis ticks and labels for cleaner appearance

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Main segmentation pipeline
# ---------------------------------------------------------------------------
def segment_image(
    image_path: str,
    k: int = K,
    output_dir: str = "output",
    display: bool = True,
) -> dict[str, np.ndarray]:
    """Run the full document image segmentation pipeline.

    This is the primary public entry point for programmatic use.  It
    orchestrates all six pipeline stages in the correct order and returns
    the generated images so callers can apply further processing without
    re-running the clustering.

    Pipeline stages:

        1. **Load** - validate and decode the input image.
        2. **Cluster** - apply K-Means to the pixel colour vectors.
        3. **Assign roles** - map cluster indices to semantic labels via
           luminance ranking.
        4. **Build outputs** - generate the visualisation image and three
           transparent RGBA extraction images.
        5. **Save** - write all four images to *output_dir* as PNG files.
        6. **Display** - (optional) render a Matplotlib figure.

    Args:
        image_path: Path to the input image file.  Any format supported by
            OpenCV (JPEG, PNG, BMP, TIFF, WebP) is accepted.
        k: Number of K-Means clusters.  Must be >= 3 for the three-role
            assignment heuristic to function correctly.  Defaults to
            :data:`K` (3).
        output_dir: Directory where the four output PNG files will be saved.
            Created automatically if it does not exist.
        display: If ``True``, call :func:`display_results` to open a
            Matplotlib window.  Set to ``False`` for headless / batch use.

    Returns:
        A dictionary with four entries:

        .. code-block:: python

            {
                "visualisation": np.ndarray,  # BGR  (H, W, 3) - colour map
                "foreground":    np.ndarray,  # BGRA (H, W, 4) - transparent PNG
                "background":    np.ndarray,  # BGRA (H, W, 4) - transparent PNG
                "noise":         np.ndarray,  # BGRA (H, W, 4) - transparent PNG
            }

    Raises:
        FileNotFoundError: Propagated from :func:`load_image` if
            *image_path* does not exist.
        ValueError: Propagated from :func:`load_image` if the file cannot
            be decoded by OpenCV.

    Example:
        >>> results = segment_image(
        ...     image_path="uploads/scan.jpeg",
        ...     k=3,
        ...     output_dir="output",
        ...     display=False,
        ... )
        >>> results["foreground"].shape
        (784, 500, 4)
    """
    # ------------------------------------------------------------------
    # Stage 1: Load
    # ------------------------------------------------------------------
    image: np.ndarray = load_image(image_path)

    # ------------------------------------------------------------------
    # Stage 2: Cluster
    # ------------------------------------------------------------------
    labels: np.ndarray
    centers: np.ndarray
    labels, centers = run_kmeans(image, k=k)

    # ------------------------------------------------------------------
    # Stage 3: Assign semantic roles
    # ------------------------------------------------------------------
    role_to_index: dict[str, int] = assign_cluster_roles(centers)

    # ------------------------------------------------------------------
    # Stage 4a: Cluster visualisation (3-channel BGR)
    # ------------------------------------------------------------------
    vis: np.ndarray = build_cluster_visualisation(
        labels, role_to_index, image.shape[:2]
    )

    # ------------------------------------------------------------------
    # Stage 4b: Per-cluster transparent RGBA extractions
    # ------------------------------------------------------------------
    fg_image: np.ndarray = build_cluster_extraction(
        image, labels, role_to_index[LABEL_FOREGROUND]
    )
    bg_image: np.ndarray = build_cluster_extraction(
        image, labels, role_to_index[LABEL_BACKGROUND]
    )
    noise_image: np.ndarray = build_cluster_extraction(
        image, labels, role_to_index[LABEL_NOISE]
    )

    # ------------------------------------------------------------------
    # Stage 5: Save - derive the output stem from the input filename
    # ------------------------------------------------------------------
    # os.path.basename strips directory components; splitext drops the extension.
    base: str = os.path.splitext(os.path.basename(image_path))[0]
    save_image(vis, os.path.join(output_dir, f"{base}_cluster_vis.png"))
    save_image(fg_image, os.path.join(output_dir, f"{base}_foreground.png"))
    save_image(bg_image, os.path.join(output_dir, f"{base}_background.png"))
    save_image(noise_image, os.path.join(output_dir, f"{base}_noise.png"))

    # ------------------------------------------------------------------
    # Stage 6: Display (optional)
    # ------------------------------------------------------------------
    if display:
        display_results(image, vis, fg_image, bg_image, noise_image)

    return {
        "visualisation": vis,
        "foreground": fg_image,
        "background": bg_image,
        "noise": noise_image,
    }


# ---------------------------------------------------------------------------
# 7. CLI entry-point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Defines four optional arguments:

    * ``--image`` / ``-i``: Path to the input image.
    * ``--k``: Number of K-Means clusters.
    * ``--output-dir`` / ``-o``: Directory for saved output images.
    * ``--no-display``: Flag to suppress the Matplotlib window.

    Returns:
        An :class:`argparse.Namespace` object whose attributes correspond to
        the defined arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Segment a scanned document image into Foreground / Background / "
            "Noise using K-Means clustering (default k=3)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        "-i",
        default=os.path.join("uploads", "test_doc_image.jpeg"),
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="Number of K-Means clusters.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory in which to save the four output PNG files.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the Matplotlib display window (useful in headless or "
        "batch environments).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the command-line interface.

    Parses arguments, calls :func:`segment_image`, and exits with a non-zero
    status code if a recoverable error occurs (bad path or undecodable file).
    """
    args = parse_args()

    # Print a brief configuration summary so the user can verify their inputs
    # before the (potentially slow) K-Means step begins.
    print("=" * 60)
    print("  K-Means Document Image Segmentation")
    print("=" * 60)
    print(f"  Image      : {args.image}")
    print(f"  k          : {args.k}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Display    : {not args.no_display}")
    print("=" * 60)

    try:
        segment_image(
            image_path=args.image,
            k=args.k,
            output_dir=args.output_dir,
            display=not args.no_display,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Print the error to stderr so it can be captured separately in
        # pipelines, then exit with a non-zero code.
        print(exc, file=sys.stderr)
        sys.exit(1)

    print("[DONE]  All outputs saved.")


if __name__ == "__main__":
    main()
