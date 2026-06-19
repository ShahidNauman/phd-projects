import os
import sys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hsi_analysis.parser import parse_envi_header


def save_classified_image(
    row_boundaries,
    col_start,
    col_end,
    norm_band,
    labels,
    k,
    output_dir,
    method="original",
):
    """
    Creates an RGB image from the grayscale Band 30, then colors the segmented
    handwriting text pixels based on their cluster labels.
    """
    # Base grayscale image from Band 30 (norm_band) normalized to 0-255
    gray_img = (norm_band * 255.0).astype(np.uint8)

    # Create RGB representation
    rgb_img = np.stack([gray_img, gray_img, gray_img], axis=-1)

    # Distinct color palette (R, G, B) for clusters
    colors = [
        [220, 20, 60],  # Red/Crimson
        [34, 139, 34],  # Forest Green
        [30, 144, 255],  # Dodger Blue
        [255, 140, 0],  # Dark Orange
        [147, 112, 219],  # Medium Purple
        [0, 139, 139],  # Dark Cyan
    ]

    # Color the handwriting pixels
    for i in range(12):
        y_start = row_boundaries[i] + 5
        y_end = row_boundaries[i + 1] - 5

        cell_norm = norm_band[y_start:y_end, col_start:col_end]
        thresh = np.percentile(cell_norm, 12)
        local_y, local_x = np.where(cell_norm <= thresh)

        global_y = local_y + y_start
        global_x = local_x + col_start

        cluster_id = labels[i]
        color = colors[cluster_id % len(colors)]

        rgb_img[global_y, global_x] = color

    img = Image.fromarray(rgb_img)
    if method == "original":
        filename = f"classified_inks_k{k}.png"
    else:
        filename = f"classified_inks_{method}_k{k}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    print(f"Saved color-labeled visualization to: {filepath}")


def detect_inks(args):
    hdr_path = args.hdr
    raw_path = args.raw
    output_dir = args.output
    method = getattr(args, "method", "original")

    if not os.path.exists(hdr_path):
        print(f"Error: Header file not found at '{hdr_path}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(raw_path):
        print(f"Error: Raw file not found at '{raw_path}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    try:
        metadata = parse_envi_header(hdr_path)
    except Exception as e:
        print(f"Error: Failed to parse ENVI header file. Details: {e}", file=sys.stderr)
        sys.exit(1)

    samples = metadata.get("samples")
    lines = metadata.get("lines")
    total_bands = metadata.get("bands")

    if not all([samples, lines, total_bands]):
        print("Error: Missing required dimension metadata in header.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"INK DETECTION PATTERN RECOGNITION USING {method.upper()} FEATURES")
    print("=" * 60)
    print(f"Header:       {hdr_path}")
    print(f"Raw Cube:     {raw_path}")
    print(f"Dimensions:   {samples} x {lines} x {total_bands}")
    print("-" * 60)

    try:
        cube_data = np.fromfile(raw_path, dtype="<f4").reshape(
            (total_bands, lines, samples)
        )
    except Exception as e:
        print(f"Error loading raw data: {e}", file=sys.stderr)
        sys.exit(1)

    # Dimensionality reduction projection if method is pca or cae
    if method == "pca":
        from hsi_analysis.reduce_dimensions import get_pca_projection

        print("Computing PCA projection (3 components)...")
        pca_images, _ = get_pca_projection(
            cube_data, lines, samples, total_bands, n_components=3
        )
        feature_cube = pca_images
        num_features = 3
    elif method == "cae":
        from hsi_analysis.reduce_dimensions import get_cae_projection

        print("Computing CAE projection (3 components)...")
        latent_transposed, _ = get_cae_projection(
            cube_data, lines, samples, total_bands, n_components=3
        )
        feature_cube = latent_transposed
        num_features = 3
    else:  # original
        feature_cube = np.transpose(cube_data, (1, 2, 0))
        num_features = total_bands

    # Segment each of the 12 text cells
    row_boundaries = [44, 100, 148, 199, 247, 295, 339, 386, 435, 485, 533, 575, 620]
    col_start = 55
    col_end = 470

    seg_band_idx = 29
    seg_band = np.clip(cube_data[seg_band_idx], 0.0, 10.0)
    norm_band = (seg_band - seg_band.min()) / (seg_band.max() - seg_band.min())

    signatures = []
    print("Segmenting handwriting cells and extracting feature signatures...")
    for i in range(12):
        y_start = row_boundaries[i] + 5
        y_end = row_boundaries[i + 1] - 5

        cell_norm = norm_band[y_start:y_end, col_start:col_end]
        thresh = np.percentile(cell_norm, 12)
        local_y, local_x = np.where(cell_norm <= thresh)

        global_y = local_y + y_start
        global_x = local_x + col_start

        # Extract values for segmented pixels inside feature cube
        # feature_cube shape is (lines, samples, num_features)
        pixel_vals = feature_cube[
            global_y, global_x
        ]  # shape: (len(global_y), num_features)

        sig = []
        for f in range(num_features):
            feat_vals = pixel_vals[:, f]
            # Filter finite values
            finite_vals = feat_vals[
                np.isfinite(feat_vals) & (feat_vals < 1e10) & (feat_vals > -1e10)
            ]
            sig.append(np.mean(finite_vals) if len(finite_vals) > 0 else 0.0)
        signatures.append(sig)

    signatures = np.array(signatures)

    # Standardize the feature curves to remove baseline/intensity offsets
    stds = signatures.std(axis=1, keepdims=True)
    stds[stds == 0.0] = 1.0
    signatures_norm = (signatures - signatures.mean(axis=1, keepdims=True)) / stds

    print("-" * 60)
    print("Evaluating K-Means Clustering Silhouette Scores:")
    print("-" * 60)

    best_k = 2
    best_score = -1.0
    results_by_k = {}

    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(signatures_norm)
        try:
            score = silhouette_score(signatures_norm, labels)
        except ValueError:
            score = 0.0
        results_by_k[k] = (score, labels)
        print(f"K={k} clusters: Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print("-" * 60)
    print(
        f"Optimal Number of Ink Clusters Detected: {best_k} (Silhouette = {best_score:.4f})"
    )
    print("-" * 60)

    # Print groupings and save color-labeled images for K=2, 3, and 5
    for k in [best_k, 3, 5]:
        score, labels = results_by_k[k]
        print(f"\nGroupings for K = {k} (Silhouette = {score:.4f}):")
        clusters = {c: [] for c in range(k)}
        for idx, label in enumerate(labels):
            pen_num = (idx // 2) + 1
            line_num = (idx % 2) + 1
            clusters[label].append(f"Pen {pen_num} (Line {line_num})")

        for c in range(k):
            print(f"  Cluster {c}: {', '.join(clusters[c])}")

        # Save color-labeled visualization image
        save_classified_image(
            row_boundaries,
            col_start,
            col_end,
            norm_band,
            labels,
            k,
            output_dir,
            method=method,
        )

    print("=" * 60)
