import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hsi_analysis.parser import parse_envi_header


def reduce_dimensions(args):
    hdr_path = args.hdr
    raw_path = args.raw
    output_dir = args.output
    n_components = args.components

    if not os.path.exists(hdr_path):
        print(f"Error: Header file not found at '{hdr_path}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(raw_path):
        print(f"Error: Raw file not found at '{raw_path}'", file=sys.stderr)
        sys.exit(1)

    if n_components < 1:
        print("Error: Number of components must be at least 1.", file=sys.stderr)
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
    print("HYPERSPECTRAL PCA DIMENSIONALITY REDUCTION")
    print("=" * 60)
    print(f"Header:       {hdr_path}")
    print(f"Raw Cube:     {raw_path}")
    print(f"Dimensions:   {samples} x {lines} x {total_bands}")
    print(f"Target PCs:   {n_components}")
    print("-" * 60)

    try:
        # Load raw data (assumed BSQ: bands x lines x samples)
        cube_data = np.fromfile(raw_path, dtype="<f4").reshape(
            (total_bands, lines, samples)
        )
    except Exception as e:
        print(f"Error loading raw data: {e}", file=sys.stderr)
        sys.exit(1)

    print("Preprocessing HSI data cube...")
    # Reshape from (total_bands, lines, samples) to (lines * samples, total_bands)
    cube_transposed = np.transpose(cube_data, (1, 2, 0))
    pixel_matrix = cube_transposed.reshape((-1, total_bands))

    # Clean and clip values to valid reflectance range [0.0, 10.0]
    # This removes huge no-data values (e.g. 3.4e38) that corrupt PCA
    pixel_matrix_clean = np.nan_to_num(pixel_matrix, nan=0.0, posinf=10.0, neginf=0.0)
    pixel_matrix_clean = np.clip(pixel_matrix_clean, 0.0, 10.0)

    # Standardize data to have zero mean and unit variance per band
    print("Standardizing spectral features...")
    scaler = StandardScaler()
    pixel_matrix_scaled = scaler.fit_transform(pixel_matrix_clean)

    # Perform PCA for output components
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(pixel_matrix_scaled)
    pca_images = pca_result.reshape((lines, samples, n_components))

    # Save components as grayscale images
    print("Saving individual PCA component images...")
    for c in range(n_components):
        pc_img = pca_images[:, :, c]

        # Min-max normalize to 0-255 range for image storage
        pc_min = pc_img.min()
        pc_max = pc_img.max()
        if pc_max > pc_min:
            pc_norm = (pc_img - pc_min) / (pc_max - pc_min)
        else:
            pc_norm = np.zeros_like(pc_img)

        pc_uint8 = (pc_norm * 255.0).astype(np.uint8)
        img = Image.fromarray(pc_uint8, mode="L")
        filename = f"pca_component_{c+1}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        print(f"Saved:        {filepath} (PC {c+1})")

    # If components >= 3, save RGB pseudo-color composite
    if n_components >= 3:
        print("Generating PCA RGB composite (PC1=R, PC2=G, PC3=B)...")
        rgb_composite = np.zeros((lines, samples, 3), dtype=np.uint8)
        for c in range(3):
            pc_img = pca_images[:, :, c]
            pc_min = pc_img.min()
            pc_max = pc_img.max()
            if pc_max > pc_min:
                pc_norm = (pc_img - pc_min) / (pc_max - pc_min)
                rgb_composite[:, :, c] = (pc_norm * 255.0).astype(np.uint8)

        composite_img = Image.fromarray(rgb_composite, mode="RGB")
        composite_path = os.path.join(output_dir, "pca_composite.png")
        composite_img.save(composite_path)
        print(f"Saved:        {composite_path} (RGB Composite)")

    # Compute explained variance metrics on up to 10 components
    n_var_components = min(10, total_bands)
    print(f"Computing explained variance for top {n_var_components} components...")
    pca_var = PCA(n_components=n_var_components)
    pca_var.fit(pixel_matrix_scaled)

    evr = pca_var.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    # Plot Scree Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(1, n_var_components + 1),
        evr,
        alpha=0.7,
        align="center",
        label="Individual Variance Explained",
        color="#1f77b4",
    )
    plt.step(
        range(1, n_var_components + 1),
        cum_evr,
        where="mid",
        label="Cumulative Variance Explained",
        color="#ff7f0e",
        linewidth=2,
    )

    # Add text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Principal Component Index", fontsize=12, labelpad=10)
    plt.ylabel("Explained Variance Ratio", fontsize=12, labelpad=10)
    plt.title(
        "PCA Explained Variance Scree Plot", fontsize=14, fontweight="bold", pad=15
    )
    plt.xticks(range(1, n_var_components + 1))
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    variance_plot_path = os.path.join(output_dir, "pca_variance_plot.png")
    plt.savefig(variance_plot_path, dpi=150)
    plt.close()
    print(f"Saved:        {variance_plot_path} (Variance Scree Plot)")

    # Print summary report
    print("-" * 60)
    print("PCA EXPLAINED VARIANCE REPORT")
    print("-" * 60)
    print(
        f"{'Component':<10} | {'Explained Variance Ratio':<24} | {'Cumulative Variance':<20}"
    )
    print("-" * 60)
    for idx, (ev, cv) in enumerate(zip(evr, cum_evr)):
        print(f"PC {idx+1:<7} | {ev:<24.6f} | {cv:<20.6f}")
    print("=" * 60)
