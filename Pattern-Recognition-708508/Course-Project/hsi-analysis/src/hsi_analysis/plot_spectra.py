import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hsi_analysis.parser import parse_envi_header


def plot_spectra(args):
    hdr_path = args.hdr
    raw_path = args.raw
    output_dir = args.output

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
    wavelengths = metadata.get("wavelength")

    if not all([samples, lines, total_bands]):
        print("Error: Missing required dimension metadata in header.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("LOADING HYPERSPECTRAL DATA CUBE")
    print("=" * 60)
    print(f"Header:       {hdr_path}")
    print(f"Raw Cube:     {raw_path}")
    print(f"Dimensions:   {samples} x {lines} x {total_bands}")
    print("-" * 60)

    try:
        # Load from file (dimensions are bands, lines, samples in BSQ format)
        cube_data = np.fromfile(raw_path, dtype="<f4").reshape(
            (total_bands, lines, samples)
        )
    except Exception as e:
        print(f"Error loading raw data: {e}", file=sys.stderr)
        sys.exit(1)

    # Define row ranges for the 12 text lines (obtained from vertical profiling analysis)
    # The grid lines are: 44, 100, 148, 199, 247, 295, 339, 386, 435, 485, 533, 575, 620
    row_boundaries = [44, 100, 148, 199, 247, 295, 339, 386, 435, 485, 533, 575, 620]

    # Col boundaries to avoid grid lines (table is between x=35 and x=478, grid divider is at x=49)
    # So text column is x=49 to x=478. We use x=55 to x=470 to avoid vertical grid borders.
    col_start = 55
    col_end = 470

    # We will use Band 30 (idx 29) to segment the text pixels (where ink is highly visible)
    seg_band_idx = 29
    seg_band = np.clip(cube_data[seg_band_idx], 0.0, 10.0)

    # Global normalization for thresholding
    min_val = np.min(seg_band)
    max_val = np.max(seg_band)
    norm_band = (seg_band - min_val) / (max_val - min_val)

    # Plotting setup
    plt.figure(figsize=(12, 7))

    # Curated color palette for the 6 different pens (so the two lines of the same pen have similar colors but different linestyles)
    pen_colors = {
        1: "#1f77b4",  # blue
        2: "#ff7f0e",  # orange
        3: "#2ca02c",  # green
        4: "#d62728",  # red
        5: "#9467bd",  # purple
        6: "#8c564b",  # brown
    }

    print("EXTRACTING SPECTRAL RESPONSES FOR TEXT CELLS")
    print("-" * 60)

    for i in range(12):
        pen_num = (i // 2) + 1
        line_num = (i % 2) + 1

        y_start = row_boundaries[i]
        y_end = row_boundaries[i + 1]

        # Buffer to avoid horizontal cell border lines
        y_start_buf = y_start + 5
        y_end_buf = y_end - 5

        # Crop the cell from normalized Band 30
        cell_norm = norm_band[y_start_buf:y_end_buf, col_start:col_end]

        # Find the text pixels (dark pixels)
        # We use a dynamic threshold: pixels that are in the bottom 12% of intensities inside this cell
        thresh = np.percentile(cell_norm, 12)

        # Get absolute coordinates of text pixels in the original image
        cell_mask = cell_norm <= thresh
        local_y, local_x = np.where(cell_mask)

        global_y = local_y + y_start_buf
        global_x = local_x + col_start

        # Extract mean spectrum for these pixels across all bands
        spectrum = []
        for b in range(total_bands):
            pixel_vals = cube_data[b, global_y, global_x]
            # Filter out any infinite/NaN or extreme values
            finite_vals = pixel_vals[
                np.isfinite(pixel_vals) & (pixel_vals < 1e10) & (pixel_vals > -1e10)
            ]
            if len(finite_vals) > 0:
                spectrum.append(np.mean(finite_vals))
            else:
                spectrum.append(0.0)

        spectrum = np.array(spectrum)

        # Label and style
        label = f"Pen {pen_num} (Line {line_num})"
        color = pen_colors[pen_num]
        linestyle = "-" if line_num == 1 else "--"

        # Plot
        if wavelengths is not None and len(wavelengths) == total_bands:
            plt.plot(
                wavelengths,
                spectrum,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
            )
        else:
            plt.plot(
                range(total_bands),
                spectrum,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
            )

        print(
            f"Pen {pen_num} Line {line_num} -> Extracted {len(global_y)} text pixels. Reflectance: {spectrum.min():.3f} to {spectrum.max():.3f}"
        )

    # Finalize plot
    plt.title(
        "Spectral Reflectance of Foreground (Text) Pixels for each Pen",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    if wavelengths is not None and len(wavelengths) == total_bands:
        plt.xlabel("Wavelength (nm)", fontsize=12, labelpad=10)
    else:
        plt.xlabel("Band Number", fontsize=12, labelpad=10)
    plt.ylabel("Reflectance Intensity", fontsize=12, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "spectral_reflectance.png")
    plt.savefig(output_path, dpi=150)
    print("-" * 60)
    print(f"Saved spectral plot to: {output_path}")
    print("=" * 60)
