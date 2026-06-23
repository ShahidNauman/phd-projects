import os
import sys
import numpy as np
from PIL import Image
from hsi_analysis.parser import parse_envi_header


def generate_images(args):
    hdr_path = args.hdr
    raw_path = args.raw
    bands_to_extract = args.band
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(hdr_path):
        print(f"Error: Header file not found at '{hdr_path}'", file=sys.stderr)
        sys.exit(1)

    if not raw_path or not os.path.exists(raw_path):
        print(f"Error: Raw file not found at '{raw_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        metadata = parse_envi_header(hdr_path)
    except Exception as e:
        print(f"Error: Failed to parse ENVI header file. Details: {e}", file=sys.stderr)
        sys.exit(1)

    samples = metadata.get("samples")
    lines = metadata.get("lines")
    total_bands = metadata.get("bands")

    if not all([samples, lines, total_bands]):
        print(
            "Error: Missing required dimension metadata (samples, lines, bands) in header.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("GENERATING GRAYSCALE BAND IMAGES")
    print("=" * 60)
    print(f"Header:       {hdr_path}")
    print(f"Raw Cube:     {raw_path}")
    print(f"Dimensions:   {samples} x {lines} x {total_bands}")
    print("-" * 60)

    # Process band arguments
    indices_to_extract = []
    band_labels = []

    for band_arg in bands_to_extract:
        band_arg_lower = band_arg.strip().lower()
        if band_arg_lower == "first":
            idx = 0
            label = "1 (first)"
        elif band_arg_lower == "last":
            idx = total_bands - 1
            label = f"{total_bands} (last)"
        else:
            try:
                band_num = int(band_arg_lower)
                if band_num < 1 or band_num > total_bands:
                    print(
                        f"Error: Band number {band_num} is out of range (1 to {total_bands}).",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                idx = band_num - 1
                label = str(band_num)
            except ValueError:
                print(
                    f"Error: Invalid band argument '{band_arg}'. Must be 'first', 'last', or an integer.",
                    file=sys.stderr,
                )
                sys.exit(1)
        indices_to_extract.append((idx, label))

    for idx, label in indices_to_extract:
        # Read the band
        try:
            bytes_per_pixel = 4  # float32
            band_size_bytes = samples * lines * bytes_per_pixel
            offset = idx * band_size_bytes

            with open(raw_path, "rb") as f:
                f.seek(offset)
                raw_data = f.read(band_size_bytes)

            if len(raw_data) < band_size_bytes:
                print(
                    f"Error: Failed to read band {idx+1}. Read only {len(raw_data)} bytes.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Parse and reshape
            band_data = np.frombuffer(raw_data, dtype="<f4")
            band_image = band_data.reshape((lines, samples))

            # Normalize to 0-255, filtering out background/no-data values outside [0.0, 10.0]
            valid_mask = (band_image >= 0.0) & (band_image <= 10.0)
            valid_pixels = band_image[valid_mask]

            if len(valid_pixels) == 0:
                img_array = np.zeros(band_image.shape, dtype=np.uint8)
            else:
                min_val = np.min(valid_pixels)
                max_val = np.max(valid_pixels)
                if max_val == min_val:
                    img_array = np.zeros(band_image.shape, dtype=np.uint8)
                else:
                    normalized = (band_image - min_val) / (max_val - min_val)
                    # Set invalid/background pixels to black (0)
                    normalized[~valid_mask] = 0.0
                    normalized = np.clip(normalized, 0.0, 1.0)
                    img_array = (normalized * 255.0).astype(np.uint8)

            # Create PIL image
            img = Image.fromarray(img_array, mode="L")

            # Save to output folder
            output_path = os.path.join(output_dir, f"band_{idx+1}.png")
            img.save(output_path)
            print(f"Saved:        {output_path} (Band {label})")

        except Exception as e:
            print(f"Error processing band {idx+1}: {e}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
