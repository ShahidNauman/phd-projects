import os
import sys

# from hsi_analysis.parser import parse_envi_header, validate_raw_file, DATA_TYPE_NAMES
from hsi_analysis.parser import parse_envi_header


def print_cube_info(args):
    hdr_path = args.hdr
    raw_path = args.raw

    if not os.path.exists(hdr_path):
        print(f"Error: Header file not found at '{hdr_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        metadata = parse_envi_header(hdr_path)
    except Exception as e:
        print(f"Error: Failed to parse ENVI header file. Details: {e}", file=sys.stderr)
        sys.exit(1)

    bands = metadata.get("bands", "Unknown")
    wavelengths = metadata.get("wavelength", [])
    wavelength_units = metadata.get("wavelength_units", "nm")

    # Calculate starting and ending wavelength range
    if wavelengths:
        start_wl = wavelengths[0]
        end_wl = wavelengths[-1]
        wl_range_str = (
            f"{start_wl:.3f} {wavelength_units} to {end_wl:.3f} {wavelength_units}"
        )
    else:
        # Check band names or fwhm or others if wavelength array is not directly populated
        band_names = metadata.get("band_names", [])
        if band_names:
            wl_range_str = f"{band_names[0]} to {band_names[-1]}"
        else:
            wl_range_str = "Unknown"

    print("=" * 60)
    print("HYPERSPECTRAL CUBE METADATA")
    print("=" * 60)
    print(f"Header File:      {hdr_path}")
    if raw_path:
        print(f"Raw Data File:    {raw_path}")
    print("-" * 60)

    # Dimensions
    samples = metadata.get("samples", "Unknown")
    lines = metadata.get("lines", "Unknown")
    print(f"Dimensions:       {samples} (samples/width) x {lines} (lines/height)")
    print(f"Total Bands:      {bands}")

    # Wavelength Range
    print(f"Wavelength Range: {wl_range_str}")

    # # Other metadata
    # interleave = metadata.get("interleave", "Unknown").upper()
    # print(f"Interleave:       {interleave}")

    # data_type = metadata.get("data_type")
    # if data_type is not None:
    #     data_type_name = DATA_TYPE_NAMES.get(data_type, "Unknown")
    #     print(f"Data Type:        {data_type} ({data_type_name})")
    # else:
    #     print("Data Type:        Unknown")

    # # Raw File Validation
    # if raw_path:
    #     if os.path.exists(raw_path):
    #         is_valid, expected, actual = validate_raw_file(metadata, raw_path)
    #         print("-" * 60)
    #         print("RAW FILE VALIDATION")
    #         print("-" * 60)
    #         print(f"Expected Size:    {expected:,} bytes")
    #         print(f"Actual Size:      {actual:,} bytes")
    #         if is_valid:
    #             print(
    #                 "Status:           SUCCESS (File size matches dimensions and data type)"
    #             )
    #         else:
    #             print(
    #                 "Status:           FAILURE (File size mismatch or metadata missing)"
    #             )
    #     else:
    #         print("-" * 60)
    #         print(f"Warning: Raw file '{raw_path}' was specified but does not exist.")

    print("=" * 60)
