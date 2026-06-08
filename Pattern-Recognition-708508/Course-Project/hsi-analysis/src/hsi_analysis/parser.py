import os
import re

# DATA_TYPE_SIZES = {
#     1: 1,  # Byte (8-bit unsigned integer)
#     2: 2,  # Integer (16-bit signed integer)
#     3: 4,  # Long integer (32-bit signed integer)
#     4: 4,  # Floating-point (32-bit single-precision)
#     5: 8,  # Double-precision floating-point (64-bit)
#     6: 8,  # Complex (2x32-bit single-precision floating-point)
#     9: 16,  # Double-precision complex (2x64-bit floating-point)
#     12: 2,  # Unsigned integer (16-bit)
#     13: 4,  # Unsigned long integer (32-bit)
#     14: 8,  # 64-bit long integer (signed)
#     15: 8,  # Unsigned 64-bit long integer
# }

# DATA_TYPE_NAMES = {
#     1: "8-bit Unsigned Byte",
#     2: "16-bit Signed Integer",
#     3: "32-bit Signed Long Integer",
#     4: "32-bit Floating-Point",
#     5: "64-bit Double-Precision Floating-Point",
#     6: "Complex (2x32-bit float)",
#     9: "Double Complex (2x64-bit float)",
#     12: "16-bit Unsigned Integer",
#     13: "32-bit Unsigned Long Integer",
#     14: "64-bit Signed Long Integer",
#     15: "64-bit Unsigned Long Integer",
# }


def parse_envi_header(filepath):
    """
    Parses an ENVI header (.hdr) file and returns a dictionary of metadata.
    Handles multi-line list properties wrapped in {}.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Header file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Header file is empty")

    # Check for ENVI header signature, skipping empty lines/comments at start
    has_signature = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.upper() == "ENVI":
            has_signature = True
            break
        else:
            # If we see any non-comment, non-empty line before "ENVI", it's invalid
            break

    if not has_signature:
        raise ValueError("Not a valid ENVI header file (missing 'ENVI' identifier)")

    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(";"):
            continue
        clean_lines.append(line)

    metadata = {}

    current_key = None
    current_val_parts = []
    in_brace = False

    for line in clean_lines:
        line_str = line.strip()
        if not line_str or line_str.upper() == "ENVI":
            continue

        if in_brace:
            if "}" in line_str:
                brace_content = line_str.split("}")[0]
                current_val_parts.append(brace_content)
                in_brace = False
                value_str = " ".join(current_val_parts).strip()
                metadata[current_key] = value_str
                current_key = None
                current_val_parts = []
            else:
                current_val_parts.append(line_str)
            continue

        if "=" in line_str:
            parts = line_str.split("=", 1)
            key = parts[0].strip()
            val = parts[1].strip()

            if val.startswith("{"):
                if val.endswith("}"):
                    metadata[key] = val[1:-1].strip()
                else:
                    in_brace = True
                    current_key = key
                    current_val_parts = [val[1:].strip()]
            else:
                metadata[key] = val

    processed_metadata = {}
    for k, v in metadata.items():
        clean_key = k.lower().replace(" ", "_")

        # Determine if value should be treated as a list
        is_list = clean_key in [
            "wavelength",
            "fwhm",
            "band_names",
            "frame_positions",
            "frame_timestamps",
        ]

        if is_list:
            # Split by commas or whitespace, filtering out empty items
            elements = [el.strip() for el in v.split(",") if el.strip()]
            if not elements:
                # Fallback: check if space separated instead
                elements = [el.strip() for el in v.split() if el.strip()]

            parsed_list = []
            for el in elements:
                # Remove unit suffixes if any (e.g. "478.783nm" -> "478.783")
                clean_el = re.sub(r"(?i)nm", "", el).strip()
                try:
                    if "." in clean_el:
                        parsed_list.append(float(clean_el))
                    else:
                        parsed_list.append(int(clean_el))
                except ValueError:
                    parsed_list.append(el)
            processed_metadata[clean_key] = parsed_list
        else:
            try:
                if "." in v:
                    processed_metadata[clean_key] = float(v)
                else:
                    processed_metadata[clean_key] = int(v)
            except ValueError:
                processed_metadata[clean_key] = v

    return processed_metadata


# def validate_raw_file(hdr_metadata, raw_filepath):
#     """
#     Validates the size of the raw binary file matches the metadata calculations.
#     Returns (is_valid, expected_size, actual_size)
#     """
#     if not os.path.exists(raw_filepath):
#         return False, 0, 0

#     samples = hdr_metadata.get("samples")
#     lines = hdr_metadata.get("lines")
#     bands = hdr_metadata.get("bands")
#     data_type = hdr_metadata.get("data_type")

#     if None in (samples, lines, bands, data_type):
#         return False, 0, 0

#     bytes_per_pixel = DATA_TYPE_SIZES.get(data_type, 0)
#     if bytes_per_pixel == 0:
#         return False, 0, 0

#     expected_size = samples * lines * bands * bytes_per_pixel
#     actual_size = os.path.getsize(raw_filepath)

#     return expected_size == actual_size, expected_size, actual_size
