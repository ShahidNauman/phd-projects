import argparse
import sys

from hsi_analysis.cube_info import print_cube_info
from hsi_analysis.image_generator import generate_images


def main():
    parser = argparse.ArgumentParser(
        description="Hyperspectral Image Analysis CLI Tool"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # cube-info command
    cube_info_parser = subparsers.add_parser(
        "cube-info", help="Display information about a hyperspectral cube"
    )
    cube_info_parser.add_argument(
        "--hdr", required=True, help="Path to the ENVI header file (.hdr)"
    )
    cube_info_parser.add_argument(
        "--raw", help="Path to the raw binary spectral cube file (.raw)"
    )

    # generate-images command
    generate_parser = subparsers.add_parser(
        "generate-images", help="Generate grayscale images for specific bands"
    )
    generate_parser.add_argument(
        "--band",
        action="append",
        required=True,
        help="Band to extract (first, last, or integer index). Can be specified multiple times.",
    )
    generate_parser.add_argument(
        "--hdr", required=True, help="Path to the ENVI header file (.hdr)"
    )
    generate_parser.add_argument(
        "--raw", required=True, help="Path to the raw binary spectral cube file (.raw)"
    )
    generate_parser.add_argument(
        "--output",
        "--out",
        "-o",
        default="output/images",
        help="Directory to save the generated images (default: output/images)",
    )

    args = parser.parse_args()

    if args.command == "cube-info":
        print_cube_info(args)
    elif args.command == "generate-images":
        generate_images(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
