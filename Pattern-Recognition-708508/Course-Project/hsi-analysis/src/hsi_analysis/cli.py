import argparse
import sys

from hsi_analysis.cube_info import print_cube_info
from hsi_analysis.image_generator import generate_images
from hsi_analysis.plot_spectra import plot_spectra
from hsi_analysis.detect_inks import detect_inks
from hsi_analysis.reduce_dimensions import reduce_dimensions


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

    # plot-spectra command
    plot_parser = subparsers.add_parser(
        "plot-spectra", help="Plot the spectral responses of foreground text pixels"
    )
    plot_parser.add_argument(
        "--hdr", required=True, help="Path to the ENVI header file (.hdr)"
    )
    plot_parser.add_argument(
        "--raw", required=True, help="Path to the raw binary spectral cube file (.raw)"
    )
    plot_parser.add_argument(
        "--output",
        "--out",
        "-o",
        default="output/images",
        help="Directory to save the generated plot (default: output/images)",
    )

    # detect-inks command
    detect_parser = subparsers.add_parser(
        "detect-inks", help="Detect the number of distinct inks present in the document"
    )
    detect_parser.add_argument(
        "--hdr", required=True, help="Path to the ENVI header file (.hdr)"
    )
    detect_parser.add_argument(
        "--raw", required=True, help="Path to the raw binary spectral cube file (.raw)"
    )
    detect_parser.add_argument(
        "--output",
        "--out",
        "-o",
        default="output/images",
        help="Directory to save the generated visualization (default: output/images)",
    )

    # reduce-dimensions command
    reduce_parser = subparsers.add_parser(
        "reduce-dimensions", help="Perform PCA dimensionality reduction on the HSI cube"
    )
    reduce_parser.add_argument(
        "--hdr", required=True, help="Path to the ENVI header file (.hdr)"
    )
    reduce_parser.add_argument(
        "--raw", required=True, help="Path to the raw binary spectral cube file (.raw)"
    )
    reduce_parser.add_argument(
        "--output",
        "--out",
        "-o",
        default="output/images",
        help="Directory to save the PCA outputs (default: output/images)",
    )
    reduce_parser.add_argument(
        "--components",
        "-c",
        type=int,
        default=3,
        help="Number of principal components to save as images (default: 3)",
    )

    args = parser.parse_args()

    if args.command == "cube-info":
        print_cube_info(args)
    elif args.command == "generate-images":
        generate_images(args)
    elif args.command == "plot-spectra":
        plot_spectra(args)
    elif args.command == "detect-inks":
        detect_inks(args)
    elif args.command == "reduce-dimensions":
        reduce_dimensions(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
