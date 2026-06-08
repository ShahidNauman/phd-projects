import argparse
import sys

from hsi_analysis.cube_info import print_cube_info


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

    args = parser.parse_args()

    if args.command == "cube-info":
        print_cube_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
