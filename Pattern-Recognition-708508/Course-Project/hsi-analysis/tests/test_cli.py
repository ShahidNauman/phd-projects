import unittest
import tempfile
import os

# from hsi_analysis.parser import parse_envi_header, validate_raw_file
from hsi_analysis.parser import parse_envi_header


class TestHSIAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a mock HDR file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.hdr_path = os.path.join(self.temp_dir.name, "test_sample.hdr")
        self.raw_path = os.path.join(self.temp_dir.name, "test_sample.raw")

        self.mock_hdr_content = """ENVI
description = {
  Mock ENVI header for test cases }
samples = 10
lines = 20
bands = 5
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
sensor type = Unknown
wavelength units = nm
wavelength = {
  400.0, 500.0, 600.0, 700.0, 800.0
}
"""
        with open(self.hdr_path, "w", encoding="utf-8") as f:
            f.write(self.mock_hdr_content)

        # Create a mock raw file of correct size: 10 * 20 * 5 * 4 bytes = 4000 bytes
        with open(self.raw_path, "wb") as f:
            f.write(b"\x00" * 4000)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_envi_header(self):
        metadata = parse_envi_header(self.hdr_path)
        self.assertEqual(metadata["samples"], 10)
        self.assertEqual(metadata["lines"], 20)
        self.assertEqual(metadata["bands"], 5)
        self.assertEqual(metadata["data_type"], 4)
        self.assertEqual(metadata["interleave"].lower(), "bsq")
        self.assertEqual(metadata["wavelength_units"], "nm")
        self.assertEqual(metadata["wavelength"], [400.0, 500.0, 600.0, 700.0, 800.0])

    # def test_validate_raw_file_valid(self):
    #     metadata = parse_envi_header(self.hdr_path)
    #     is_valid, expected, actual = validate_raw_file(metadata, self.raw_path)
    #     self.assertTrue(is_valid)
    #     self.assertEqual(expected, 4000)
    #     self.assertEqual(actual, 4000)

    # def test_validate_raw_file_invalid_size(self):
    #     # Write incorrect size raw file
    #     bad_raw_path = os.path.join(self.temp_dir.name, "test_bad.raw")
    #     with open(bad_raw_path, "wb") as f:
    #         f.write(b"\x00" * 3000)

    #     metadata = parse_envi_header(self.hdr_path)
    #     is_valid, expected, actual = validate_raw_file(metadata, bad_raw_path)
    #     self.assertFalse(is_valid)
    #     self.assertEqual(expected, 4000)
    #     self.assertEqual(actual, 3000)

    def test_reduce_dimensions(self):
        from hsi_analysis.reduce_dimensions import reduce_dimensions

        class Args:
            def __init__(self, hdr, raw, output, components=3, using="pca"):
                self.hdr = hdr
                self.raw = raw
                self.output = output
                self.components = components
                self.using = using

        with tempfile.TemporaryDirectory() as temp_out:
            args = Args(
                self.hdr_path, self.raw_path, temp_out, components=3, using="pca"
            )
            reduce_dimensions(args)

            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "pca_component_1.png"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "pca_component_2.png"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "pca_component_3.png"))
            )
            self.assertTrue(os.path.exists(os.path.join(temp_out, "pca_composite.png")))
            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "pca_variance_plot.png"))
            )

    def test_reduce_dimensions_cae(self):
        try:
            import torch
        except ImportError:
            # Skip test if torch is not installed yet
            return

        from hsi_analysis.reduce_dimensions import reduce_dimensions

        class Args:
            def __init__(self, hdr, raw, output, components=3, using="cae"):
                self.hdr = hdr
                self.raw = raw
                self.output = output
                self.components = components
                self.using = using

        with tempfile.TemporaryDirectory() as temp_out:
            args = Args(
                self.hdr_path, self.raw_path, temp_out, components=3, using="cae"
            )
            reduce_dimensions(args)

            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "cae_component_1.png"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "cae_component_2.png"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "cae_component_3.png"))
            )
            self.assertTrue(os.path.exists(os.path.join(temp_out, "cae_composite.png")))
            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "cae_training_loss.png"))
            )

    def test_detect_inks_pca(self):
        from hsi_analysis.detect_inks import detect_inks

        # Create a mock HSI with valid dimensions for segmentation
        hdr_path = os.path.join(self.temp_dir.name, "detect_sample.hdr")
        raw_path = os.path.join(self.temp_dir.name, "detect_sample.raw")

        hdr_content = """ENVI
samples = 512
lines = 650
bands = 31
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
"""
        with open(hdr_path, "w", encoding="utf-8") as f:
            f.write(hdr_content)

        # Write 512 * 650 * 31 * 4 = 41,267,200 bytes
        with open(raw_path, "wb") as f:
            f.write(b"\x00" * 41267200)

        class Args:
            def __init__(self, hdr, raw, output, method="pca"):
                self.hdr = hdr
                self.raw = raw
                self.output = output
                self.method = method

        with tempfile.TemporaryDirectory() as temp_out:
            args = Args(hdr_path, raw_path, temp_out, method="pca")
            detect_inks(args)

            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "classified_inks_pca_k3.png"))
            )

    def test_detect_inks_cae(self):
        try:
            import torch
        except ImportError:
            return

        from hsi_analysis.detect_inks import detect_inks

        # Create a mock HSI with valid dimensions for segmentation
        hdr_path = os.path.join(self.temp_dir.name, "detect_sample_cae.hdr")
        raw_path = os.path.join(self.temp_dir.name, "detect_sample_cae.raw")

        hdr_content = """ENVI
samples = 512
lines = 650
bands = 31
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
"""
        with open(hdr_path, "w", encoding="utf-8") as f:
            f.write(hdr_content)

        # Write 512 * 650 * 31 * 4 = 41,267,200 bytes
        with open(raw_path, "wb") as f:
            f.write(b"\x00" * 41267200)

        class Args:
            def __init__(self, hdr, raw, output, method="cae"):
                self.hdr = hdr
                self.raw = raw
                self.output = output
                self.method = method

        with tempfile.TemporaryDirectory() as temp_out:
            args = Args(hdr_path, raw_path, temp_out, method="cae")
            detect_inks(args)

            self.assertTrue(
                os.path.exists(os.path.join(temp_out, "classified_inks_cae_k3.png"))
            )


if __name__ == "__main__":
    unittest.main()
