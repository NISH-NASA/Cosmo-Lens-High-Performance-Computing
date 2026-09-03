"""
Unit and Integration Tests for HPC Image Processing & Morphology Engine.
"""

import unittest
import numpy as np
from cosmolens.hpc.fits_processor import asinh_stretch, log_stretch, create_rgb_composite, array_to_base64_png
from cosmolens.hpc.morphology import compute_gini, compute_m20, compute_concentration, compute_shape_moments, extract_morphology_profile
from cosmolens.hpc.source_extractor import extract_sources
from cosmolens.hpc.parallel_engine import HPCEngine
from cosmolens.hpc.sample_data import generate_benchmark_deepfield


class TestHPCProcessing(unittest.TestCase):
    def test_contrast_stretches(self):
        arr = np.random.uniform(0.0, 1000.0, (100, 100)).astype(np.float32)
        stretched_asinh = asinh_stretch(arr, Q=8.0)
        self.assertEqual(stretched_asinh.shape, (100, 100))
        self.assertGreaterEqual(stretched_asinh.min(), 0.0)
        self.assertLessEqual(stretched_asinh.max(), 1.0)

        stretched_log = log_stretch(arr)
        self.assertEqual(stretched_log.shape, (100, 100))
        self.assertGreaterEqual(stretched_log.min(), 0.0)
        self.assertLessEqual(stretched_log.max(), 1.0)

    def test_rgb_composite_synthesis(self):
        r = np.ones((50, 50), dtype=np.float32) * 500.0
        g = np.ones((50, 50), dtype=np.float32) * 300.0
        b = np.ones((50, 50), dtype=np.float32) * 100.0
        rgb = create_rgb_composite(r, g, b, stretch="asinh")
        self.assertEqual(rgb.shape, (50, 50, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        b64 = array_to_base64_png(rgb)
        self.assertTrue(b64.startswith("data:image/png;base64,"))

    def test_morphology_calculations(self):
        # Create a centered synthetic galaxy
        img = np.zeros((64, 64), dtype=np.float32)
        y, x = np.indices((64, 64))
        r = np.sqrt((x - 32)**2 + (y - 32)**2)
        img = 100.0 * np.exp(-r / 5.0)

        gini = compute_gini(img)
        self.assertGreater(gini, 0.2)
        self.assertLess(gini, 1.0)

        m20 = compute_m20(img, 32.0, 32.0)
        self.assertLess(m20, 0.0)

        c_idx, r20, r80 = compute_concentration(img, 32.0, 32.0)
        self.assertGreater(c_idx, 1.0)
        self.assertGreater(r80, r20)

        moments = compute_shape_moments(img, 32.0, 32.0)
        self.assertIn("ellipticity", moments)
        self.assertLess(moments["ellipticity"], 0.2) # Nearly round

    def test_benchmark_deepfield_generation(self):
        r, g, b, gt, header = generate_benchmark_deepfield(width=300, height=300)
        self.assertEqual(r.shape, (300, 300))
        self.assertGreater(len(gt), 0)
        self.assertIn("RA_DEG", header)

    def test_hpc_pipeline_execution(self):
        engine = HPCEngine()
        res = engine.run_pipeline(cutout_size=32)
        self.assertEqual(res["status"], "success")
        self.assertIn("telemetry", res)
        self.assertGreater(res["sources_count"], 0)
        self.assertGreater(res["telemetry"]["throughput_mpix_per_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
