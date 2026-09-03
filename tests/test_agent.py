"""
Unit and Integration Tests for Gemini AI Agent & Discovery Modules.
"""

import unittest
from cosmolens.ai.gemini_agent import GeminiAstroAgent
from cosmolens.ai.discovery_report import generate_apj_discovery_memo
from cosmolens.ai.natural_search import search_sky_catalog


class TestGeminiAgent(unittest.TestCase):
    def setUp(self):
        self.agent = GeminiAstroAgent()
        self.mock_source = {
            "id": "JWST-CL-0001",
            "x": 650.0,
            "y": 620.0,
            "ra_str": "07h23m19.40s",
            "dec_str": "-73°27'15.0\"",
            "snr": 28.5,
            "total_flux": 1200.0,
            "f444_f090_ratio": 1.4,
            "morphology": {
                "gini": 0.58,
                "m20": -1.45,
                "concentration": 3.4,
                "asymmetry": 0.15,
                "ellipticity": 0.72,
                "curvature_score": 0.65,
                "lens_geometric_score": 0.78,
                "dist_to_cluster_core_px": 140.0,
                "tangential_predicted_deg": 48.0,
                "is_lens_candidate": True,
                "is_merger_candidate": False
            }
        }
        self.header = {"TELESCOP": "JWST", "INSTRUME": "NIRCAM"}

    def test_analysis_inference(self):
        analysis = self.agent.analyze_source(self.mock_source)
        self.assertIn("classification", analysis)
        self.assertEqual(analysis["classification"], "EINSTEIN_RING_OR_ARC")
        self.assertGreater(analysis["confidence"], 0.7)
        self.assertIn("magnification_factor", analysis)

    def test_discovery_memo_generation(self):
        analysis = self.agent.analyze_source(self.mock_source)
        memo = generate_apj_discovery_memo(self.mock_source, analysis, self.header)
        self.assertIn("DISCOVERY MEMORANDUM", memo)
        self.assertIn("JWST-CL-0001", memo)
        self.assertIn("theta_E", memo) # LaTeX formula

    def test_semantic_sky_search(self):
        catalog = [self.mock_source]
        res = search_sky_catalog("find einstein arcs with high curvature", catalog, self.agent)
        self.assertEqual(res["total_matches"], 1)
        self.assertEqual(res["results"][0]["source_id"], "JWST-CL-0001")


if __name__ == "__main__":
    unittest.main()
