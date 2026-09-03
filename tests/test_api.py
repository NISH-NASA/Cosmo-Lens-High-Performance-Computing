"""
FastAPI Server Endpoint Tests.
"""

import unittest
from fastapi.testclient import TestClient
from cosmolens.server.main import app


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("cpu_cores", data)

    def test_hpc_run_endpoint(self):
        res = self.client.post("/api/hpc/run", json={"threshold_sigma": 3.5, "cutout_size": 32})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("telemetry", data)

    def test_deepfield_and_sources(self):
        # Deepfield endpoint
        df_res = self.client.get("/api/deepfield")
        self.assertEqual(df_res.status_code, 200)
        self.assertIn("mosaic_b64", df_res.json())

        # Sources endpoint
        src_res = self.client.get("/api/sources")
        self.assertEqual(src_res.status_code, 200)
        sources = src_res.json()["sources"]
        self.assertGreater(len(sources), 0)

        # Single source inspection
        top_id = sources[0]["id"]
        detail_res = self.client.get(f"/api/source/{top_id}")
        self.assertEqual(detail_res.status_code, 200)

        # Gemini analyze endpoint
        ai_res = self.client.post(f"/api/gemini/analyze/{top_id}")
        self.assertEqual(ai_res.status_code, 200)
        self.assertIn("analysis", ai_res.json())

        # Discovery report endpoint
        rep_res = self.client.get(f"/api/gemini/report/{top_id}")
        self.assertEqual(rep_res.status_code, 200)
        self.assertIn("memo_markdown", rep_res.json())

    def test_semantic_search_endpoint(self):
        res = self.client.post("/api/gemini/search", json={"query": "find einstein rings"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("results", data)


if __name__ == "__main__":
    unittest.main()
