import unittest
import os
from fastapi.testclient import TestClient
from src.api.main import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        os.environ["FORCE_MOCK"] = "true"
        self.client = TestClient(app)

    def test_get_scenarios(self):
        response = self.client.get("/api/scenarios")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(len(data) > 0)
        
        # Verify first scenario structure
        first_sc = data[0]
        self.assertEqual(first_sc["scenario_id"], "sc_001")
        self.assertIn("injected_failure", first_sc)
        self.assertIn("is_materialized", first_sc)

    def test_scenario_investigation_and_logs(self):
        # Run investigation on control scenario
        resp_investigate = self.client.post("/api/scenarios/sc_001/investigate")
        self.assertEqual(resp_investigate.status_code, 200)
        inv_data = resp_investigate.json()
        self.assertEqual(inv_data["cause"], "control")
        self.assertEqual(inv_data["confidence"], 1.0)
        
        # Retrieve logs
        resp_logs = self.client.get("/api/scenarios/sc_001/logs")
        self.assertEqual(resp_logs.status_code, 200)
        logs_data = resp_logs.json()
        self.assertTrue(len(logs_data) > 0)
        
        # Verify details of logged findings
        latest_log = logs_data[0]
        self.assertEqual(latest_log["detected_root_cause"], "control")
        self.assertEqual(latest_log["confidence"], 1.0)
        self.assertTrue(len(latest_log["evidence"]) > 0)

    def test_run_evaluations_api(self):
        # Evaluate on the first scenario
        response = self.client.post("/api/evaluations", json={"limit": 1})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("summary", data)
        self.assertIn("breakdown", data)
        self.assertEqual(data["summary"]["total_evaluated"], 1)
        self.assertEqual(data["summary"]["overall_accuracy"], 100.0)

if __name__ == "__main__":
    unittest.main()
