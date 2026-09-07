import unittest
import os
from src.config import BASE_DIR
from src.harness.scenario_generator import load_metadata
from src.harness.eval_harness import evaluate_scenario, print_summary_table

class TestEvalHarness(unittest.TestCase):
    def setUp(self):
        os.environ["FORCE_MOCK"] = "true"

    def test_evaluate_single_scenario(self):
        metadata = load_metadata()
        self.assertTrue(len(metadata) > 0)
        
        # Test evaluation of the first scenario sc_001 (control)
        sc1 = metadata[0]
        self.assertEqual(sc1["scenario_id"], "sc_001")
        
        # Run evaluation
        res = evaluate_scenario(sc1)
        
        # Verify result fields
        self.assertEqual(res["scenario_id"], "sc_001")
        self.assertEqual(res["failure_mode"], "control")
        self.assertEqual(res["predicted_cause"], "control")
        self.assertTrue(res["is_correct"])
        self.assertIn("confidence", res)
        self.assertIn("latency_sec", res)
        self.assertIn("evidence_correct", res)
        self.assertTrue(res["evidence_correct"])
        self.assertIn("reasoning", res)

    def test_summary_table_rendering(self):
        # Create dummy evaluation results list
        dummy_results = [
            {
                "scenario_id": "sc_001",
                "failure_mode": "control",
                "predicted_cause": "control",
                "is_correct": True,
                "confidence": 1.0,
                "latency_sec": 1.5,
                "evidence_correct": True
            },
            {
                "scenario_id": "sc_002",
                "failure_mode": "feature_drift",
                "predicted_cause": "feature_drift",
                "is_correct": True,
                "confidence": 0.95,
                "latency_sec": 2.1,
                "evidence_correct": True
            },
            {
                "scenario_id": "sc_003",
                "failure_mode": "bad_model_deploy",
                "predicted_cause": "control",  # False prediction
                "is_correct": False,
                "confidence": 0.5,
                "latency_sec": 1.9,
                "evidence_correct": False
            }
        ]
        
        # Ensure print_summary_table runs successfully without throwing exceptions
        try:
            print_summary_table(dummy_results)
            success = True
        except Exception as e:
            print(f"Summary table rendering failed: {e}")
            success = False
            
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
