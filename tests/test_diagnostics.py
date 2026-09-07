import unittest
import os
from datetime import datetime
from src.config import BASE_DIR
from src.tools.diagnostics import (
    query_feature_store_drift, 
    query_model_registry_diff, 
    query_git_log,
    query_label_drift,
    query_feature_staleness
)

class TestDiagnosticsTools(unittest.TestCase):
    def setUp(self):
        # Base paths for the generated scenarios
        self.sc_001_dir = os.path.join(BASE_DIR, "scenarios", "sc_001")
        self.sc_002_dir = os.path.join(BASE_DIR, "scenarios", "sc_002")
        self.sc_003_dir = os.path.join(BASE_DIR, "scenarios", "sc_003")
        self.sc_004_dir = os.path.join(BASE_DIR, "scenarios", "sc_004")

    def test_drift_tool_control_vs_drift(self):
        # Define baseline (July) and target (August) date ranges
        baseline_start = datetime(2026, 7, 1)
        baseline_end = datetime(2026, 7, 30)
        target_start = datetime(2026, 8, 1)
        target_end = datetime(2026, 8, 10)
        
        # Test Control (sc_001) - Should NOT have significant drift for income_band
        db_url_sc1 = f"sqlite:///{os.path.join(self.sc_001_dir, 'ml_failure_investigator.db')}"
        res_sc1 = query_feature_store_drift(
            db_url=db_url_sc1,
            feature_name="income_band",
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            target_start=target_start,
            target_end=target_end
        )
        self.assertFalse(res_sc1.get("error"))
        self.assertFalse(res_sc1["is_drifted"], f"Control scenario should not drift: {res_sc1}")
        
        # Test Feature Drift (sc_002) - Should have SIGNIFICANT drift for income_band
        db_url_sc2 = f"sqlite:///{os.path.join(self.sc_002_dir, 'ml_failure_investigator.db')}"
        res_sc2 = query_feature_store_drift(
            db_url=db_url_sc2,
            feature_name="income_band",
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            target_start=target_start,
            target_end=target_end
        )
        self.assertFalse(res_sc2.get("error"))
        self.assertTrue(res_sc2["is_drifted"], f"Drift scenario should detect drift: {res_sc2}")

    def test_model_registry_diff_control_vs_deploy(self):
        # Test Control (sc_001) - Should only have 1 model version
        mlflow_uri_sc1 = f"sqlite:///{os.path.join(self.sc_001_dir, 'mlflow.db')}"
        res_sc1 = query_model_registry_diff(mlflow_uri=mlflow_uri_sc1, model_name="LoanApprovalClassifier")
        self.assertFalse(res_sc1.get("error"))
        self.assertEqual(res_sc1.get("version_count"), 1)
        self.assertIn("latest_version_params", res_sc1)
        
        # Test Bad Deploy (sc_003) - Should have 2 model versions with comparison metrics
        mlflow_uri_sc3 = f"sqlite:///{os.path.join(self.sc_003_dir, 'mlflow.db')}"
        res_sc3 = query_model_registry_diff(mlflow_uri=mlflow_uri_sc3, model_name="LoanApprovalClassifier")
        self.assertFalse(res_sc3.get("error"))
        self.assertEqual(res_sc3.get("latest_version"), "2")
        self.assertEqual(res_sc3.get("previous_version"), "1")
        self.assertIn("param_changes", res_sc3)
        self.assertIn("metric_changes", res_sc3)
        
        # In our simulation, bad model is trained on random labels, so F1/accuracy should drop
        metric_changes = res_sc3["metric_changes"]
        if "accuracy" in metric_changes:
            # We expect accuracy delta to be negative (performance dropped)
            self.assertLessEqual(metric_changes["accuracy"]["delta"], 0)

    def test_git_log_bug(self):
        # Pipeline bug scenario (sc_004) has commits between August 7 and August 11, 2026
        start_date = datetime(2026, 8, 7)
        end_date = datetime(2026, 8, 11)
        
        commits = query_git_log(
            repo_path=BASE_DIR,
            start_date=start_date,
            end_date=end_date,
            filepath_filter="src/pipeline/processing.py"
        )
        
        self.assertTrue(len(commits) > 0, "Should find pipeline bug commits in Git log")
        
        # Verify first commit details
        first_commit = commits[0]
        self.assertIn("commit_hash", first_commit)
        self.assertIn("author", first_commit)
        self.assertIn("diffs", first_commit)
        
        # The commit message should contain the scenario ID
        self.assertTrue(any("sc_004" in c["message"] for c in commits))
        
        # Verify code diff text contains the bug injection indicator
        diff_texts = [d["diff_text"] for c in commits for d in c["diffs"]]
        self.assertTrue(any("BUG: corrupt a subset of credit scores to 0" in text for text in diff_texts))

    def test_label_drift_and_staleness_tools(self):
        db_url_sc1 = f"sqlite:///{os.path.join(self.sc_001_dir, 'ml_failure_investigator.db')}"
        baseline_start = datetime(2026, 8, 1)
        baseline_end = datetime(2026, 8, 4)
        target_start = datetime(2026, 8, 5)
        target_end = datetime(2026, 8, 10)
        
        res_label = query_label_drift(
            db_url=db_url_sc1,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            target_start=target_start,
            target_end=target_end
        )
        self.assertFalse(res_label.get("error"))
        self.assertFalse(res_label["is_drifted"])
        
        res_stale = query_feature_staleness(
            db_url=db_url_sc1,
            target_start=target_start,
            target_end=target_end
        )
        self.assertFalse(res_stale.get("error"))
        self.assertFalse(res_stale["is_stale"])

if __name__ == "__main__":
    unittest.main()
