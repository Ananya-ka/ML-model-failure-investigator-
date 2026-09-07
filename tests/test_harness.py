import unittest
import os
import json
import shutil
import git
from datetime import datetime
from src.config import BASE_DIR
from src.harness.scenario_generator import generate_metadata, load_metadata
from src.pipeline.simulation import run_simulation
from src.database.models import init_db, get_session, ServingLog
import mlflow
import numpy as np

class TestFailureHarness(unittest.TestCase):
    def setUp(self):
        self.test_scenarios_dir = os.path.join(BASE_DIR, "test_scenarios")
        os.makedirs(self.test_scenarios_dir, exist_ok=True)
        
    def tearDown(self):
        # Clean up test directories
        if os.path.exists(self.test_scenarios_dir):
            shutil.rmtree(self.test_scenarios_dir)

    def test_metadata_generation(self):
        # Test generating 14 scenarios (2 of each type)
        scenarios = generate_metadata(num_scenarios=14)
        self.assertEqual(len(scenarios), 14)
        
        # Verify keys
        first_scenario = scenarios[0]
        self.assertIn("scenario_id", first_scenario)
        self.assertIn("injected_failure", first_scenario)
        self.assertIn("true_root_cause", first_scenario)
        self.assertIn("evidence_should_include", first_scenario)
        
        # Verify balancing
        failures = [s["injected_failure"] for s in scenarios]
        self.assertEqual(failures.count("control"), 2)
        self.assertEqual(failures.count("feature_drift"), 2)
        self.assertEqual(failures.count("bad_model_deploy"), 2)
        self.assertEqual(failures.count("pipeline_bug"), 2)
        self.assertEqual(failures.count("label_shift"), 2)
        self.assertEqual(failures.count("serving_skew"), 2)
        self.assertEqual(failures.count("stale_feature"), 2)

    def test_mini_simulation_control(self):
        db_url = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_control.db')}"
        mlflow_uri = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_mlflow_control.db')}"
        
        # Run a 2-day simulation with 5 records per day under normal conditions
        run_simulation(
            db_url=db_url,
            mlflow_uri=mlflow_uri,
            scenario_id="test_sc_control",
            failure_mode="control",
            injection_date=datetime(2026, 8, 5),
            start_serving_date=datetime(2026, 8, 4),
            serving_days=2,
            model_name="TestControlModel"
        )
        
        # Verify database is populated
        engine = init_db(db_url)
        session = get_session(engine)
        logs = session.query(ServingLog).all()
        self.assertEqual(len(logs), 200) # 100 per day
        session.close()

    def test_mini_simulation_drift(self):
        db_url = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_drift.db')}"
        mlflow_uri = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_mlflow_drift.db')}"
        
        # Run a 2-day simulation. Injection date is Day 2.
        # Day 1: 2026-08-04 (Normal)
        # Day 2: 2026-08-05 (Drift)
        run_simulation(
            db_url=db_url,
            mlflow_uri=mlflow_uri,
            scenario_id="test_sc_drift",
            failure_mode="feature_drift",
            injection_date=datetime(2026, 8, 5),
            start_serving_date=datetime(2026, 8, 4),
            serving_days=2,
            model_name="TestDriftModel"
        )
        
        # Verify database features. Under drift, income_band is shifted downward (more 1s and 2s).
        engine = init_db(db_url)
        session = get_session(engine)
        
        # Fetch day 1 logs (normal)
        day1_logs = session.query(ServingLog).filter(ServingLog.timestamp < datetime(2026, 8, 5)).all()
        # Fetch day 2 logs (drifted)
        day2_logs = session.query(ServingLog).filter(ServingLog.timestamp >= datetime(2026, 8, 5)).all()
        
        self.assertTrue(len(day1_logs) > 0)
        self.assertTrue(len(day2_logs) > 0)
        
        day1_income = [log.income_band for log in day1_logs]
        day2_income = [log.income_band for log in day2_logs]
        
        # Mean income band on day 2 should be significantly lower than day 1 due to drift
        self.assertLess(np.mean(day2_income), np.mean(day1_income))
        session.close()

    def test_mini_simulation_bug(self):
        db_url = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_bug.db')}"
        mlflow_uri = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_mlflow_bug.db')}"
        
        # Day 1: 2026-08-04 (Normal)
        # Day 2: 2026-08-05 (Bug Injected)
        git_hash = run_simulation(
            db_url=db_url,
            mlflow_uri=mlflow_uri,
            scenario_id="test_sc_bug",
            failure_mode="pipeline_bug",
            injection_date=datetime(2026, 8, 5),
            start_serving_date=datetime(2026, 8, 4),
            serving_days=2,
            model_name="TestBugModel"
        )
        
        # Verify database features. Under bug, a subset of credit scores on day 2 should be 0.
        engine = init_db(db_url)
        session = get_session(engine)
        
        day1_logs = session.query(ServingLog).filter(ServingLog.timestamp < datetime(2026, 8, 5)).all()
        day2_logs = session.query(ServingLog).filter(ServingLog.timestamp >= datetime(2026, 8, 5)).all()
        
        self.assertTrue(len(day1_logs) > 0)
        self.assertTrue(len(day2_logs) > 0)
        
        day1_scores = [log.credit_score for log in day1_logs]
        day2_scores = [log.credit_score for log in day2_logs]
        
        # Day 1 scores should all be >= 300 (normal range)
        self.assertTrue(all(s >= 300 for s in day1_scores))
        
        # Day 2 scores should contain some 0s due to the bug
        self.assertTrue(any(s == 0 for s in day2_scores))
        session.close()
        
        # Verify git commit was created and matches git_hash
        if git_hash:
            repo = git.Repo(BASE_DIR)
            commit = repo.commit(git_hash)
            self.assertIn("test_sc_bug", commit.message)

    def test_mini_simulation_label_shift(self):
        db_url = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_label_shift.db')}"
        mlflow_uri = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_mlflow_label_shift.db')}"
        
        run_simulation(
            db_url=db_url,
            mlflow_uri=mlflow_uri,
            scenario_id="test_sc_label_shift",
            failure_mode="label_shift",
            injection_date=datetime(2026, 8, 5),
            start_serving_date=datetime(2026, 8, 4),
            serving_days=2,
            model_name="TestLabelShiftModel"
        )
        
        engine = init_db(db_url)
        session = get_session(engine)
        
        day1_logs = session.query(ServingLog).filter(ServingLog.timestamp < datetime(2026, 8, 5)).all()
        day2_logs = session.query(ServingLog).filter(ServingLog.timestamp >= datetime(2026, 8, 5)).all()
        
        day1_labels = [log.true_label for log in day1_logs if log.true_label is not None]
        day2_labels = [log.true_label for log in day2_logs if log.true_label is not None]
        
        self.assertLess(np.mean(day2_labels), np.mean(day1_labels))
        session.close()

    def test_mini_simulation_serving_skew(self):
        db_url = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_skew.db')}"
        mlflow_uri = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_mlflow_skew.db')}"
        
        run_simulation(
            db_url=db_url,
            mlflow_uri=mlflow_uri,
            scenario_id="test_sc_skew",
            failure_mode="serving_skew",
            injection_date=datetime(2026, 8, 5),
            start_serving_date=datetime(2026, 8, 4),
            serving_days=2,
            model_name="TestSkewModel"
        )
        
        engine = init_db(db_url)
        session = get_session(engine)
        
        day1_logs = session.query(ServingLog).filter(ServingLog.timestamp < datetime(2026, 8, 5)).all()
        day2_logs = session.query(ServingLog).filter(ServingLog.timestamp >= datetime(2026, 8, 5)).all()
        
        self.assertTrue(len(day1_logs) > 0)
        self.assertTrue(len(day2_logs) > 0)
        
        day1_dti = [log.debt_to_income for log in day1_logs]
        day2_dti = [log.debt_to_income for log in day2_logs]
        
        self.assertGreater(np.mean(day2_dti), 10.0)
        self.assertLess(np.mean(day1_dti), 1.0)
        session.close()

    def test_mini_simulation_stale_feature(self):
        db_url = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_stale.db')}"
        mlflow_uri = f"sqlite:///{os.path.join(self.test_scenarios_dir, 'test_mlflow_stale.db')}"
        
        run_simulation(
            db_url=db_url,
            mlflow_uri=mlflow_uri,
            scenario_id="test_sc_stale",
            failure_mode="stale_feature",
            injection_date=datetime(2026, 8, 5),
            start_serving_date=datetime(2026, 8, 4),
            serving_days=2,
            model_name="TestStaleModel"
        )
        
        engine = init_db(db_url)
        session = get_session(engine)
        
        day2_logs = session.query(ServingLog).filter(ServingLog.timestamp >= datetime(2026, 8, 5)).all()
        self.assertTrue(len(day2_logs) > 0)
        
        from src.tools.diagnostics import query_feature_staleness
        res = query_feature_staleness(db_url, datetime(2026, 8, 5), datetime(2026, 8, 5, 23, 59, 59))
        self.assertTrue(res["is_stale"])
        self.assertGreater(res["mean_lag_days"], 0.5)
        session.close()

if __name__ == "__main__":
    unittest.main()
