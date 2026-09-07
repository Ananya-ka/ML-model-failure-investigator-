import unittest
import os
from datetime import datetime
from src.config import BASE_DIR
from src.agent.agent import run_agent_investigation
from src.database.models import init_db, get_session, InvestigationLog, EvidenceRecord

class TestInvestigationAgent(unittest.TestCase):
    def setUp(self):
        os.environ["FORCE_MOCK"] = "true"

    def test_agent_control_sc_001(self):
        # Run agent on Control scenario
        root_cause = run_agent_investigation(
            scenario_id="sc_001",
            alert_metric="accuracy",
            alert_value=0.79,
            alert_date=datetime(2026, 8, 8)
        )
        self.assertEqual(root_cause["cause"], "control")
        self.assertEqual(root_cause["confidence"], 1.0)
        self.assertIn("bad_model_deploy", root_cause["ruled_out"])
        
        # Verify database logging
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_001', 'ml_failure_investigator.db')}"
        engine = init_db(db_url)
        session = get_session(engine)
        
        # Check logs
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        self.assertIsNotNone(inv_log)
        self.assertEqual(inv_log.detected_root_cause, "control")
        self.assertEqual(inv_log.confidence, 1.0)
        
        session.close()

    def test_agent_drift_sc_002(self):
        # Run agent on Feature Drift scenario
        root_cause = run_agent_investigation(
            scenario_id="sc_002",
            alert_metric="accuracy",
            alert_value=0.75,
            alert_date=datetime(2026, 8, 8)
        )
        self.assertEqual(root_cause["cause"], "feature_drift")
        self.assertGreaterEqual(root_cause["confidence"], 0.9)
        self.assertIn("income_band", root_cause["reasoning"])
        
        # Verify database logging and evidence recording
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_002', 'ml_failure_investigator.db')}"
        engine = init_db(db_url)
        session = get_session(engine)
        
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        self.assertIsNotNone(inv_log)
        self.assertEqual(inv_log.detected_root_cause, "feature_drift")
        
        evidence_recs = session.query(EvidenceRecord).filter_by(investigation_id=inv_log.investigation_id).all()
        self.assertTrue(len(evidence_recs) > 0)
        
        # Verify at least one evidence item registers drift
        self.assertTrue(any(e.evidence_type == "feature_drift" for e in evidence_recs))
        session.close()

    def test_agent_deploy_sc_003(self):
        # Run agent on Bad Deploy scenario
        root_cause = run_agent_investigation(
            scenario_id="sc_003",
            alert_metric="accuracy",
            alert_value=0.72,
            alert_date=datetime(2026, 8, 8)
        )
        self.assertEqual(root_cause["cause"], "bad_model_deploy")
        self.assertGreaterEqual(root_cause["confidence"], 0.9)
        
        # Verify database logs
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_003', 'ml_failure_investigator.db')}"
        engine = init_db(db_url)
        session = get_session(engine)
        
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        self.assertIsNotNone(inv_log)
        self.assertEqual(inv_log.detected_root_cause, "bad_model_deploy")
        
        evidence_recs = session.query(EvidenceRecord).filter_by(investigation_id=inv_log.investigation_id).all()
        self.assertTrue(any(e.evidence_type == "model_registry_diff" for e in evidence_recs))
        session.close()

    def test_agent_bug_sc_004(self):
        # Run agent on Pipeline Bug scenario
        root_cause = run_agent_investigation(
            scenario_id="sc_004",
            alert_metric="accuracy",
            alert_value=0.70,
            alert_date=datetime(2026, 8, 8)
        )
        self.assertEqual(root_cause["cause"], "pipeline_bug")
        self.assertGreaterEqual(root_cause["confidence"], 0.9)
        self.assertIn("credit_score", root_cause["reasoning"])
        
        # Verify database logs and git commit diff evidence
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_004', 'ml_failure_investigator.db')}"
        engine = init_db(db_url)
        session = get_session(engine)
        
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        self.assertIsNotNone(inv_log)
        self.assertEqual(inv_log.detected_root_cause, "pipeline_bug")
        
        evidence_recs = session.query(EvidenceRecord).filter_by(investigation_id=inv_log.investigation_id).all()
        # Verify that git commit changes were captured as evidence
        self.assertTrue(any(e.evidence_type == "git_commit" for e in evidence_recs))
        session.close()

    def test_agent_label_shift_sc_005(self):
        root_cause = run_agent_investigation(
            scenario_id="sc_005",
            alert_metric="accuracy",
            alert_value=0.72,
            alert_date=datetime(2026, 8, 8)
        )
        self.assertEqual(root_cause["cause"], "label_shift")
        self.assertGreaterEqual(root_cause["confidence"], 0.9)
        self.assertIn("approval rate", root_cause["reasoning"])
        
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_005', 'ml_failure_investigator.db')}"
        engine = init_db(db_url)
        session = get_session(engine)
        
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        self.assertIsNotNone(inv_log)
        self.assertEqual(inv_log.detected_root_cause, "label_shift")
        
        evidence_recs = session.query(EvidenceRecord).filter_by(investigation_id=inv_log.investigation_id).all()
        self.assertTrue(any(e.evidence_type == "label_drift" for e in evidence_recs))
        session.close()

    def test_agent_serving_skew_sc_006(self):
        root_cause = run_agent_investigation(
            scenario_id="sc_006",
            alert_metric="accuracy",
            alert_value=0.71,
            alert_date=datetime(2026, 8, 8)
        )
        self.assertEqual(root_cause["cause"], "serving_skew")
        self.assertGreaterEqual(root_cause["confidence"], 0.9)
        self.assertIn("debt_to_income", root_cause["reasoning"])
        
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_006', 'ml_failure_investigator.db')}"
        engine = init_db(db_url)
        session = get_session(engine)
        
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        self.assertIsNotNone(inv_log)
        self.assertEqual(inv_log.detected_root_cause, "serving_skew")
        
        evidence_recs = session.query(EvidenceRecord).filter_by(investigation_id=inv_log.investigation_id).all()
        self.assertTrue(any(e.evidence_type == "feature_drift" for e in evidence_recs))
        session.close()

    def test_agent_stale_feature_sc_007(self):
        root_cause = run_agent_investigation(
            scenario_id="sc_007",
            alert_metric="accuracy",
            alert_value=0.74,
            alert_date=datetime(2026, 8, 8)
        )
        self.assertEqual(root_cause["cause"], "stale_feature")
        self.assertGreaterEqual(root_cause["confidence"], 0.9)
        self.assertIn("lag", root_cause["reasoning"])
        
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_007', 'ml_failure_investigator.db')}"
        engine = init_db(db_url)
        session = get_session(engine)
        
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        self.assertIsNotNone(inv_log)
        self.assertEqual(inv_log.detected_root_cause, "stale_feature")
        
        evidence_recs = session.query(EvidenceRecord).filter_by(investigation_id=inv_log.investigation_id).all()
        self.assertTrue(any(e.evidence_type == "feature_staleness" for e in evidence_recs))
        session.close()

    def test_dynamic_feature_discovery(self):
        from src.agent.agent import get_feature_names
        db_url = f"sqlite:///{os.path.join(BASE_DIR, 'scenarios', 'sc_001', 'ml_failure_investigator.db')}"
        feature_cols = get_feature_names(db_url)
        self.assertIn("income_band", feature_cols)
        self.assertIn("credit_score", feature_cols)
        self.assertIn("debt_to_income", feature_cols)
        self.assertIn("employment_years", feature_cols)
        self.assertNotIn("user_id", feature_cols)
        self.assertNotIn("timestamp", feature_cols)

    def test_evidence_heuristics_generalization(self):
        from src.agent.agent import evaluate_evidence_heuristics
        # Test drift evidence on a custom feature name
        custom_drift_evidence = [
            {
                "type": "feature_drift",
                "description": "Feature 'income_band' mean shifted",
                "data": {"feature_name": "income_band", "is_drifted": True, "p_value": 0.001}
            }
        ]
        result = evaluate_evidence_heuristics(custom_drift_evidence)
        self.assertEqual(result["cause"], "feature_drift")
        self.assertIn("income_band", result["reasoning"])
        self.assertEqual(result["proven_hyp_id"], "h1")

        # Test pipeline bug evidence without any scenario ID
        custom_bug_evidence = [
            {
                "type": "git_commit",
                "description": "Commit by dev: BUG: corrupted credit_score calculation",
                "data": {"message": "BUG: credit_score"}
            },
            {
                "type": "feature_drift",
                "description": "Feature 'credit_score' mean shifted to 0",
                "data": {"feature_name": "credit_score", "is_drifted": True, "p_value": 0.0001}
            }
        ]
        bug_result = evaluate_evidence_heuristics(custom_bug_evidence)
        self.assertEqual(bug_result["cause"], "pipeline_bug")
        self.assertEqual(bug_result["proven_hyp_id"], "h3")

if __name__ == "__main__":
    unittest.main()

