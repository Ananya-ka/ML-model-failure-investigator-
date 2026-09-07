import unittest
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime, timezone, timedelta
from src.pipeline.toy_pipeline import generate_user_features, generate_labels, train_and_register_model
from src.database.models import init_db, get_session, ServingLog
from src.config import DATABASE_URL

class TestMLPipeline(unittest.TestCase):
    def test_data_generation(self):
        # Generate user features
        num_records = 50
        df = generate_user_features(
            num_records=num_records,
            start_date=datetime(2026, 8, 1),
            end_date=datetime(2026, 8, 2),
            drift_income=False,
            pipeline_bug=False
        )
        self.assertEqual(len(df), num_records)
        self.assertIn("user_id", df.columns)
        self.assertIn("income_band", df.columns)
        self.assertIn("credit_score", df.columns)
        
        # Test normal range
        self.assertTrue(df["credit_score"].min() >= 300)
        self.assertTrue(df["credit_score"].max() <= 850)
        self.assertTrue(df["income_band"].min() >= 1)
        self.assertTrue(df["income_band"].max() <= 5)

    def test_label_generation(self):
        num_records = 20
        df = generate_user_features(
            num_records=num_records,
            start_date=datetime(2026, 8, 1),
            end_date=datetime(2026, 8, 2),
            drift_income=False,
            pipeline_bug=False
        )
        labels = generate_labels(df, label_shift=False)
        self.assertEqual(len(labels), num_records)
        self.assertTrue(set(labels.unique()).issubset({0, 1}))

    def test_db_serving_log(self):
        # Connect to temporary in-memory DB for test
        engine = init_db("sqlite:///:memory:")
        session = get_session(engine)
        
        log = ServingLog(
            user_id="usr_123",
            timestamp=datetime.now(timezone.utc),
            income_band=3,
            credit_score=720,
            debt_to_income=0.35,
            employment_years=5,
            prediction=1,
            probability=0.88,
            true_label=1
        )
        session.add(log)
        session.commit()
        
        # Query back
        retrieved = session.query(ServingLog).filter_by(user_id="usr_123").first()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.credit_score, 720)
        self.assertEqual(retrieved.prediction, 1)
        session.close()

if __name__ == "__main__":
    unittest.main()
