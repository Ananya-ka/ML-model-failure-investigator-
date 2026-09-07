import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.pipeline.mock_feature_store import MockFeatureStore

class TestMockFeatureStore(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_url = "sqlite:///:memory:"
        self.fs = MockFeatureStore(self.db_url)
        
        # Seed feature data for usr_101
        self.feature_data = pd.DataFrame([
            {"user_id": "usr_101", "timestamp": "2026-07-01 10:00:00", "income_band": 3, "credit_score": 600},
            {"user_id": "usr_101", "timestamp": "2026-07-05 10:00:00", "income_band": 4, "credit_score": 650},
            {"user_id": "usr_101", "timestamp": "2026-07-10 10:00:00", "income_band": 5, "credit_score": 700},
        ])
        # Push to feature view "user_features"
        self.fs.push("user_features", self.feature_data)

    def test_asof_joins(self):
        # Query 1: usr_101 at 2026-07-03 (should get features from 2026-07-01: income_band=3, credit_score=600)
        entity_df_1 = pd.DataFrame([
            {"user_id": "usr_101", "timestamp": datetime(2026, 7, 3, 12, 0, 0)}
        ])
        res_1 = self.fs.get_historical_features(
            entity_df=entity_df_1,
            feature_view_name="user_features",
            feature_names=["income_band", "credit_score"],
            entity_id_col="user_id"
        )
        self.assertEqual(res_1.loc[0, "income_band"], 3)
        self.assertEqual(res_1.loc[0, "credit_score"], 600)

        # Query 2: usr_101 at 2026-07-06 (should get features from 2026-07-05: income_band=4, credit_score=650)
        entity_df_2 = pd.DataFrame([
            {"user_id": "usr_101", "timestamp": datetime(2026, 7, 6, 12, 0, 0)}
        ])
        res_2 = self.fs.get_historical_features(
            entity_df=entity_df_2,
            feature_view_name="user_features",
            feature_names=["income_band", "credit_score"],
            entity_id_col="user_id"
        )
        self.assertEqual(res_2.loc[0, "income_band"], 4)
        self.assertEqual(res_2.loc[0, "credit_score"], 650)

        # Query 3: usr_101 at 2026-07-12 (should get features from 2026-07-10: income_band=5, credit_score=700)
        entity_df_3 = pd.DataFrame([
            {"user_id": "usr_101", "timestamp": datetime(2026, 7, 12, 12, 0, 0)}
        ])
        res_3 = self.fs.get_historical_features(
            entity_df=entity_df_3,
            feature_view_name="user_features",
            feature_names=["income_band", "credit_score"],
            entity_id_col="user_id"
        )
        self.assertEqual(res_3.loc[0, "income_band"], 5)
        self.assertEqual(res_3.loc[0, "credit_score"], 700)

        # Query 4: usr_101 at 2026-06-30 (should get NaN since it is before the first record)
        entity_df_4 = pd.DataFrame([
            {"user_id": "usr_101", "timestamp": datetime(2026, 6, 30, 12, 0, 0)}
        ])
        res_4 = self.fs.get_historical_features(
            entity_df=entity_df_4,
            feature_view_name="user_features",
            feature_names=["income_band", "credit_score"],
            entity_id_col="user_id"
        )
        self.assertTrue(pd.isna(res_4.loc[0, "income_band"]))
        self.assertTrue(pd.isna(res_4.loc[0, "credit_score"]))

    def test_online_features(self):
        # Online features should get latest values: income_band=5, credit_score=700
        res = self.fs.get_online_features(
            entity_ids=["usr_101"],
            feature_view_name="user_features",
            feature_names=["income_band", "credit_score"],
            entity_id_col="user_id"
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res.loc[0, "income_band"], 5)
        self.assertEqual(res.loc[0, "credit_score"], 700)

if __name__ == "__main__":
    unittest.main()
