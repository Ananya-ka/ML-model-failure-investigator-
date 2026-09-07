import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from typing import List, Union

class MockFeatureStore:
    def __init__(self, db_url: str):
        """
        Initializes the Mock Feature Store with an engine.
        Supports both SQLite and PostgreSQL.
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def push(self, feature_view_name: str, df: pd.DataFrame):
        """
        Pushes features into the specified feature view (table).
        The DataFrame must contain a 'timestamp' column and an entity ID column.
        """
        # Ensure timestamp is datetime type
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Write to SQL (append if exists)
        df.to_sql(name=f"fv_{feature_view_name}", con=self.engine, if_exists="append", index=False)

    def get_historical_features(
        self, 
        entity_df: pd.DataFrame, 
        feature_view_name: str, 
        feature_names: List[str], 
        entity_id_col: str = "user_id"
    ) -> pd.DataFrame:
        """
        Retrieves historical features for a set of entities at specific timestamps.
        Implements point-in-time correct (ASOF) join to prevent target leakage.
        """
        # Read the feature view data
        query = f"SELECT * FROM fv_{feature_view_name}"
        feature_df = pd.read_sql(query, con=self.engine)
        
        # Parse timestamps
        entity_df = entity_df.copy()
        entity_df['timestamp'] = pd.to_datetime(entity_df['timestamp'])
        feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])

        # Sort values for merge_asof
        entity_df = entity_df.sort_values('timestamp')
        feature_df = feature_df.sort_values('timestamp')

        # Select only the relevant feature columns plus entity and timestamp
        cols_to_keep = [entity_id_col, 'timestamp'] + [c for c in feature_names if c in feature_df.columns]
        feature_df_subset = feature_df[cols_to_keep]

        # Perform the point-in-time correct join
        merged_df = pd.merge_asof(
            entity_df,
            feature_df_subset,
            on='timestamp',
            by=entity_id_col,
            direction='backward'  # match with the latest feature value before or equal to entity timestamp
        )
        return merged_df

    def get_online_features(
        self, 
        entity_ids: List[Union[int, str]], 
        feature_view_name: str, 
        feature_names: List[str], 
        entity_id_col: str = "user_id"
    ) -> pd.DataFrame:
        """
        Retrieves the latest (online) features for the given list of entity IDs.
        """
        if not entity_ids:
            return pd.DataFrame()

        # Query latest records for the entity IDs
        # To handle multiple backend SQL dialects safely, we load the entities' records and filter in pandas
        # or execute a standard window query. Since SQLite and Postgres both support ROW_NUMBER(), we can use:
        ids_placeholder = ", ".join([f"'{eid}'" if isinstance(eid, str) else str(eid) for eid in entity_ids])
        query = f"""
            WITH ranked_features AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY {entity_id_col} ORDER BY timestamp DESC) as rn
                FROM fv_{feature_view_name}
                WHERE {entity_id_col} IN ({ids_placeholder})
            )
            SELECT * FROM ranked_features WHERE rn = 1
        """
        feature_df = pd.read_sql(query, con=self.engine)
        
        # Select only relevant columns
        cols_to_keep = [entity_id_col, 'timestamp'] + [c for c in feature_names if c in feature_df.columns]
        return feature_df[cols_to_keep]
