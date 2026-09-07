import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.config import DATABASE_URL, MLFLOW_TRACKING_URI
from src.database.models import init_db, get_session, ServingLog
from src.pipeline.mock_feature_store import MockFeatureStore
from src.pipeline.toy_pipeline import (
    generate_user_features, 
    generate_labels, 
    train_and_register_model
)
import mlflow

def run_simulation():
    print("=" * 60)
    print("STARTING BASELINE ML PIPELINE SIMULATION")
    print("=" * 60)
    
    # 1. Initialize Database & Feature Store
    print("[1/4] Initializing Database & Feature Store...")
    engine = init_db(DATABASE_URL)
    session = get_session(engine)
    
    # Clean up previous serving logs if any (to start fresh)
    session.query(ServingLog).delete()
    session.commit()
    
    fs = MockFeatureStore(DATABASE_URL)
    
    # 2. Historical Feature Ingestion
    print("[2/4] Simulating Historical Feature Ingestion...")
    start_train_date = datetime(2026, 7, 1)
    end_train_date = datetime(2026, 7, 30)
    
    # Generate 1000 users for training
    train_features_df = generate_user_features(
        num_records=1000, 
        start_date=start_train_date, 
        end_date=end_train_date,
        drift_income=False,
        pipeline_bug=False
    )
    
    # Push to feature store
    fs.push(feature_view_name="user_features", df=train_features_df)
    print(f"Pushed {len(train_features_df)} user feature records to the Mock Feature Store.")

    # 3. Model Training
    print("[3/4] Running Model Training & MLflow Registration...")
    # Retrieve historical features for training (ensuring point-in-time correctness)
    entity_df = train_features_df[["user_id", "timestamp"]].copy()
    
    # Fetch historical features
    training_data = fs.get_historical_features(
        entity_df=entity_df,
        feature_view_name="user_features",
        feature_names=["income_band", "credit_score", "debt_to_income", "employment_years"],
        entity_id_col="user_id"
    )
    
    # Generate ground-truth labels for training
    training_data["loan_approved"] = generate_labels(training_data, label_shift=False)
    
    # Train and register model in MLflow
    run_id = train_and_register_model(
        train_df=training_data, 
        target_col="loan_approved", 
        model_name="LoanApprovalClassifier"
    )
    
    # 4. Simulate Production Serving (Days 1 to 10)
    print("[4/4] Simulating Production Serving (10 Days of Normal Operation)...")
    
    # Load model from MLflow (using run ID path is 100% reliable in local SQLite environment)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    
    start_serving_date = datetime(2026, 8, 1)
    
    for day in range(10):
        current_date = start_serving_date + timedelta(days=day)
        print(f"\n--- Day {day+1}: {current_date.strftime('%Y-%m-%d')} ---")
        
        # Generate new serving requests (100 users per day)
        day_features_df = generate_user_features(
            num_records=100, 
            start_date=current_date, 
            end_date=current_date + timedelta(hours=23),
            drift_income=False,
            pipeline_bug=False
        )
        
        # Ingest feature updates into feature store
        fs.push(feature_view_name="user_features", df=day_features_df)
        
        # Retrieve online features for prediction
        user_ids = day_features_df["user_id"].tolist()
        online_features = fs.get_online_features(
            entity_ids=user_ids,
            feature_view_name="user_features",
            feature_names=["income_band", "credit_score", "debt_to_income", "employment_years"],
            entity_id_col="user_id"
        )
        
        # Prepare inputs for prediction (matching features schema)
        X_serving = online_features[["income_band", "credit_score", "debt_to_income", "employment_years"]]
        
        # Run predictions
        predictions = model.predict(X_serving)
        probabilities = model.predict_proba(X_serving)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(predictions)
        
        # Generate true labels (simulated outcome available shortly after)
        true_labels = generate_labels(online_features, label_shift=False)
        
        # Store logs in database
        logs_to_insert = []
        for i, user_id in enumerate(user_ids):
            # Find the features for this user
            user_feat = online_features[online_features["user_id"] == user_id].iloc[0]
            
            log_entry = ServingLog(
                user_id=user_id,
                timestamp=current_date + timedelta(minutes=int(np.random.randint(0, 1440))),
                income_band=int(user_feat["income_band"]),
                credit_score=int(user_feat["credit_score"]),
                debt_to_income=float(user_feat["debt_to_income"]),
                employment_years=int(user_feat["employment_years"]),
                prediction=int(predictions[i]),
                probability=float(probabilities[i]),
                true_label=int(true_labels.iloc[i])
            )
            logs_to_insert.append(log_entry)
            
        session.bulk_save_objects(logs_to_insert)
        session.commit()
        
        # Calculate daily metrics
        accuracy = (predictions == true_labels).mean()
        print(f"Logged {len(logs_to_insert)} serving requests. Daily Accuracy: {accuracy:.4f}")

    print("\n" + "=" * 60)
    print("BASELINE ML PIPELINE SIMULATION COMPLETED")
    print("=" * 60)
    session.close()

if __name__ == "__main__":
    run_simulation()
