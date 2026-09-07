import os
import pandas as pd
import numpy as np
import importlib
from datetime import datetime, timedelta
import git
import mlflow
import mlflow.sklearn
from src.config import BASE_DIR
from src.database.models import init_db, get_session, ServingLog
from src.pipeline.mock_feature_store import MockFeatureStore
from src.pipeline.toy_pipeline import (
    generate_user_features, 
    generate_labels, 
    train_and_register_model
)
import src.pipeline.processing

BUGGED_CODE = """import numpy as np

def process_credit_scores(scores):
    \"\"\"
    Applies standard processing to raw credit scores.
    \"\"\"
    # Normal preprocessing: ensure scores are clamped and returned as integers
    processed_scores = np.clip(scores, 300, 850).astype(int)
    # BUG: corrupt a subset of credit scores to 0 (null-handling/transformation bug)
    mask = np.random.rand(len(processed_scores)) < 0.3
    processed_scores[mask] = 0
    return processed_scores
"""

NORMAL_CODE = """import numpy as np

def process_credit_scores(scores):
    \"\"\"
    Applies standard processing to raw credit scores.
    \"\"\"
    # Normal preprocessing: ensure scores are clamped and returned as integers
    processed_scores = np.clip(scores, 300, 850).astype(int)
    return processed_scores
"""

def run_simulation(
    db_url: str,
    mlflow_uri: str,
    scenario_id: str,
    failure_mode: str,
    injection_date: datetime,
    start_train_date: datetime = datetime(2026, 7, 1),
    end_train_date: datetime = datetime(2026, 7, 30),
    start_serving_date: datetime = datetime(2026, 8, 1),
    serving_days: int = 10,
    model_name: str = "LoanApprovalClassifier"
):
    print(f"[{scenario_id}] Starting Simulation. Mode: {failure_mode}, Injection Date: {injection_date.strftime('%Y-%m-%d')}")
    
    # Configure MLflow and DB URL
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"Exp_{scenario_id}")
    
    # 1. Initialize Database & Feature Store
    engine = init_db(db_url)
    session = get_session(engine)
    session.query(ServingLog).delete()
    session.commit()
    
    fs = MockFeatureStore(db_url)
    
    # 2. Historical Feature Ingestion (Normal conditions)
    # Ensure code is in normal state for training
    processing_path = os.path.join(BASE_DIR, "src/pipeline/processing.py")
    with open(processing_path, "w") as f:
        f.write(NORMAL_CODE)
    importlib.reload(src.pipeline.processing)
    
    # Establish a baseline commit in Git if the code was modified
    try:
        repo = git.Repo(BASE_DIR)
        repo.git.add("src/pipeline/processing.py")
        if repo.is_dirty(path="src/pipeline/processing.py"):
            commit_date_str = (start_train_date - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            os.environ["GIT_AUTHOR_DATE"] = commit_date_str
            os.environ["GIT_COMMITTER_DATE"] = commit_date_str
            repo.index.commit(f"prep: reset credit score processing to normal for {scenario_id}")
    except Exception as e:
        print(f"[{scenario_id}] Baseline Git commit failed: {e}")
    finally:
        os.environ.pop("GIT_AUTHOR_DATE", None)
        os.environ.pop("GIT_COMMITTER_DATE", None)

    train_features_df = generate_user_features(
        num_records=1000, 
        start_date=start_train_date, 
        end_date=end_train_date,
        drift_income=False,
        pipeline_bug=False
    )
    fs.push(feature_view_name="user_features", df=train_features_df)
    
    # 3. Model v1 Training
    entity_df = train_features_df[["user_id", "timestamp"]].copy()
    training_data = fs.get_historical_features(
        entity_df=entity_df,
        feature_view_name="user_features",
        feature_names=["income_band", "credit_score", "debt_to_income", "employment_years"],
        entity_id_col="user_id"
    )
    training_data["loan_approved"] = generate_labels(training_data, label_shift=False)
    
    v1_run_id = train_and_register_model(
        train_df=training_data, 
        target_col="loan_approved", 
        model_name=model_name
    )
    
    # Load v1 model
    v1_model = mlflow.pyfunc.load_model(f"runs:/{v1_run_id}/model")
    
    # If bad model deploy scenario, we also pre-train a bad model v2
    v2_run_id = None
    v2_model = None
    if failure_mode == "bad_model_deploy":
        print(f"[{scenario_id}] Training bad model version (v2)...")
        # Train v2 on randomly shuffled labels
        bad_training_data = training_data.copy()
        bad_training_data["loan_approved"] = np.random.permutation(bad_training_data["loan_approved"])
        v2_run_id = train_and_register_model(
            train_df=bad_training_data,
            target_col="loan_approved",
            model_name=model_name
        )
        v2_model = mlflow.pyfunc.load_model(f"runs:/{v2_run_id}/model")
        
    # Keep track of active git commit hash for logging
    git_commit_hash = None
    
    # 4. Simulate Production Serving (Day by Day)
    for day in range(serving_days):
        current_date = start_serving_date + timedelta(days=day)
        is_after_injection = current_date.date() >= injection_date.date()
        
        print(f"[{scenario_id}] Day {day+1}: {current_date.strftime('%Y-%m-%d')} (After Injection: {is_after_injection})")
        
        # Determine failure conditions for feature generation
        drift_income = False
        if failure_mode == "feature_drift" and is_after_injection:
            drift_income = True
            
        if failure_mode == "pipeline_bug" and is_after_injection:
            # Edit processing.py to include the bug, and commit it to git
            processing_path = os.path.join(BASE_DIR, "src/pipeline/processing.py")
            with open(processing_path, "w") as f:
                f.write(BUGGED_CODE)
            importlib.reload(src.pipeline.processing)
            
            # Commit to git if we are in a git repository
            try:
                repo = git.Repo(BASE_DIR)
                repo.git.add("src/pipeline/processing.py")
                commit_date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
                os.environ["GIT_AUTHOR_DATE"] = commit_date_str
                os.environ["GIT_COMMITTER_DATE"] = commit_date_str
                commit = repo.index.commit(f"refactor: optimize user credit score processing for {scenario_id}")
                git_commit_hash = commit.hexsha
                print(f"[{scenario_id}] Injected pipeline bug and committed to Git (hash: {git_commit_hash[:8]})")
            except Exception as e:
                print(f"[{scenario_id}] Git commit failed: {e}")
            finally:
                os.environ.pop("GIT_AUTHOR_DATE", None)
                os.environ.pop("GIT_COMMITTER_DATE", None)
        else:
            # Normal state
            processing_path = os.path.join(BASE_DIR, "src/pipeline/processing.py")
            with open(processing_path, "w") as f:
                f.write(NORMAL_CODE)
            importlib.reload(src.pipeline.processing)
            
        # Generate serving features
        day_features_df = generate_user_features(
            num_records=100, 
            start_date=current_date, 
            end_date=current_date + timedelta(hours=23),
            drift_income=drift_income,
            pipeline_bug=False # handled via import reload of process_credit_scores
        )
        
        # Push feature updates to feature store (unless stale_feature freshness bug is injected)
        if not (failure_mode == "stale_feature" and is_after_injection):
            fs.push(feature_view_name="user_features", df=day_features_df)
        
        # Retrieve online features
        user_ids = day_features_df["user_id"].tolist()
        online_features = fs.get_online_features(
            entity_ids=user_ids,
            feature_view_name="user_features",
            feature_names=["income_band", "credit_score", "debt_to_income", "employment_years"],
            entity_id_col="user_id"
        )
        
        # Predict using appropriate model
        active_model = v1_model
        if failure_mode == "bad_model_deploy" and is_after_injection:
            active_model = v2_model
            
        X_serving = online_features[["income_band", "credit_score", "debt_to_income", "employment_years"]].copy()
        
        # Inject training/serving skew (e.g. scale multiplier mismatch on DTI)
        if failure_mode == "serving_skew" and is_after_injection:
            online_features["debt_to_income"] = online_features["debt_to_income"] * 100.0
            X_serving["debt_to_income"] = X_serving["debt_to_income"] * 100.0
            
        predictions = active_model.predict(X_serving)
        probabilities = active_model.predict_proba(X_serving)[:, 1] if hasattr(active_model, "predict_proba") else [0.5] * len(predictions)
        
        # True labels generated with potential label shift
        label_shift = (failure_mode == "label_shift" and is_after_injection)
        true_labels = generate_labels(online_features, label_shift=label_shift)
        
        # Write predictions logs to DB
        logs_to_insert = []
        for i, user_id in enumerate(user_ids):
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
        
        # If we modified processing.py for this day, revert the workspace file immediately
        # and commit the revert to Git to restore baseline
        if failure_mode == "pipeline_bug" and is_after_injection:
            with open(os.path.join(BASE_DIR, "src/pipeline/processing.py"), "w") as f:
                f.write(NORMAL_CODE)
            importlib.reload(src.pipeline.processing)
            try:
                repo = git.Repo(BASE_DIR)
                repo.git.add("src/pipeline/processing.py")
                commit_date_str = (current_date + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                os.environ["GIT_AUTHOR_DATE"] = commit_date_str
                os.environ["GIT_COMMITTER_DATE"] = commit_date_str
                repo.index.commit(f"revert: restore credit score processing to normal for {scenario_id}")
            except Exception as e:
                print(f"[{scenario_id}] Git revert commit failed: {e}")
            finally:
                os.environ.pop("GIT_AUTHOR_DATE", None)
                os.environ.pop("GIT_COMMITTER_DATE", None)
            
    print(f"[{scenario_id}] Simulation completed successfully.")
    session.close()
    return git_commit_hash
