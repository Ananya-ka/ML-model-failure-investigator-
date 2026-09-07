import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
from datetime import datetime, timedelta
from src.config import MLFLOW_TRACKING_URI, DATABASE_URL
from src.pipeline.mock_feature_store import MockFeatureStore

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Loan_Approval_Classifier")

def generate_user_features(
    num_records: int, 
    start_date: datetime, 
    end_date: datetime, 
    drift_income: bool = False,
    pipeline_bug: bool = False
) -> pd.DataFrame:
    """
    Generates synthetic user features.
    - If drift_income=True, income_band distribution is shifted down.
    - If pipeline_bug=True, credit_score is corrupted (e.g., set to 0 or scaled incorrectly).
    """
    np.random.seed(42)
    
    # Generate user IDs
    user_ids = [f"usr_{100000 + i}" for i in range(num_records)]
    
    # Generate timestamps evenly distributed between start_date and end_date
    delta = end_date - start_date
    timestamps = [start_date + timedelta(seconds=int(np.random.randint(0, int(delta.total_seconds())))) for _ in range(num_records)]
    timestamps = sorted(timestamps)
    
    # Generate income_band (normally 1 to 5, centered around 3.5)
    if drift_income:
        # Shifting distribution downwards (drift: average income decreases)
        income_bands = np.random.choice([1, 2, 3, 4, 5], size=num_records, p=[0.4, 0.3, 0.2, 0.08, 0.02])
    else:
        income_bands = np.random.choice([1, 2, 3, 4, 5], size=num_records, p=[0.05, 0.15, 0.4, 0.3, 0.1])
        
    # Generate credit_score
    raw_credit_scores = np.random.normal(loc=680, scale=70, size=num_records)
    from src.pipeline.processing import process_credit_scores
    credit_scores = process_credit_scores(raw_credit_scores)
        
    # Generate debt_to_income (DTI) ratio (typically between 0.1 and 0.6)
    debt_to_incomes = np.random.beta(a=2, b=5, size=num_records) * 0.9 + 0.1
    
    # Generate employment_years (typically between 0 and 30)
    employment_years = np.random.exponential(scale=7, size=num_records)
    employment_years = np.clip(employment_years, 0, 40).astype(int)
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "timestamp": timestamps,
        "income_band": income_bands,
        "credit_score": credit_scores,
        "debt_to_income": debt_to_incomes,
        "employment_years": employment_years
    })
    return df

def generate_labels(df: pd.DataFrame, label_shift: bool = False) -> pd.Series:
    """
    Generates loan approval labels (0 or 1) based on user features.
    - If label_shift=True, approvals are much harder to get (base rate drops).
    """
    # Simple probability logic
    # Higher credit score, income, employment_years increase approval chance.
    # Higher DTI decreases approval chance.
    z = (
        0.012 * (df["credit_score"] - 600) + 
        0.7 * df["income_band"] - 
        3.2 * df["debt_to_income"] + 
        0.06 * df["employment_years"] - 
        1.2
    )
    
    # Sigmoid function for probability
    probs = 1 / (1 + np.exp(-z))
    
    if label_shift:
        # Shift the threshold or scale probabilities down to simulate a macro drop in approvals
        probs = probs * 0.4
        
    # Generate labels using probabilities
    labels = (np.random.rand(len(df)) < probs).astype(int)
    return labels

def train_and_register_model(
    train_df: pd.DataFrame, 
    target_col: str = "loan_approved",
    model_name: str = "LoanApprovalClassifier"
):
    """
    Trains a Random Forest classifier and logs parameters/metrics to MLflow.
    Registers the model under model_name.
    """
    # Exclude non-feature columns
    feature_cols = ["income_band", "credit_score", "debt_to_income", "employment_years"]
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameters
    n_estimators = 50
    max_depth = 5
    
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        # Log params
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            registered_model_name=model_name
        )
        
        print(f"Model trained successfully. Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(f"Registered model as: {model_name} (Run ID: {run.info.run_id})")
        return run.info.run_id
