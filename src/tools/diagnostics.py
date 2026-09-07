import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy import create_engine
import mlflow
import git
from typing import Dict, List, Any, Optional

def query_feature_store_drift(
    db_url: str,
    feature_name: str,
    baseline_start: datetime,
    baseline_end: datetime,
    target_start: datetime,
    target_end: datetime,
    feature_view_name: str = "user_features"
) -> Dict[str, Any]:
    """
    Queries the feature store history and performs a Kolmogorov-Smirnov test 
    to detect distribution drift for a specific feature between baseline and target periods.
    """
    engine = create_engine(db_url)
    
    # Read the features table
    table_name = f"fv_{feature_view_name}"
    try:
        df = pd.read_sql(f"SELECT timestamp, {feature_name} FROM {table_name}", con=engine)
    except Exception as e:
        return {
            "error": f"Failed to query database: {e}",
            "is_drifted": False
        }
        
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Extract baseline and target datasets
    baseline_df = df[(df["timestamp"] >= pd.to_datetime(baseline_start)) & (df["timestamp"] <= pd.to_datetime(baseline_end))]
    target_df = df[(df["timestamp"] >= pd.to_datetime(target_start)) & (df["timestamp"] <= pd.to_datetime(target_end))]
    
    baseline_vals = baseline_df[feature_name].dropna().values
    target_vals = target_df[feature_name].dropna().values
    
    if len(baseline_vals) < 10 or len(target_vals) < 10:
        return {
            "error": f"Insufficient data for drift test. Baseline count: {len(baseline_vals)}, Target count: {len(target_vals)}",
            "is_drifted": False,
            "baseline_count": len(baseline_vals),
            "target_count": len(target_vals)
        }
        
    # Perform Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(baseline_vals, target_vals)
    
    baseline_mean = float(np.mean(baseline_vals))
    target_mean = float(np.mean(target_vals))
    
    # A p-value less than 0.05 rejects the null hypothesis (meaning distributions differ)
    is_drifted = bool(p_value < 0.05)
    
    return {
        "feature_name": feature_name,
        "baseline_count": len(baseline_vals),
        "target_count": len(target_vals),
        "baseline_mean": baseline_mean,
        "target_mean": target_mean,
        "baseline_std": float(np.std(baseline_vals)),
        "target_std": float(np.std(target_vals)),
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "is_drifted": is_drifted,
        "summary": f"Feature '{feature_name}' mean shifted from {baseline_mean:.4f} to {target_mean:.4f} (KS p-value: {p_value:.4e})."
    }

def query_model_registry_diff(
    mlflow_uri: str,
    model_name: str = "LoanApprovalClassifier"
) -> Dict[str, Any]:
    """
    Connects to MLflow, retrieves registered model versions, and outputs a comparison
    of parameters and metrics between the latest version and the previous version.
    """
    # Initialize MLflow client
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.client.MlflowClient()
    
    try:
        # Search all model versions for this name
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        return {
            "error": f"Failed to search model registry: {e}",
            "model_name": model_name
        }
        
    if not versions:
        return {
            "error": f"No model versions found registered under name '{model_name}'.",
            "model_name": model_name
        }
        
    # Sort versions by version number ascending
    versions_sorted = sorted(versions, key=lambda v: int(v.version))
    
    latest_version = versions_sorted[-1]
    
    if len(versions_sorted) == 1:
        # Only one version exists
        run = client.get_run(latest_version.run_id)
        return {
            "model_name": model_name,
            "version_count": 1,
            "latest_version": str(latest_version.version),
            "latest_version_run_id": latest_version.run_id,
            "latest_version_params": run.data.params,
            "latest_version_metrics": run.data.metrics,
            "summary": f"Only model version {latest_version.version} is registered. Parameters: {run.data.params}. Metrics: {run.data.metrics}."
        }
        
    # Compare latest version with previous version
    prev_version = versions_sorted[-2]
    
    try:
        run_latest = client.get_run(latest_version.run_id)
        run_prev = client.get_run(prev_version.run_id)
    except Exception as e:
        return {
            "error": f"Failed to retrieve runs from MLflow: {e}",
            "latest_version": latest_version.version,
            "previous_version": prev_version.version
        }
        
    params_latest = run_latest.data.params
    metrics_latest = run_latest.data.metrics
    
    params_prev = run_prev.data.params
    metrics_prev = run_prev.data.metrics
    
    # Calculate parameter differences
    param_changes = {}
    all_param_keys = set(params_latest.keys()) | set(params_prev.keys())
    for k in all_param_keys:
        val_latest = params_latest.get(k)
        val_prev = params_prev.get(k)
        if val_latest != val_prev:
            param_changes[k] = {"previous": val_prev, "latest": val_latest}
            
    # Calculate metric differences (deltas)
    metric_changes = {}
    all_metric_keys = set(metrics_latest.keys()) | set(metrics_prev.keys())
    for k in all_metric_keys:
        val_latest = metrics_latest.get(k)
        val_prev = metrics_prev.get(k)
        if val_latest is not None and val_prev is not None:
            metric_changes[k] = {
                "previous": float(val_prev),
                "latest": float(val_latest),
                "delta": float(val_latest - val_prev)
            }
        else:
            metric_changes[k] = {"previous": val_prev, "latest": val_latest, "delta": None}
            
    return {
        "model_name": model_name,
        "version_count": len(versions_sorted),
        "latest_version": str(latest_version.version),
        "previous_version": str(prev_version.version),
        "param_changes": param_changes,
        "metric_changes": metric_changes,
        "summary": f"Compared version {latest_version.version} vs version {prev_version.version}. Param changes: {list(param_changes.keys())}. Metric deltas: {{k: v['delta'] for k, v in metric_changes.items() if v['delta'] is not None}}."
    }

def query_git_log(
    repo_path: str,
    start_date: datetime,
    end_date: datetime,
    filepath_filter: Optional[str] = "src/pipeline/processing.py"
) -> List[Dict[str, Any]]:
    """
    Queries git history for commits and retrieves unified code diffs of changed files
    in the specified date range.
    """
    try:
        repo = git.Repo(repo_path)
    except Exception as e:
        return [{"error": f"Failed to initialize git repository: {e}"}]
        
    commits_in_range = []
    
    # Iterate through active branch commits
    # Note: gitpython committed_date is seconds since epoch (UTC)
    for commit in repo.iter_commits():
        commit_time = datetime.fromtimestamp(commit.committed_date)
        if start_date <= commit_time <= end_date:
            # Check if commit contains changes to filter filepath
            if filepath_filter:
                files_changed = list(commit.stats.files.keys())
                # Match path partially or fully
                if not any(filepath_filter in f for f in files_changed):
                    continue
            commits_in_range.append(commit)
            
    results = []
    for commit in commits_in_range:
        commit_time = datetime.fromtimestamp(commit.committed_date)
        
        # Get parent commit to generate diff
        parent = commit.parents[0] if commit.parents else git.NULL_TREE
        diff_index = parent.diff(commit, create_patch=True)
        
        diffs = []
        for diff in diff_index:
            file_diff = {
                "file_path": diff.b_path,
                "deleted_file": diff.deleted_file,
                "new_file": diff.new_file,
                "diff_text": diff.diff.decode('utf-8', errors='ignore') if diff.diff else ""
            }
            diffs.append(file_diff)
            
        results.append({
            "commit_hash": commit.hexsha,
            "author": commit.author.name,
            "date": commit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": commit.message.strip(),
            "changed_files": list(commit.stats.files.keys()),
            "diffs": diffs
        })
        
    return results

def query_label_drift(
    db_url: str,
    baseline_start: datetime,
    baseline_end: datetime,
    target_start: datetime,
    target_end: datetime
) -> Dict[str, Any]:
    """
    Compares the mean of the true labels ('true_label') in the serving logs 
    between baseline and target periods to detect label distribution shift.
    """
    engine = create_engine(db_url)
    try:
        df = pd.read_sql("SELECT timestamp, true_label FROM serving_logs WHERE true_label IS NOT NULL", con=engine)
    except Exception as e:
        return {
            "error": f"Failed to query serving logs: {e}",
            "is_drifted": False
        }
        
    if df.empty:
        return {
            "error": "No serving logs with labels found.",
            "is_drifted": False
        }
        
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Baseline vs Target slices
    baseline_df = df[(df["timestamp"] >= pd.to_datetime(baseline_start)) & (df["timestamp"] <= pd.to_datetime(baseline_end))]
    target_df = df[(df["timestamp"] >= pd.to_datetime(target_start)) & (df["timestamp"] <= pd.to_datetime(target_end))]
    
    baseline_labels = baseline_df["true_label"].dropna().values
    target_labels = target_df["true_label"].dropna().values
    
    if len(baseline_labels) < 10 or len(target_labels) < 10:
        return {
            "error": f"Insufficient labels. Baseline count: {len(baseline_labels)}, Target count: {len(target_labels)}",
            "is_drifted": False,
            "baseline_count": len(baseline_labels),
            "target_count": len(target_labels)
        }
        
    baseline_mean = float(np.mean(baseline_labels))
    target_mean = float(np.mean(target_labels))
    
    # Flag drift if the target approval rate drops by more than 15% (e.g. from 0.50 to <0.35)
    drift_delta = baseline_mean - target_mean
    is_drifted = bool(drift_delta >= 0.15)
    
    return {
        "baseline_count": len(baseline_labels),
        "target_count": len(target_labels),
        "baseline_mean": baseline_mean,
        "target_mean": target_mean,
        "drift_delta": drift_delta,
        "is_drifted": is_drifted,
        "summary": f"Target approval rate shifted from {baseline_mean:.4f} to {target_mean:.4f} (Delta: {drift_delta:.4f}, Threshold: 0.1500)."
    }

def query_feature_staleness(
    db_url: str,
    target_start: datetime,
    target_end: datetime,
    feature_view_name: str = "user_features"
) -> Dict[str, Any]:
    """
    Compares the serving prediction event timestamps in 'serving_logs' against the 
    ingestion timestamps in the feature store table to detect freshness (staleness) bugs.
    """
    engine = create_engine(db_url)
    try:
        logs_df = pd.read_sql(
            f"SELECT user_id, timestamp as event_time FROM serving_logs WHERE timestamp BETWEEN '{target_start.strftime('%Y-%m-%d %H:%M:%S')}' AND '{target_end.strftime('%Y-%m-%d %H:%M:%S')}'", 
            con=engine
        )
        features_df = pd.read_sql(
            f"SELECT user_id, timestamp as feature_time FROM fv_{feature_view_name}", 
            con=engine
        )
    except Exception as e:
        return {
            "error": f"Failed to query database for staleness: {e}",
            "is_stale": False
        }
        
    if logs_df.empty or features_df.empty:
        return {
            "error": f"Insufficient records to check staleness. Logs count: {len(logs_df)}, Features count: {len(features_df)}",
            "is_stale": False
        }
        
    logs_df["event_time"] = pd.to_datetime(logs_df["event_time"])
    features_df["feature_time"] = pd.to_datetime(features_df["feature_time"])
    
    features_df = features_df.sort_values("feature_time")
    
    merged = pd.merge_asof(
        logs_df.sort_values("event_time"),
        features_df.sort_values("feature_time"),
        left_on="event_time",
        right_on="feature_time",
        by="user_id",
        direction="backward"
    )
    
    merged["time_delta_days"] = (merged["event_time"] - merged["feature_time"]).dt.total_seconds() / 86400.0
    valid_deltas = merged["time_delta_days"].dropna().values
    
    if len(valid_deltas) < 10:
        return {
            "error": f"Insufficient matched user records for staleness. Matched: {len(valid_deltas)}",
            "is_stale": False
        }
        
    mean_lag_days = float(np.mean(valid_deltas))
    max_lag_days = float(np.max(valid_deltas))
    is_stale = bool(mean_lag_days > 0.5)
    
    return {
        "matched_count": len(valid_deltas),
        "mean_lag_days": mean_lag_days,
        "max_lag_days": max_lag_days,
        "is_stale": is_stale,
        "summary": f"Feature store staleness check: average feature age lag is {mean_lag_days:.2f} days (Max: {max_lag_days:.2f} days, Threshold: 0.5 days)."
    }
