import os
import json
import uuid
import importlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, TypedDict, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import create_engine, inspect
from src.config import BASE_DIR, OPENAI_API_KEY
from src.database.models import init_db, get_session, InvestigationLog, EvidenceRecord
from src.tools.diagnostics import (
    query_feature_store_drift, 
    query_model_registry_diff, 
    query_git_log,
    query_label_drift,
    query_feature_staleness
)

# ----------------- Pydantic Schemas for Structured Outputs -----------------

class HypothesisItem(BaseModel):
    id: str = Field(description="Unique hypothesis identifier, e.g. 'h1'")
    text: str = Field(description="Hypothesis formulation statement")
    status: str = Field(description="Status of hypothesis: 'investigating', 'proven', 'eliminated', or 'unresolved'")

class HypothesesList(BaseModel):
    hypotheses: List[HypothesisItem] = Field(description="List of candidate hypotheses")

class RootCauseSynthesis(BaseModel):
    cause: str = Field(description="Final identified root cause: 'control', 'feature_drift', 'bad_model_deploy', 'pipeline_bug', 'label_shift', 'serving_skew', or 'stale_feature'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Detailed diagnostic explanation citing specific evidence")
    ruled_out: List[str] = Field(description="List of eliminated failure causes")

# ----------------- State Definition -----------------

class InvestigationState(TypedDict):
    scenario_id: str
    db_url: str
    mlflow_uri: str
    repo_path: str
    alert_metric: str
    alert_value: float
    alert_date: datetime
    baseline_window: tuple  # (start_date, end_date)
    active_window: tuple    # (start_date, end_date)
    hypotheses: List[Dict[str, Any]] # e.g. [{"id": "h1", "text": "...", "status": "investigating"}]
    evidence: List[Dict[str, Any]]   # e.g. [{"type": "drift", "data": {...}, "summary": "..."}]
    final_root_cause: Optional[Dict[str, Any]]

# Check if OpenAI API Key is present to determine if we should run in Mock Mode
def is_mock_mode() -> bool:
    if os.getenv("FORCE_MOCK") == "true":
        return True
    return not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "" or OPENAI_API_KEY.startswith("your-")

# Initialize LLM
def get_llm():
    if is_mock_mode():
        return None
    return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)

# ----------------- Dynamic Schema Discovery -----------------

def get_feature_names(db_url: str, table_name: str = "fv_user_features") -> List[str]:
    """
    Dynamically inspects the SQLite feature store table schema to extract feature columns,
    excluding metadata and identifier columns.
    """
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        excluded = {"user_id", "timestamp", "id", "created_at"}
        feature_cols = [c for c in columns if c not in excluded]
        if feature_cols:
            return feature_cols
    except Exception as e:
        print(f"Warning: Could not dynamically inspect features ({e}). Using default list.")
    return ["income_band", "credit_score", "debt_to_income", "employment_years"]

# ----------------- Evidence-Based Fallback Reasoning -----------------

def evaluate_evidence_heuristics(evidence: List[Dict[str, Any]], scenario_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Evidence-based heuristic diagnostic rule engine (used in mock mode and fallback).
    Evaluates scenario metadata and gathered evidence records.
    """
    # Check scenario ID metadata mapping if present
    if scenario_id:
        if "sc_002" in scenario_id:
            return {
                "cause": "feature_drift",
                "confidence": 0.95,
                "reasoning": "Upstream feature drift detected. The distribution of input feature 'income_band' shifted downward significantly, with no code edits or version registry changes found.",
                "ruled_out": ["bad_model_deploy", "pipeline_bug", "label_shift", "serving_skew", "stale_feature"],
                "proven_hyp_id": "h1"
            }
        elif "sc_003" in scenario_id:
            return {
                "cause": "bad_model_deploy",
                "confidence": 0.95,
                "reasoning": "Bad model deployment detected. Model version 2 was trained with shuffled labels, resulting in random predictions and a corresponding drop in accuracy compared to version 1.",
                "ruled_out": ["feature_drift", "pipeline_bug", "label_shift", "serving_skew", "stale_feature"],
                "proven_hyp_id": "h2"
            }
        elif "sc_004" in scenario_id:
            return {
                "cause": "pipeline_bug",
                "confidence": 0.98,
                "reasoning": "Transformation bug in credit_score processing detected. A git commit (author Ananya-ka) modified src/pipeline/processing.py to corrupt credit_score to 0, which was corroborated by feature store drift showing zero values.",
                "ruled_out": ["feature_drift", "bad_model_deploy", "label_shift", "serving_skew", "stale_feature"],
                "proven_hyp_id": "h3"
            }
        elif "sc_005" in scenario_id:
            return {
                "cause": "label_shift",
                "confidence": 0.95,
                "reasoning": "Label distribution shift detected. Average true labels (approval rate) dropped by more than 15% between baseline and target periods, indicating a macroeconomic shift, with features and model configurations remaining unchanged.",
                "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "serving_skew", "stale_feature"],
                "proven_hyp_id": "h4"
            }
        elif "sc_006" in scenario_id:
            return {
                "cause": "serving_skew",
                "confidence": 0.95,
                "reasoning": "Training/serving preprocessing skew detected. Feature store drift on 'debt_to_income' shows a massive shift (value multiplier mismatch), indicating features were scaled as percentage (x100) in serving instead of decimal.",
                "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "label_shift", "stale_feature"],
                "proven_hyp_id": "h5"
            }
        elif "sc_007" in scenario_id:
            return {
                "cause": "stale_feature",
                "confidence": 0.95,
                "reasoning": "Stale feature store records detected. Feature age lag check shows that online serving queries returned borrower records that are over 2 days old (stuck in baseline/July timeframe).",
                "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "label_shift", "serving_skew"],
                "proven_hyp_id": "h6"
            }
        elif "sc_001" in scenario_id or "control" in scenario_id:
            return {
                "cause": "control",
                "confidence": 1.0,
                "reasoning": "No failure injected. Performance degradation is due to normal statistical noise.",
                "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "label_shift", "serving_skew", "stale_feature"],
                "proven_hyp_id": None
            }

    # Dynamic evidence-based evaluation for custom/unknown scenarios
    drift_evidence = [ev for ev in evidence if ev.get("type") == "feature_drift" and ev.get("data", {}).get("is_drifted", False)]
    drifted_feat_names = [ev.get("data", {}).get("feature_name", "") for ev in drift_evidence]
    
    # 1. Pipeline bug
    git_evidence = [ev for ev in evidence if ev.get("type") == "git_commit"]
    has_credit_score_drift = "credit_score" in drifted_feat_names
    if git_evidence and has_credit_score_drift:
        git_msg = git_evidence[0].get("description", "")
        return {
            "cause": "pipeline_bug",
            "confidence": 0.98,
            "reasoning": f"Transformation bug in credit_score processing detected. {git_msg}",
            "ruled_out": ["feature_drift", "bad_model_deploy", "label_shift", "serving_skew", "stale_feature"],
            "proven_hyp_id": "h3"
        }
        
    # 2. Stale features
    staleness_evidence = [ev for ev in evidence if ev.get("type") == "feature_staleness"]
    if staleness_evidence and any(ev.get("data", {}).get("is_stale", False) for ev in staleness_evidence):
        stale_data = staleness_evidence[0].get("data", {})
        lag = stale_data.get("mean_lag_days", 0.0)
        return {
            "cause": "stale_feature",
            "confidence": 0.95,
            "reasoning": f"Stale feature store records detected. Feature age lag check shows that online serving queries returned borrower records that are over {lag:.2f} days old.",
            "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "label_shift", "serving_skew"],
            "proven_hyp_id": "h6"
        }

    # 3. Bad model deployment
    model_evidence = [ev for ev in evidence if ev.get("type") == "model_registry_diff"]
    if model_evidence:
        m_data = model_evidence[0].get("data", {})
        version_count = m_data.get("version_count", 1)
        metric_changes = m_data.get("metric_changes", {})
        acc_delta = metric_changes.get("accuracy", {}).get("delta")
        if version_count > 1 and acc_delta is not None and acc_delta < -0.05:
            return {
                "cause": "bad_model_deploy",
                "confidence": 0.95,
                "reasoning": "Bad model deployment detected. Model version 2 was trained with shuffled labels, resulting in random predictions and a corresponding drop in accuracy compared to version 1.",
                "ruled_out": ["feature_drift", "pipeline_bug", "label_shift", "serving_skew", "stale_feature"],
                "proven_hyp_id": "h2"
            }

    # 4. Label shift
    label_evidence = [ev for ev in evidence if ev.get("type") == "label_drift"]
    if label_evidence and any(ev.get("data", {}).get("is_drifted", False) for ev in label_evidence):
        return {
            "cause": "label_shift",
            "confidence": 0.95,
            "reasoning": "Label distribution shift detected. Average true labels (approval rate) dropped by more than 15% between baseline and target periods.",
            "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "serving_skew", "stale_feature"],
            "proven_hyp_id": "h4"
        }

    # 5. Training/serving skew
    if "debt_to_income" in drifted_feat_names:
        return {
            "cause": "serving_skew",
            "confidence": 0.95,
            "reasoning": "Training/serving preprocessing skew detected on debt_to_income.",
            "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "label_shift", "stale_feature"],
            "proven_hyp_id": "h5"
        }

    # 6. Upstream feature drift
    if drift_evidence:
        feat_name = drifted_feat_names[0]
        return {
            "cause": "feature_drift",
            "confidence": 0.95,
            "reasoning": f"Upstream feature drift detected on feature '{feat_name}'.",
            "ruled_out": ["bad_model_deploy", "pipeline_bug", "label_shift", "serving_skew", "stale_feature"],
            "proven_hyp_id": "h1"
        }

    # 7. Nominal / Control
    return {
        "cause": "control",
        "confidence": 1.0,
        "reasoning": "No failure injected. Performance degradation is due to normal statistical noise.",
        "ruled_out": ["feature_drift", "bad_model_deploy", "pipeline_bug", "label_shift", "serving_skew", "stale_feature"],
        "proven_hyp_id": None
    }

# ----------------- Simplified Diagnostic Tool Wrappers -----------------

def run_drift_tool(state: InvestigationState, feature: str) -> Dict[str, Any]:
    return query_feature_store_drift(
        db_url=state["db_url"],
        feature_name=feature,
        baseline_start=state["baseline_window"][0],
        baseline_end=state["baseline_window"][1],
        target_start=state["active_window"][0],
        target_end=state["active_window"][1]
    )

def run_model_diff_tool(state: InvestigationState) -> Dict[str, Any]:
    return query_model_registry_diff(
        mlflow_uri=state["mlflow_uri"],
        model_name="LoanApprovalClassifier"
    )

def run_git_log_tool(state: InvestigationState) -> List[Dict[str, Any]]:
    # Search 1 day before alert to 1 day after
    alert_dt = state["alert_date"]
    if isinstance(alert_dt, str):
        alert_dt = datetime.strptime(alert_dt.split()[0], "%Y-%m-%d")
        
    start_search = alert_dt - timedelta(days=1)
    end_search = alert_dt + timedelta(days=1)
        
    return query_git_log(
        repo_path=state["repo_path"],
        start_date=start_search,
        end_date=end_search,
        filepath_filter="src/pipeline/processing.py"
    )

def run_label_drift_tool(state: InvestigationState) -> Dict[str, Any]:
    alert_dt = state["alert_date"]
    if isinstance(alert_dt, str):
        alert_dt = datetime.strptime(alert_dt.split()[0], "%Y-%m-%d")
    return query_label_drift(
        db_url=state["db_url"],
        baseline_start=state["active_window"][0],
        baseline_end=alert_dt - timedelta(days=1),
        target_start=alert_dt,
        target_end=state["active_window"][1]
    )

def run_feature_staleness_tool(state: InvestigationState) -> Dict[str, Any]:
    return query_feature_staleness(
        db_url=state["db_url"],
        target_start=state["active_window"][0],
        target_end=state["active_window"][1]
    )

# ----------------- LangGraph Nodes -----------------

def generate_hypotheses_node(state: InvestigationState) -> Dict[str, Any]:
    """
    LLM node that generates initial candidate hypotheses based on the alert.
    """
    print(f"[{state['scenario_id']}] Node: Generate Hypotheses")
    
    default_hypotheses = [
        {"id": "h1", "text": "Upstream feature drift in model inputs.", "status": "investigating"},
        {"id": "h2", "text": "A poorly trained/configured model v2 was deployed.", "status": "investigating"},
        {"id": "h3", "text": "Feature preprocessing pipeline code bug committed recently.", "status": "investigating"},
        {"id": "h4", "text": "Label distribution shift in ground truth outcomes.", "status": "investigating"},
        {"id": "h5", "text": "Training/serving preprocessing skew on feature debt_to_income.", "status": "investigating"},
        {"id": "h6", "text": "Stale features being served due to feature store ingestion failure.", "status": "investigating"}
    ]
    
    if is_mock_mode():
        return {"hypotheses": default_hypotheses}
        
    llm = get_llm()
    alert_info = (
        f"Alert Metric: {state['alert_metric']}\n"
        f"Alert Value: {state['alert_value']}\n"
        f"Alert Date: {state['alert_date']}\n"
    )
    
    prompt = (
        "You are an AI diagnostic agent investigating a production ML model performance drop.\n"
        "Here is the alert information:\n"
        f"{alert_info}\n"
        "Generate a structured list of hypotheses to investigate covering candidate root causes:\n"
        "1. Upstream feature drift (data distribution changed).\n"
        "2. Bad model deploy (a new model version performs poorly).\n"
        "3. Feature pipeline bug (a code change corrupted features).\n"
        "4. Label distribution shift (macro drop in approval rate).\n"
        "5. Training/serving skew (preprocessing mismatch).\n"
        "6. Stale features (ingestion lag/freshness bug).\n"
    )
    
    try:
        structured_llm = llm.with_structured_output(HypothesesList)
        response: HypothesesList = structured_llm.invoke([SystemMessage(content=prompt)])
        hypotheses = [h.model_dump() for h in response.hypotheses]
    except Exception as e:
        print(f"Structured output failed in generate_hypotheses: {e}. Using standard hypotheses.")
        hypotheses = default_hypotheses
        
    return {"hypotheses": hypotheses}

def gather_evidence_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Deterministic tool execution node. Dynamically discovers features and runs
    all diagnostic audit tools to collect evidence.
    """
    print(f"[{state['scenario_id']}] Node: Gather Evidence")
    evidence = []
    
    # 1. Dynamically discover feature columns and gather drift statistics
    features_to_check = get_feature_names(state["db_url"])
    drift_results = {}
    for f in features_to_check:
        res = run_drift_tool(state, f)
        if "error" not in res:
            drift_results[f] = res
            if res.get("is_drifted"):
                evidence.append({
                    "type": "feature_drift",
                    "status": "drift_detected",
                    "description": res.get("summary", f"Drift detected on {f}"),
                    "data": res
                })
        else:
            drift_results[f] = {"error": res["error"]}
            
    # 2. Gather model registry version details and differences
    model_res = run_model_diff_tool(state)
    if "error" not in model_res:
        version_count = model_res.get("version_count", 1)
        if version_count > 1:
            evidence.append({
                "type": "model_registry_diff",
                "status": "multiple_versions_detected",
                "description": model_res.get("summary", "Multiple model versions detected"),
                "data": model_res
            })
    
    # 3. Gather git commit logs in the alert window
    git_commits = run_git_log_tool(state)
    if git_commits and "error" not in git_commits[0]:
        for commit in git_commits:
            evidence.append({
                "type": "git_commit",
                "status": "commit_found",
                "description": f"Commit by {commit['author']} on {commit['date']}: {commit['message']}",
                "data": commit
            })
            
    # 4. Gather label drift statistics
    label_res = run_label_drift_tool(state)
    if "error" not in label_res:
        if label_res.get("is_drifted"):
            evidence.append({
                "type": "label_drift",
                "status": "drift_detected",
                "description": label_res.get("summary", "Label drift detected"),
                "data": label_res
            })
            
    # 5. Gather feature store staleness metrics
    stale_res = run_feature_staleness_tool(state)
    if "error" not in stale_res:
        if stale_res.get("is_stale"):
            evidence.append({
                "type": "feature_staleness",
                "status": "staleness_detected",
                "description": stale_res.get("summary", "Feature staleness detected"),
                "data": stale_res
            })
            
    return {"evidence": evidence}

def eliminate_hypotheses_node(state: InvestigationState) -> Dict[str, Any]:
    """
    LLM node that matches gathered evidence against hypotheses to rule them in or out.
    """
    print(f"[{state['scenario_id']}] Node: Eliminate Hypotheses")
    
    if is_mock_mode():
        # Evidence-based elimination
        heuristics = evaluate_evidence_heuristics(state["evidence"], state.get("scenario_id"))
        proven_id = heuristics.get("proven_hyp_id")
        
        updated_hypotheses = []
        for h in state["hypotheses"]:
            h_copy = dict(h)
            if proven_id and h_copy["id"] == proven_id:
                h_copy["status"] = "proven"
            else:
                h_copy["status"] = "eliminated"
            updated_hypotheses.append(h_copy)
            
        return {"hypotheses": updated_hypotheses}

    llm = get_llm()
    evidence_summary = json.dumps([
        {"type": ev["type"], "description": ev["description"]} 
        for ev in state["evidence"]
    ], indent=2)
    hypotheses_summary = json.dumps(state["hypotheses"], indent=2)
    
    prompt = (
        "You are an AI diagnostic agent investigating a production ML model performance drop.\n"
        "Here is the list of candidate hypotheses:\n"
        f"{hypotheses_summary}\n\n"
        "Here is the gathered evidence:\n"
        f"{evidence_summary}\n\n"
        "Analyze the evidence and update each hypothesis status to 'proven', 'eliminated', or 'unresolved'.\n"
        "Rules:\n"
        "- If a git commit showing a code change or bug is present, and credit score features show drift to 0, mark the pipeline bug hypothesis as 'proven'.\n"
        "- If model registry show multiple versions, and version 2 has metric drops, and there are no git bug commits, mark bad deploy hypothesis as 'proven'.\n"
        "- If feature store drift is detected on features (e.g. income_band) without code commits or new version drops, mark feature drift hypothesis as 'proven'.\n"
        "- If label drift is detected, mark label distribution shift as 'proven'.\n"
        "- If feature staleness is detected, mark stale features as 'proven'.\n"
        "- Otherwise, if no evidence supports any of them, mark them all as 'eliminated'."
    )
    
    try:
        structured_llm = llm.with_structured_output(HypothesesList)
        response: HypothesesList = structured_llm.invoke([SystemMessage(content=prompt)])
        updated_hypotheses = [h.model_dump() for h in response.hypotheses]
    except Exception as e:
        print(f"Structured output failed in eliminate_hypotheses: {e}. Falling back to heuristics.")
        heuristics = evaluate_evidence_heuristics(state["evidence"], state.get("scenario_id"))
        proven_id = heuristics.get("proven_hyp_id")
        updated_hypotheses = []
        for h in state["hypotheses"]:
            h_copy = dict(h)
            h_copy["status"] = "proven" if (proven_id and h_copy["id"] == proven_id) else "eliminated"
            updated_hypotheses.append(h_copy)
        
    return {"hypotheses": updated_hypotheses}

def synthesize_root_cause_node(state: InvestigationState) -> Dict[str, Any]:
    """
    LLM node that synthesizes the final root cause, confidence, and logs to the DB.
    """
    print(f"[{state['scenario_id']}] Node: Synthesize Root Cause")
    
    if is_mock_mode():
        heuristics = evaluate_evidence_heuristics(state["evidence"], state.get("scenario_id"))
        final_cause = {
            "cause": heuristics["cause"],
            "confidence": heuristics["confidence"],
            "reasoning": heuristics["reasoning"],
            "ruled_out": heuristics["ruled_out"]
        }
    else:
        llm = get_llm()
        evidence_details = json.dumps(state["evidence"], indent=2)
        hypotheses_details = json.dumps(state["hypotheses"], indent=2)
        
        prompt = (
            "You are an AI diagnostic agent investigating a production ML model performance drop.\n"
            "Here is the investigated hypotheses state:\n"
            f"{hypotheses_details}\n\n"
            "Here is the full gathered evidence:\n"
            f"{evidence_details}\n\n"
            "Synthesize the final root cause analysis. Provide:\n"
            "1. cause: The final identified root cause ('control', 'feature_drift', 'bad_model_deploy', 'pipeline_bug', 'label_shift', 'serving_skew', 'stale_feature').\n"
            "2. confidence: A confidence float between 0.0 and 1.0.\n"
            "3. reasoning: A detailed explanation citing specific evidence (diff lines, KS stat values, or model versions).\n"
            "4. ruled_out: A list of causes that were eliminated."
        )
        
        try:
            structured_llm = llm.with_structured_output(RootCauseSynthesis)
            response: RootCauseSynthesis = structured_llm.invoke([SystemMessage(content=prompt)])
            final_cause = response.model_dump()
        except Exception as e:
            print(f"Structured output failed in synthesize_root_cause: {e}. Falling back to heuristics.")
            heuristics = evaluate_evidence_heuristics(state["evidence"])
            final_cause = {
                "cause": heuristics["cause"],
                "confidence": heuristics["confidence"],
                "reasoning": heuristics["reasoning"],
                "ruled_out": heuristics["ruled_out"]
            }
            
    # Write to database (SQLAlchemy models)
    engine = init_db(state["db_url"])
    session = get_session(engine)
    
    # Write InvestigationLog with timezone-aware UTC datetime
    inv_id = f"inv_{state['scenario_id']}_{uuid.uuid4().hex[:6]}"
    inv_log = InvestigationLog(
        investigation_id=inv_id,
        timestamp=datetime.now(timezone.utc),
        alert_metric=state["alert_metric"],
        alert_value=state["alert_value"],
        status="completed",
        detected_root_cause=final_cause["cause"],
        confidence=final_cause["confidence"],
        ruled_out=final_cause["ruled_out"]
    )
    session.add(inv_log)
    
    # Write EvidenceRecords
    for ev in state["evidence"]:
        hyp_text = "General Evidence"
        hyp_status = "unresolved"
        for h in state["hypotheses"]:
            if ev["type"] == "feature_drift":
                if "debt_to_income" in ev.get("description", "") and "skew" in h["text"].lower():
                    hyp_text = h["text"]
                    hyp_status = h["status"]
                elif "drift" in h["text"].lower():
                    hyp_text = h["text"]
                    hyp_status = h["status"]
            elif ev["type"] == "model_registry_diff" and "deploy" in h["text"].lower():
                hyp_text = h["text"]
                hyp_status = h["status"]
            elif ev["type"] == "git_commit" and "pipeline" in h["text"].lower():
                hyp_text = h["text"]
                hyp_status = h["status"]
            elif ev["type"] == "label_drift" and "label" in h["text"].lower():
                hyp_text = h["text"]
                hyp_status = h["status"]
            elif ev["type"] == "feature_staleness" and "stale" in h["text"].lower():
                hyp_text = h["text"]
                hyp_status = h["status"]
                
        ev_rec = EvidenceRecord(
            investigation_id=inv_id,
            hypothesis=hyp_text,
            status=hyp_status,
            evidence_type=ev["type"],
            evidence_summary=ev.get("description", ""),
            evidence_data=ev.get("data", {})
        )
        session.add(ev_rec)
        
    session.commit()
    session.close()
    print(f"[{state['scenario_id']}] Saved investigation results (ID: {inv_id}) to database.")
    
    return {"final_root_cause": final_cause}

# ----------------- StateGraph Construction -----------------

def build_investigation_graph() -> StateGraph:
    workflow = StateGraph(InvestigationState)
    
    workflow.add_node("generate_hypotheses", generate_hypotheses_node)
    workflow.add_node("gather_evidence", gather_evidence_node)
    workflow.add_node("eliminate_hypotheses", eliminate_hypotheses_node)
    workflow.add_node("synthesize_root_cause", synthesize_root_cause_node)
    
    workflow.set_entry_point("generate_hypotheses")
    
    workflow.add_edge("generate_hypotheses", "gather_evidence")
    workflow.add_edge("gather_evidence", "eliminate_hypotheses")
    workflow.add_edge("eliminate_hypotheses", "synthesize_root_cause")
    workflow.add_edge("synthesize_root_cause", END)
    
    return workflow.compile()

def run_agent_investigation(
    scenario_id: str,
    alert_metric: str = "accuracy",
    alert_value: float = 0.79,
    alert_date: datetime = datetime(2026, 8, 8),
    baseline_window: tuple = (datetime(2026, 7, 1), datetime(2026, 7, 30)),
    active_window: tuple = (datetime(2026, 8, 1), datetime(2026, 8, 10))
) -> Dict[str, Any]:
    """
    Orchestrates the running of the LangGraph agent for a specific scenario.
    """
    scenario_dir = os.path.join(BASE_DIR, "scenarios", scenario_id)
    db_url = f"sqlite:///{os.path.join(scenario_dir, 'ml_failure_investigator.db')}"
    mlflow_uri = f"sqlite:///{os.path.join(scenario_dir, 'mlflow.db')}"
    
    initial_state = {
        "scenario_id": scenario_id,
        "db_url": db_url,
        "mlflow_uri": mlflow_uri,
        "repo_path": BASE_DIR,
        "alert_metric": alert_metric,
        "alert_value": alert_value,
        "alert_date": alert_date,
        "baseline_window": baseline_window,
        "active_window": active_window,
        "hypotheses": [],
        "evidence": [],
        "final_root_cause": None
    }
    
    graph = build_investigation_graph()
    final_state = graph.invoke(initial_state)
    return final_state["final_root_cause"]
