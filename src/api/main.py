import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import BASE_DIR
from src.harness.scenario_generator import load_metadata, execute_scenario, SCENARIOS_DIR
from src.agent.agent import run_agent_investigation
from src.harness.eval_harness import evaluate_scenario, print_summary_table
from src.database.models import init_db, get_session, InvestigationLog, EvidenceRecord

app = FastAPI(title="ML failure Investigator API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    limit: Optional[int] = None

@app.get("/api/scenarios")
def get_scenarios():
    """
    Returns all scenarios from metadata and checks if their simulation directories exist.
    """
    try:
        metadata = load_metadata()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scenario metadata: {e}")
        
    results = []
    for sc in metadata:
        sc_id = sc["scenario_id"]
        sc_dir = os.path.join(SCENARIOS_DIR, sc_id)
        db_file = os.path.join(sc_dir, "ml_failure_investigator.db")
        is_materialized = os.path.exists(db_file)
        results.append({
            **sc,
            "is_materialized": is_materialized
        })
    return results

@app.post("/api/scenarios/{scenario_id}/simulate")
def simulate_scenario(scenario_id: str):
    """
    Triggers failure harness simulation for the given scenario ID.
    """
    try:
        # Run simulation
        execute_scenario(scenario_id)
        return {"status": "success", "message": f"Scenario {scenario_id} materialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed for {scenario_id}: {e}")

@app.post("/api/scenarios/{scenario_id}/investigate")
def investigate_scenario(scenario_id: str):
    """
    Triggers LangGraph investigator agent on the scenario.
    """
    metadata = load_metadata()
    scenario = next((s for s in metadata if s["scenario_id"] == scenario_id), None)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found in metadata.")
        
    try:
        alert_date = datetime.strptime(scenario["injection_date"], "%Y-%m-%d")
        alert_metric = "accuracy"
        alert_value = 0.70 if scenario["injected_failure"] != "control" else 0.79
        
        # Run agent
        agent_output = run_agent_investigation(
            scenario_id=scenario_id,
            alert_metric=alert_metric,
            alert_value=alert_value,
            alert_date=alert_date
        )
        return agent_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent investigation failed for {scenario_id}: {e}")

@app.get("/api/scenarios/{scenario_id}/logs")
def get_scenario_logs(scenario_id: str):
    """
    Retrieves the SQLite database records (InvestigationLog and EvidenceRecord) for the scenario.
    """
    sc_dir = os.path.join(SCENARIOS_DIR, scenario_id)
    db_file = os.path.join(sc_dir, "ml_failure_investigator.db")
    if not os.path.exists(db_file):
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} database not found. Materialize it first.")
        
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        # Query logs
        logs = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).all()
        result = []
        for l in logs:
            evidence = session.query(EvidenceRecord).filter_by(investigation_id=l.investigation_id).all()
            result.append({
                "investigation_id": l.investigation_id,
                "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                "alert_metric": l.alert_metric,
                "alert_value": l.alert_value,
                "detected_root_cause": l.detected_root_cause,
                "confidence": l.confidence,
                "ruled_out": l.ruled_out,
                "evidence": [
                    {
                        "evidence_id": e.id,
                        "hypothesis": e.hypothesis,
                        "status": e.status,
                        "evidence_type": e.evidence_type,
                        "evidence_summary": e.evidence_summary,
                        "evidence_data": e.evidence_data
                    }
                    for e in evidence
                ]
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")
    finally:
        session.close()

@app.post("/api/evaluations")
def run_evaluations(req: EvaluationRequest):
    """
    Runs evaluations on a subset of scenarios and returns aggregate metrics.
    """
    try:
        metadata = load_metadata()
        if req.limit:
            metadata = metadata[:req.limit]
            
        results = []
        for scenario in metadata:
            res = evaluate_scenario(scenario)
            results.append(res)
            
        # Compute aggregates
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = (correct / total) * 100 if total > 0 else 0
        mean_latency = sum(r["latency_sec"] for r in results) / total if total > 0 else 0
        evidence_acc = (sum(1 for r in results if r["evidence_correct"]) / total) * 100 if total > 0 else 0
        
        # Calculate metric breakdown by failure mode
        breakdown = {}
        for r in results:
            fm = r["failure_mode"]
            if fm not in breakdown:
                breakdown[fm] = {"count": 0, "correct": 0, "evidence_correct": 0, "latency": 0.0}
            breakdown[fm]["count"] += 1
            if r["is_correct"]:
                breakdown[fm]["correct"] += 1
            if r["evidence_correct"]:
                breakdown[fm]["evidence_correct"] += 1
            breakdown[fm]["latency"] += r["latency_sec"]
            
        formatted_breakdown = []
        for fm, stats in breakdown.items():
            formatted_breakdown.append({
                "failure_type": fm,
                "count": stats["count"],
                "accuracy": (stats["correct"] / stats["count"]) * 100,
                "evidence_correctness": (stats["evidence_correct"] / stats["count"]) * 100,
                "avg_latency": stats["latency"] / stats["count"]
            })
            
        return {
            "summary": {
                "total_evaluated": total,
                "overall_accuracy": accuracy,
                "evidence_correctness": evidence_acc,
                "mean_latency": mean_latency
            },
            "breakdown": formatted_breakdown,
            "details": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation suite run failed: {e}")
