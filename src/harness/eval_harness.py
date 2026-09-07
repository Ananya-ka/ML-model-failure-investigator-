import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.config import BASE_DIR
from src.harness.scenario_generator import load_metadata, execute_scenario, SCENARIOS_DIR
from src.agent.agent import run_agent_investigation
from src.database.models import init_db, get_session, InvestigationLog, EvidenceRecord

EVAL_RESULTS_PATH = os.path.join(SCENARIOS_DIR, "eval_results.json")

def evaluate_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the investigator agent on a single scenario, verifying and
    materializing the database if missing, measuring latency, checking predictions,
    and verifying evidence correct logging.
    """
    scenario_id = scenario["scenario_id"]
    scenario_dir = os.path.join(SCENARIOS_DIR, scenario_id)
    db_file = os.path.join(scenario_dir, "ml_failure_investigator.db")
    
    # Self-healing: if the databases do not exist, run the simulation
    if not os.path.exists(db_file):
        print(f"[{scenario_id}] Database not found. Materializing scenario via simulation...")
        execute_scenario(scenario_id)
        
    # Get alert details from scenario metadata
    alert_date = datetime.strptime(scenario["injection_date"], "%Y-%m-%d")
    
    # In our scenarios, model performance drops after the injection date.
    # We will trigger the investigation agent with typical alert values.
    alert_metric = "accuracy"
    alert_value = 0.70 if scenario["injected_failure"] != "control" else 0.79
    
    # Set dates
    baseline_window = (datetime(2026, 7, 1), datetime(2026, 7, 30))
    active_window = (datetime(2026, 8, 1), datetime(2026, 8, 10))
    
    start_time = time.time()
    try:
        agent_output = run_agent_investigation(
            scenario_id=scenario_id,
            alert_metric=alert_metric,
            alert_value=alert_value,
            alert_date=alert_date,
            baseline_window=baseline_window,
            active_window=active_window
        )
        latency = time.time() - start_time
    except Exception as e:
        print(f"[{scenario_id}] Agent execution failed: {e}")
        return {
            "scenario_id": scenario_id,
            "failure_mode": scenario["injected_failure"],
            "predicted_cause": "error",
            "is_correct": False,
            "confidence": 0.0,
            "latency_sec": time.time() - start_time,
            "evidence_correct": False,
            "reasoning": f"Execution error: {e}"
        }
        
    # 1. Check Cause Accuracy
    predicted_cause = agent_output.get("cause", "unknown")
    expected_cause = scenario["injected_failure"]
    is_correct = (predicted_cause == expected_cause)
    
    # 2. Check Evidence Correctness by querying the scenario's SQLite database
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    session = get_session(engine)
    
    evidence_correct = True
    try:
        # Load the latest logged investigation log
        inv_log = session.query(InvestigationLog).order_by(InvestigationLog.timestamp.desc()).first()
        if not inv_log:
            evidence_correct = False
        else:
            # Query all evidence records for this investigation
            records = session.query(EvidenceRecord).filter_by(investigation_id=inv_log.investigation_id).all()
            logged_types = {r.evidence_type for r in records}
            
            # Check expected evidence matches
            # e.g., if scenario requires "git_commit_hash", we expect "git_commit" evidence type.
            for exp in scenario["evidence_should_include"]:
                if exp == "git_commit_hash" and "git_commit" not in logged_types:
                    evidence_correct = False
                elif exp == "feature_store_drift_stat" and "feature_drift" not in logged_types:
                    evidence_correct = False
                elif exp == "model_registry_diff" and "model_registry_diff" not in logged_types:
                    evidence_correct = False
                elif exp == "label_drift_stat" and "label_drift" not in logged_types:
                    evidence_correct = False
                elif exp == "feature_store_staleness_stat" and "feature_staleness" not in logged_types:
                    evidence_correct = False
    except Exception as e:
        print(f"[{scenario_id}] Failed to check evidence in DB: {e}")
        evidence_correct = False
    finally:
        session.close()
        
    return {
        "scenario_id": scenario_id,
        "failure_mode": expected_cause,
        "predicted_cause": predicted_cause,
        "is_correct": is_correct,
        "confidence": agent_output.get("confidence", 0.0),
        "latency_sec": latency,
        "evidence_correct": evidence_correct,
        "reasoning": agent_output.get("reasoning", "")
    }

def print_summary_table(results: List[Dict[str, Any]]):
    """
    Computes aggregates and prints a beautiful markdown summary table.
    """
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
        
    # Calculate False Hypothesis Rate (errored on control cases)
    control_data = breakdown.get("control", {"count": 0, "correct": 0})
    control_count = control_data["count"]
    # False hypothesis is when control is NOT correctly identified
    false_hypotheses = control_count - control_data["correct"]
    false_hyp_rate = (false_hypotheses / control_count) * 100 if control_count > 0 else 0
    
    print("\n" + "=" * 25 + " EVALUATION SUMMARY " + "=" * 25)
    print(f"Total Scenarios Evaluated: {total}")
    print(f"Overall Diagnosis Accuracy: {accuracy:.2f}%")
    print(f"Evidence Logging Correctness: {evidence_acc:.2f}%")
    print(f"False Hypothesis Rate (on controls): {false_hyp_rate:.2f}%")
    print(f"Mean Latency: {mean_latency:.2f}s")
    print("\n### Performance Breakdown by Failure Type:")
    print("| Failure Type | Count | Diagnosis Accuracy | Evidence Correctness | Avg Latency |")
    print("|--------------|-------|--------------------|----------------------|-------------|")
    
    for fm, stats in breakdown.items():
        fm_acc = (stats["correct"] / stats["count"]) * 100
        fm_ev = (stats["evidence_correct"] / stats["count"]) * 100
        avg_lat = stats["latency"] / stats["count"]
        print(f"| {fm.ljust(12)} | {str(stats['count']).rjust(5)} | {fm_acc:17.2f}% | {fm_ev:19.2f}% | {avg_lat:10.2f}s |")
    print("=" * 70 + "\n")

def run():
    parser = argparse.ArgumentParser(description="Failure Investigator Evaluation Harness")
    parser.add_argument("--limit", type=int, help="Limit the number of scenarios evaluated")
    parser.add_argument("--scenario", type=str, help="Evaluate a single scenario by ID")
    args = parser.parse_args()
    
    metadata = load_metadata()
    
    if args.scenario:
        metadata = [s for s in metadata if s["scenario_id"] == args.scenario]
        if not metadata:
            print(f"Scenario {args.scenario} not found.")
            return
            
    if args.limit:
        metadata = metadata[:args.limit]
        
    print(f"Running evaluations on {len(metadata)} scenarios...")
    results = []
    
    for scenario in metadata:
        res = evaluate_scenario(scenario)
        results.append(res)
        print(f"[{scenario['scenario_id']}] Predicted: {res['predicted_cause']} (Expected: {res['failure_mode']}) - Correct: {res['is_correct']}")
        
    # Print results summary
    print_summary_table(results)
    
    # Save results to JSON
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed logs saved to {EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    run()
