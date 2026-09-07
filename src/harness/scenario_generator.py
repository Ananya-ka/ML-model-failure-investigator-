import os
import json
import argparse
from datetime import datetime, timedelta
import random
from src.config import BASE_DIR
from src.pipeline.simulation import run_simulation

SCENARIOS_DIR = os.path.join(BASE_DIR, "scenarios")

def generate_metadata(num_scenarios: int = 40):
    """
    Generates metadata for N scenarios across 4 failure modes and saves to scenarios/metadata.json.
    """
    os.makedirs(SCENARIOS_DIR, exist_ok=True)
    
    failure_modes = ["control", "feature_drift", "bad_model_deploy", "pipeline_bug", "label_shift", "serving_skew", "stale_feature"]
    scenarios = []
    
    # We want a balanced set of scenarios
    random.seed(42)
    
    for i in range(1, num_scenarios + 1):
        scenario_id = f"sc_{i:03d}"
        
        # Balance failure modes
        failure_mode = failure_modes[(i - 1) % len(failure_modes)]
        
        # Random injection date between August 3 and August 8, 2026
        injection_day = random.randint(3, 8)
        injection_date = datetime(2026, 8, injection_day)
        
        injected_feature = None
        true_root_cause = ""
        evidence_should_include = []
        
        if failure_mode == "control":
            true_root_cause = "No failure injected. Performance degradation is due to normal statistical noise."
            evidence_should_include = []
        elif failure_mode == "feature_drift":
            injected_feature = "income_band"
            true_root_cause = "Upstream feature drift. The distribution of input feature 'income_band' shifted downward."
            evidence_should_include = ["feature_store_drift_stat"]
        elif failure_mode == "bad_model_deploy":
            true_root_cause = "Bad model deployment. Model version 2 was trained with shuffled labels, resulting in random predictions."
            evidence_should_include = ["model_registry_diff"]
        elif failure_mode == "pipeline_bug":
            injected_feature = "credit_score"
            true_root_cause = "Transformation bug in credit_score processing causing a subset of values to corrupt to 0."
            evidence_should_include = ["git_commit_hash", "feature_store_drift_stat"]
        elif failure_mode == "label_shift":
            true_root_cause = "Label distribution shift. Macro-economic changes caused a drop in borrower approval rates."
            evidence_should_include = ["label_drift_stat"]
        elif failure_mode == "serving_skew":
            injected_feature = "debt_to_income"
            true_root_cause = "Training/serving skew. Preprocessing mismatch where debt_to_income is scaled as percentage (x100) instead of decimal."
            evidence_should_include = ["feature_store_drift_stat"]
        elif failure_mode == "stale_feature":
            true_root_cause = "Stale features. Feature store ingestion pipeline crashed, serving outdated borrower features from the historical training period."
            evidence_should_include = ["feature_store_staleness_stat"]
            
        scenarios.append({
            "scenario_id": scenario_id,
            "injected_failure": failure_mode,
            "injected_feature": injected_feature,
            "injection_date": injection_date.strftime("%Y-%m-%d"),
            "true_root_cause": true_root_cause,
            "evidence_should_include": evidence_should_include
        })
        
    metadata_path = os.path.join(SCENARIOS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(scenarios, f, indent=2)
        
    print(f"Generated metadata for {num_scenarios} scenarios at {metadata_path}")
    return scenarios

def load_metadata():
    metadata_path = os.path.join(SCENARIOS_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        return generate_metadata()
    with open(metadata_path, "r") as f:
        return json.load(f)

def execute_scenario(scenario_id: str):
    """
    Runs the simulation for a single scenario and saves files to scenarios/sc_XXX/
    """
    metadata = load_metadata()
    scenario = next((s for s in metadata if s["scenario_id"] == scenario_id), None)
    
    if not scenario:
        print(f"Scenario {scenario_id} not found in metadata.json")
        return
        
    # Setup directory
    scenario_dir = os.path.join(SCENARIOS_DIR, scenario_id)
    os.makedirs(scenario_dir, exist_ok=True)
    
    db_url = f"sqlite:///{os.path.join(scenario_dir, 'ml_failure_investigator.db')}"
    mlflow_uri = f"sqlite:///{os.path.join(scenario_dir, 'mlflow.db')}"
    
    injection_date = datetime.strptime(scenario["injection_date"], "%Y-%m-%d")
    
    git_hash = run_simulation(
        db_url=db_url,
        mlflow_uri=mlflow_uri,
        scenario_id=scenario_id,
        failure_mode=scenario["injected_failure"],
        injection_date=injection_date
    )
    
    # If a git commit was made, append its hash to the metadata for verification
    if git_hash:
        scenario["git_commit_hash"] = git_hash
        # Update metadata.json with the actual git commit hash
        metadata_path = os.path.join(SCENARIOS_DIR, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    print(f"Scenario {scenario_id} executed successfully. Data saved in {scenario_dir}")

def run():
    parser = argparse.ArgumentParser(description="Scenario Generator & Runner")
    parser.add_argument("--generate-metadata", action="store_true", help="Generate scenarios metadata.json")
    parser.add_argument("--run-scenario", type=str, help="Run simulation for a specific scenario_id")
    parser.add_argument("--run-all", action="store_true", help="Run simulation for all scenarios")
    
    args = parser.parse_args()
    
    if args.generate_metadata:
        generate_metadata()
    elif args.run_scenario:
        execute_scenario(args.run_scenario)
    elif args.run_all:
        metadata = load_metadata()
        print(f"Running all {len(metadata)} scenarios...")
        for scenario in metadata:
            try:
                execute_scenario(scenario["scenario_id"])
            except Exception as e:
                print(f"Failed to execute scenario {scenario['scenario_id']}: {e}")
    else:
        # Default behavior: generate metadata if not exists
        load_metadata()
        parser.print_help()

if __name__ == "__main__":
    run()
