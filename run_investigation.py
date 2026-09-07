import os
import argparse
import json
from datetime import datetime
from src.agent.agent import run_agent_investigation

def main():
    parser = argparse.ArgumentParser(description="Autonomous ML Model Failure Investigator CLI")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario ID (e.g., sc_001, sc_002, sc_003, sc_004)")
    parser.add_argument("--metric", type=str, default="accuracy", help="Alert metric name (default: accuracy)")
    parser.add_argument("--value", type=float, default=0.79, help="Alert metric value (default: 0.79)")
    parser.add_argument("--date", type=str, default="2026-08-08", help="Alert date in YYYY-MM-DD format (default: 2026-08-08)")
    
    args = parser.parse_args()
    
    # Parse alert date
    try:
        alert_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Must be YYYY-MM-DD.")
        return
        
    print("=" * 60)
    print(f"LAUNCHING INVESTIGATION AGENT ON SCENARIO: {args.scenario}")
    print("=" * 60)
    
    # Define baseline and active windows
    baseline_window = (datetime(2026, 7, 1), datetime(2026, 7, 30))
    active_window = (datetime(2026, 8, 1), datetime(2026, 8, 10))
    
    try:
        # Run agent
        root_cause = run_agent_investigation(
            scenario_id=args.scenario,
            alert_metric=args.metric,
            alert_value=args.value,
            alert_date=alert_date,
            baseline_window=baseline_window,
            active_window=active_window
        )
        
        # Display output
        print("\n" + "=" * 25 + " AGENT DIAGNOSIS " + "=" * 25)
        print(json.dumps(root_cause, indent=2))
        print("=" * 67 + "\n")
        
    except Exception as e:
        print(f"Investigation failed with error: {e}")

if __name__ == "__main__":
    main()
