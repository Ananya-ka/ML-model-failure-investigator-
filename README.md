<div align="center">

# 🕵️‍♂️ Autonomous ML Model Failure Investigator

**An agentic, multi-signal diagnostic system that autonomously investigates production ML performance degradation, pinpoints root causes, and delivers evidence-backed audit reports.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-FF6F00.svg?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend%20API-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Vite%20Dashboard-61DAFB.svg?logo=react&logoColor=black)](https://reactjs.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry%20%26%20Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Architecture](#-system-architecture) • [Diagnostic Modes](#-supported-failure-modes) • [Quick Start](#-quick-start) • [Evaluation Suite](#-evaluation--benchmarking)

</div>

---

## 📖 Overview

When production machine learning models degrade, diagnosing the root cause typically requires tedious manual triage across database records, serving logs, git commit history, model registries, and distribution statistics.

The **Autonomous ML Model Failure Investigator** orchestrates a multi-step **LangGraph** diagnostic agent that investigates performance alerts in real-time. It gathers multi-modal diagnostic evidence, systematically eliminates candidate failure hypotheses, and synthesizes an actionable root-cause report with confidence scores and unified code diffs.

```
                   ┌────────────────────────────────────────┐
                   │    Production Metric Alert Trigger     │
                   │    (e.g., Accuracy dropped to 0.70)    │
                   └───────────────────┬────────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────┐
                     │ 1. Generate Structured Hypotheses  │
                     └─────────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────┐
                     │ 2. Deterministic Evidence Audit    │
                     │    • KS Drift Tests                │
                     │    • Git Commit Diffs              │
                     │    • MLflow Model Deltas           │
                     │    • Label Proportion Shift        │
                     │    • Feature Freshness Lag         │
                     └─────────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────┐
                     │ 3. Hypothesis Elimination Scan     │
                     └─────────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────┐
                     │ 4. Synthesize Root Cause & Log DB  │
                     └────────────────────────────────────┘
```

---

## 🎯 Supported Failure Modes

The system simulates and diagnoses **7 distinct failure classes**:

| Failure Mode | Signature & Diagnostic Evidence | Diagnostic Tool Used |
| :--- | :--- | :--- |
| **`feature_drift`** | Input feature distribution shifts downward (e.g. `income_band`). | Kolmogorov-Smirnov 2-sample test ($p < 0.05$) |
| **`bad_model_deploy`** | Model version regression registered with metric drops vs. baseline. | MLflow Client Model Registry Diff |
| **`pipeline_bug`** | Feature preprocessing code change corrupts inputs to zero. | Git Commit Log Scanner & Unified Diffs |
| **`label_shift`** | Macro ground-truth approval rate drops by >15% without feature shifts. | Serving Log Label Proportion Test |
| **`serving_skew`** | Unit multiplier mismatch (e.g. percentage vs. decimal scaling). | Point-in-time Feature Store Audit |
| **`stale_feature`** | Serving inference queries return records older than freshness threshold. | Time-lag Join (`merge_asof`) |
| **`control`** | Metric fluctuations within expected statistical confidence intervals. | Multi-Signal Nominal Verification |

---

## 🏗️ System Architecture

```mermaid
graph TD
    UI[🖥️ React + Vite Dashboard] <--> API[⚡ FastAPI Backend]
    
    subgraph Agentic Reasoning Engine [LangGraph StateGraph]
        H[Hypotheses Formulation] --> G[Evidence Gathering]
        G --> E[Hypotheses Elimination]
        E --> S[Root Cause Synthesis]
    end
    
    subgraph Diagnostic Tool Suite
        T1[📊 Feature Store Drift KS-Test]
        T2[📦 MLflow Model Registry Diff]
        T3[📜 Git Commit & Diff Inspector]
        T4[📉 Label Distribution Shift]
        T5[⏱️ Feature Store Staleness Scan]
    end

    subgraph Data Stores
        DB[(SQLite Feature Store & Logs)]
        MLF[(MLflow Run Registry)]
        GIT[(Repository Version Control)]
    end

    API --> Agentic Reasoning Engine
    G --> Diagnostic Tool Suite
    T1 & T4 & T5 --> DB
    T2 --> MLF
    T3 --> GIT
```

---

## ✨ Key Features

* **🧠 Dynamic Schema Discovery**: Automatically inspects database tables at runtime (`PRAGMA table_info`) to evaluate all feature store columns without hardcoded feature lists.
* **🛡️ Pydantic Structured Outputs**: Uses strict Pydantic schemas (`HypothesesList`, `RootCauseSynthesis`) with LangChain for guaranteed schema adherence and zero parsing failures.
* **🔍 Multi-Modal Evidence Inspector**: Interactive UI shows unified git code diffs with line-by-line syntax coloring, MLflow parameter deltas, and distribution test p-values.
* **⚡ Dual-Mode Execution**:
  * **Online Mode**: Uses frontier LLMs (`gpt-4o-mini`) with structured outputs.
  * **Deterministic Fallback/Mock Mode**: Multi-signal heuristic correlation engine for cost-free offline CI/CD test runs.
* **📊 Comprehensive Evaluation Hub**: Built-in benchmarking harness measuring diagnosis accuracy, evidence correctness, and latency across simulated scenario batches.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Node.js 18+ (for frontend)

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Ananya-ka/ML-model-failure-investigator-.git
cd "ML-model failure investigator "

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your-openai-api-key-here
# Set FORCE_MOCK=true to run in local mock/offline mode without API charges
FORCE_MOCK=false
```

### 4. Run the Application

**Start the FastAPI Backend:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Start the React Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to access the dashboard.

---

## 🧪 Running Tests

The test suite contains **30 automated unit and integration tests** covering diagnostic tools, agent graphs, scenario generators, database logging, and API endpoints:

```bash
# Run all tests using pytest
pytest

# Run tests with verbose output
pytest -v
```

---

## 📁 Repository Structure

```
.
├── src/
│   ├── agent/             # LangGraph diagnostic workflow & Pydantic schemas
│   │   └── agent.py
│   ├── api/               # FastAPI endpoints & streaming routes
│   │   └── main.py
│   ├── database/          # SQLAlchemy ORM models (ServingLogs, EvidenceRecords)
│   │   └── models.py
│   ├── harness/           # Scenario generator & evaluation benchmark harness
│   │   ├── eval_harness.py
│   │   └── scenario_generator.py
│   ├── pipeline/          # Feature store simulation & toy ML training pipeline
│   │   ├── mock_feature_store.py
│   │   ├── processing.py
│   │   ├── simulation.py
│   │   └── toy_pipeline.py
│   └── tools/             # Statistical diagnostic tools (KS test, Git, MLflow diffs)
│       └── diagnostics.py
├── frontend/              # Modern React + Vite interactive dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   └── index.css
├── scenarios/             # Materialized failure scenario databases & metadata
├── tests/                 # Comprehensive pytest test suite (30 test cases)
├── FINE_TUNING_DECISION.md# Empirical Go/No-Go report on frontier vs fine-tuned models
├── pytest.ini             # Pytest root configuration
└── requirements.txt       # Python dependencies
```

---

## 📊 Evaluation & Benchmarking

Following our [Fine-Tuning Decision Report](FINE_TUNING_DECISION.md), the prompted frontier baseline achieves **100% accuracy** and **100% evidence correctness** across our failure scenario benchmark suite at negligible cost (~$0.0003 per run):

| Failure Mode | Scenarios Evaluated | Diagnosis Accuracy | Evidence Correctness | Avg Latency (Mock) |
| :--- | :---: | :---: | :---: | :---: |
| **Control** | 1 | **100.00%** | **100.00%** | 0.42s |
| **Feature Drift** | 1 | **100.00%** | **100.00%** | 0.19s |
| **Bad Model Deploy** | 1 | **100.00%** | **100.00%** | 0.19s |
| **Pipeline Code Bug**| 1 | **100.00%** | **100.00%** | 0.58s |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Ananya-ka/ML-model-failure-investigator-/issues).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
