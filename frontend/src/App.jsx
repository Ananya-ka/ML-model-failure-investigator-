import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Terminal, 
  GitCommit, 
  FileText, 
  Play, 
  RefreshCw, 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  Database, 
  GitBranch, 
  BarChart2, 
  Clock, 
  TrendingDown, 
  Settings, 
  LogOut, 
  Search, 
  Bell, 
  MessageSquare, 
  X,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

const getRiskLabel = (failure) => {
  switch(failure) {
    case 'control': return { text: 'Informational', class: 'nominal' };
    case 'pipeline_bug':
    case 'bad_model_deploy':
      return { text: 'Critical Risk', class: 'critical' };
    default:
      return { text: 'High Risk', class: 'high' };
  }
};

const FAILURE_DESCRIPTIONS = {
  control: "Operational Serving Baseline: Core features and labels match statistical expectations. Anomaly threshold reports nominal variance.",
  feature_drift: "Upstream Data Drift Anomaly: Distributional drift detected on input features (e.g. income_band), indicating out-of-bounds client signals.",
  bad_model_deploy: "Model Version Mismatch: Pipeline registered version regression where shuffled prediction labels trigger immediate metric degradation.",
  pipeline_bug: "Feature Processing Corruption: Upstream code modification corrupts key evaluation vectors (credit_score), skewing inputs to zero values.",
  label_shift: "Concept Drift & Label Shift: Macro-economic concept changes degrade ground truth approvals, triggering system-wide prediction deviations.",
  serving_skew: "Training/Serving Processing Skew: Pre-processing mismatch detected. Feature metric (debt_to_income) scaled improperly relative to training schema.",
  stale_feature: "Feature Ingestion Latency: Pipeline failure detected in scheduled updates. Serving environment is retrieving stale data snapshots."
};

const STEP_EXPLANATIONS = {
  hypotheses: "Stage 1: Threat Hypotheses Generation — Formulating candidate integrity failures (feature drift, pipeline regression, configuration drift) based on anomaly payload.",
  evidence: "Stage 2: Diagnostic Scan & Audit — Running statistical distance scans (Kolmogorov-Smirnov), code history inspections, and version metadata queries.",
  eliminate: "Stage 3: Hypothesis Elimination — Systematically vetting metrics against structural thresholds to rule out false anomaly triggers.",
  synthesize: "Stage 4: Synthesis & Root Cause — Executing reasoning models over structured findings to compute primary cause vector and confidence ratings.",
  completed: "Operational Diagnosis Finalized — Structural audit successfully completed. Diagnostic records saved in local SQLite storage."
};

export default function App() {
  const [activeTab, setActiveTab] = useState('investigator');
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [investigating, setInvestigating] = useState(false);
  const [agentStatus, setAgentStatus] = useState('idle'); // idle, hypotheses, evidence, eliminate, synthesize, completed
  const [agentResult, setAgentResult] = useState(null);
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [activeEvIdxs, setActiveEvIdxs] = useState({});
  const [localTab, setLocalTab] = useState('all'); // all, active, pending, resolved
  
  // Evaluation State
  const [evalLimit, setEvalLimit] = useState(7);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalResults, setEvalResults] = useState(null);

  // Load scenarios on mount
  useEffect(() => {
    fetchScenarios();
  }, []);

  // Fetch logs whenever selected scenario changes
  useEffect(() => {
    if (selectedScenario) {
      fetchLogs(selectedScenario.scenario_id);
      setAgentResult(null);
      setAgentStatus('idle');
    }
  }, [selectedScenario]);

  const fetchScenarios = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/scenarios');
      if (res.ok) {
        const data = await res.json();
        setScenarios(data);
        if (data.length > 0 && !selectedScenario) {
          setSelectedScenario(data[0]);
        }
      }
    } catch (e) {
      console.error("Failed to fetch scenarios:", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchLogs = async (scId) => {
    try {
      const res = await fetch(`/api/scenarios/${scId}/logs`);
      if (res.ok) {
        const data = await res.json();
        setLogs(data);
      } else {
        setLogs([]);
      }
    } catch (e) {
      console.error("Failed to fetch logs:", e);
      setLogs([]);
    }
  };

  const handleSimulate = async () => {
    if (!selectedScenario) return;
    setSimulating(true);
    try {
      const res = await fetch(`/api/scenarios/${selectedScenario.scenario_id}/simulate`, {
        method: 'POST'
      });
      if (res.ok) {
        await fetchScenarios();
        const updated = scenarios.find(s => s.scenario_id === selectedScenario.scenario_id);
        if (updated) setSelectedScenario({ ...updated, is_materialized: true });
      }
    } catch (e) {
      console.error("Simulation failed:", e);
    } finally {
      setSimulating(false);
    }
  };

  const handleInvestigate = async () => {
    if (!selectedScenario) return;
    setInvestigating(true);
    setAgentResult(null);
    
    // Simulate progression
    setAgentStatus('hypotheses');
    await new Promise(r => setTimeout(r, 600));
    setAgentStatus('evidence');
    await new Promise(r => setTimeout(r, 800));
    setAgentStatus('eliminate');
    await new Promise(r => setTimeout(r, 600));
    setAgentStatus('synthesize');
    
    try {
      const res = await fetch(`/api/scenarios/${selectedScenario.scenario_id}/investigate`, {
        method: 'POST'
      });
      if (res.ok) {
        const data = await res.json();
        setAgentResult(data.final_root_cause);
        setAgentStatus('completed');
        await fetchLogs(selectedScenario.scenario_id);
      } else {
        setAgentStatus('idle');
      }
    } catch (e) {
      console.error("Investigation failed:", e);
      setAgentStatus('idle');
    } finally {
      setInvestigating(false);
    }
  };

  const handleRunEvaluation = async () => {
    setEvalLoading(true);
    setEvalResults(null);
    try {
      const res = await fetch('/api/evaluations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: evalLimit })
      });
      if (res.ok) {
        const data = await res.json();
        setEvalResults(data);
      }
    } catch (e) {
      console.error("Evaluation run failed:", e);
    } finally {
      setEvalLoading(false);
    }
  };

  const getCauseColorClass = (cause) => {
    switch(cause) {
      case 'control': return 'success';
      case 'feature_drift': return 'warning';
      case 'bad_model_deploy': return 'deploy';
      case 'pipeline_bug': return 'error';
      case 'label_shift': return 'success';
      case 'serving_skew': return 'warning';
      case 'stale_feature': return 'deploy';
      default: return 'inactive';
    }
  };

  const renderDiff = (diffText) => {
    if (!diffText) return null;
    const lines = diffText.split('\n');
    return (
      <div className="git-diff-viewer">
        <div className="git-diff-header">
          <span>Unified Code Diff</span>
          <span>Fira Code</span>
        </div>
        <div className="git-diff-lines">
          {lines.map((line, idx) => {
            let type = '';
            if (line.startsWith('+')) type = 'addition';
            else if (line.startsWith('-')) type = 'deletion';
            else if (line.startsWith('@@')) type = 'meta';
            return (
              <div key={idx} className={`git-diff-line ${type}`}>
                {line}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderConfidenceGauge = (confidence) => {
    const strokeDashoffset = 251.2 * (1 - confidence);
    return (
      <div className="radial-gauge-container">
        <svg width="180" height="110" className="gauge-svg">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#8E5BE1" />
              <stop offset="100%" stopColor="#AF8AEA" />
            </linearGradient>
          </defs>
          <path d="M 10 90 A 80 80 0 0 1 170 90" className="gauge-bg" />
          <path d="M 10 90 A 80 80 0 0 1 170 90" className="gauge-fill" style={{ strokeDashoffset, stroke: 'url(#gaugeGradient)' }} />
        </svg>
        <div className="gauge-text-container">
          <span className="gauge-value">{(confidence * 100).toFixed(0)}%</span>
          <span className="gauge-label">Confidence</span>
        </div>
      </div>
    );
  };

  const filteredScenarios = scenarios.filter(sc => {
    if (localTab === 'active') return sc.is_materialized && sc.injected_failure !== 'control';
    if (localTab === 'pending') return !sc.is_materialized;
    if (localTab === 'resolved') return sc.is_materialized && sc.injected_failure === 'control';
    return true; // all
  });

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="logo-container">
          <div className="logo-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17.93C9.72 17.65 7 14.93 7 11.65V11C7 10.45 7.45 10 8 10C8.55 10 9 10.45 9 11V11.65C9 13.83 10.82 15.65 13 15.93V13C13 12.45 13.45 12 14 12C14.55 12 15 12.45 15 13V17.93C15 18.5 14.5 19 14 19C13.5 19 13 18.5 13 17.93ZM17 11.65C17 9.47 15.18 7.65 13 7.37V9C13 9.55 12.55 10 12 10C11.45 10 11 9.55 11 9V7.37C7.72 7.65 5 10.35 5 13.65V14C5 14.55 4.55 15 4 15C3.45 15 3 14.55 3 14V13.65C3 8.35 7.35 4 12.65 4H13C17.97 4 22 8.03 22 13C22 17.97 17.97 22 13 22C12.45 22 12 21.55 12 21C12 20.45 12.45 20 13 20C16.87 20 20 16.87 20 13C20 9.13 16.87 6 13 6V7.37C15.18 7.65 17 9.47 17 11.65Z" fill="currentColor" />
            </svg>
          </div>
        </div>
        
        <ul className="nav-links">
          <li 
            className={`nav-item ${activeTab === 'investigator' ? 'active' : ''}`}
            onClick={() => setActiveTab('investigator')}
            data-tooltip="Root Cause Analyst"
          >
            <Terminal size={20} />
          </li>
          <li 
            className={`nav-item ${activeTab === 'evaluations' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluations')}
            data-tooltip="Evaluation Hub"
          >
            <Activity size={20} />
          </li>
        </ul>
        
        <div className="sidebar-footer" style={{ display: 'flex', flexDirection: 'column', gap: '16px', alignItems: 'center' }}>
          <Settings size={20} style={{ color: 'var(--accent-text-secondary)' }} />
          <div style={{ width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'rgba(142, 91, 225, 0.15)', border: '1px solid var(--border-card)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.1rem', fontWeight: 600 }}>
            👨‍💻
          </div>
        </div>
      </div>

      {/* Main Workspace Area */}
      <div className="main-workspace">
        
        {/* Top bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '20px' }}>
          <div className="header-title">
            <span className="overtitle-category">NIGHTFALL AI // ANOMALY DETECTION & RESPONSE</span>
            <h1>Model Performance Integrity</h1>
            <p>Autonomous diagnostics investigating statistical drift, label shifts, and processing pipeline regressions in real time.</p>
          </div>
          
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-card)', borderRadius: '24px', padding: '6px 16px', gap: '8px', width: '220px', boxShadow: 'var(--shadow-md)' }}>
              <Search size={16} color="var(--accent-text-secondary)" />
              <input type="text" placeholder="Search scenarios..." style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '0.82rem', width: '100%', color: 'var(--accent-text-primary)' }} />
            </div>
            
            <div style={{ width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-card)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-text-secondary)', cursor: 'pointer', boxShadow: 'var(--shadow-md)' }}>
              <MessageSquare size={18} />
            </div>
            <div style={{ width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-card)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-text-secondary)', cursor: 'pointer', position: 'relative', boxShadow: 'var(--shadow-md)' }}>
              <Bell size={18} />
              <span style={{ position: 'absolute', top: '10px', right: '10px', width: '6px', height: '6px', backgroundColor: 'var(--accent-rose)', borderRadius: '50%' }}></span>
            </div>
          </div>
        </div>

        {/* Collapsible Quick Start onboarding banner */}
        {showOnboarding && activeTab === 'investigator' && (
          <div className="glass-card" style={{ borderLeft: '4px solid var(--accent-purple)', position: 'relative' }}>
            <button 
              onClick={() => setShowOnboarding(false)} 
              style={{ position: 'absolute', top: '16px', right: '16px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--accent-text-secondary)' }}
            >
              <X size={18} />
            </button>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>👋</span> Quick Start Guide — ML Failure Investigator
            </h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--accent-text-secondary)', lineHeight: '1.5', marginBottom: '14px', maxWidth: '800px' }}>
              Welcome! This environment tests an autonomous, LangGraph-powered model failure analyst. 
              The agent runs structural diagnostic pipelines comparing model states, git code modifications, serving inputs, and label shifts to identify performance root causes.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '16px', borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '14px' }}>
              <div>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>1. Pick Alert</span>
                <p style={{ fontSize: '0.78rem', color: 'var(--accent-text-secondary)', marginTop: '4px' }}>Select an alert scenario from the left panel (e.g. sc_004).</p>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>2. Simulate</span>
                <p style={{ fontSize: '0.78rem', color: 'var(--accent-text-secondary)', marginTop: '4px' }}>If database files do not exist, click <strong>Simulate Failure</strong> to write SQLite logs.</p>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>3. Investigate</span>
                <p style={{ fontSize: '0.78rem', color: 'var(--accent-text-secondary)', marginTop: '4px' }}>Click <strong>Investigate Alert</strong>. Watch the stepper state transition live.</p>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>4. Audit Logs</span>
                <p style={{ fontSize: '0.78rem', color: 'var(--accent-text-secondary)', marginTop: '4px' }}>Review the final conclusion, confidence scores, and code diff panels.</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'investigator' && (
          <>
            <div className="dashboard-grid">
              
              {/* Left Column: Scenario selector */}
              <div className="scenario-selector-panel">
                <span className="section-label">Incident Directory</span>
                <div className="scenario-list">
                  {loading ? (
                    <div style={{ color: 'var(--accent-text-muted)', padding: '10px', fontSize: '0.85rem' }}>Loading incidents...</div>
                  ) : (
                    filteredScenarios.map((sc, idx) => {
                      const isActive = selectedScenario?.scenario_id === sc.scenario_id;
                      const risk = getRiskLabel(sc.injected_failure);
                      const idxStr = (idx + 1).toString().padStart(2, '0');
                      
                      return (
                        <div 
                          key={sc.scenario_id} 
                          className={`scenario-item-card ${isActive ? 'active' : ''}`}
                          onClick={() => setSelectedScenario(sc)}
                        >
                          <div style={{ display: 'flex', alignItems: 'flex-start', width: '100%' }}>
                            <span className="sc-index-number">{idxStr}</span>
                            <div className="sc-info-left" style={{ flexGrow: 1, minWidth: 0 }}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', gap: '8px' }}>
                                <span className="sc-title-text" style={{ fontSize: '0.9rem', fontWeight: isActive ? 700 : 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                  {sc.injected_failure.split('_').join(' ')}
                                </span>
                                <span className={`risk-tag ${risk.class}`} style={{ transform: 'scale(0.85)', transformOrigin: 'right', flexShrink: 0 }}>
                                  {risk.text}
                                </span>
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                                <span className="sc-id-badge" style={{ fontSize: '0.72rem', color: 'var(--accent-text-muted)' }}>{sc.scenario_id}</span>
                                <span style={{ fontSize: '0.7rem', color: 'var(--accent-text-muted)' }}>|</span>
                                <span style={{ fontSize: '0.7rem', color: 'var(--accent-text-muted)' }}>{sc.injection_date}</span>
                              </div>
                              {isActive && (
                                <p className="sc-desc-collapsed">
                                  {FAILURE_DESCRIPTIONS[sc.injected_failure]}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Right Column: Execution & Findings */}
              <div className="diagnosis-panel-wrapper">
                
                {/* Tabs Row matching Inspo */}
                <div className="nav-tab-row">
                  <button 
                    className={`nav-tab-pill ${localTab === 'all' ? 'active' : ''}`}
                    onClick={() => setLocalTab('all')}
                  >
                    All Alerts
                  </button>
                  <button 
                    className={`nav-tab-pill ${localTab === 'active' ? 'active' : ''}`}
                    onClick={() => setLocalTab('active')}
                  >
                    Active Threats
                  </button>
                  <button 
                    className={`nav-tab-pill ${localTab === 'pending' ? 'active' : ''}`}
                    onClick={() => setLocalTab('pending')}
                  >
                    Pending Queue
                  </button>
                  <button 
                    className={`nav-tab-pill ${localTab === 'resolved' ? 'active' : ''}`}
                    onClick={() => setLocalTab('resolved')}
                  >
                    Resolved Log
                  </button>
                </div>

                {/* Table Control Row */}
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap', paddingBottom: '8px', marginBottom: '10px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', backgroundColor: '#ffffff', border: '1px solid rgba(0,0,0,0.08)', borderRadius: '20px', padding: '6px 14px', gap: '8px', width: '240px', boxShadow: 'var(--shadow-sm)' }}>
                    <Search size={14} color="var(--accent-text-secondary)" />
                    <input 
                      type="text" 
                      placeholder="Search alerts directory..." 
                      style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '0.8rem', width: '100%', color: 'var(--accent-text-primary)' }} 
                    />
                  </div>
                  
                  <button className="btn" style={{ borderRadius: '20px', padding: '6px 14px', fontSize: '0.8rem' }}>
                    <Settings size={14} style={{ color: 'var(--accent-text-secondary)' }} /> Filter
                  </button>
                  <button className="btn" style={{ borderRadius: '20px', padding: '6px 14px', fontSize: '0.8rem' }}>
                    Last 30 Days
                  </button>
                  <button className="btn" style={{ borderRadius: '20px', padding: '6px 14px', fontSize: '0.8rem', marginLeft: 'auto' }}>
                    Export to CSV
                  </button>
                </div>

                {/* Active Scenario Card */}
                {selectedScenario && (
                  <div className="glass-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <span className="section-label" style={{ fontSize: '0.7rem' }}>Selected Incident Target</span>
                        <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span>Incident ID: {selectedScenario.scenario_id}</span>
                          <span className={`risk-tag ${getRiskLabel(selectedScenario.injected_failure).class}`}>
                            {getRiskLabel(selectedScenario.injected_failure).text}
                          </span>
                        </h2>
                        <p style={{ fontSize: '0.82rem', color: 'var(--accent-text-secondary)', marginTop: '4px' }}>
                          Sensor Logged Date: {selectedScenario.injection_date} | Category: <span style={{ textTransform: 'capitalize' }}>{selectedScenario.injected_failure.split('_').join(' ')}</span>
                        </p>
                      </div>
                      
                      <div style={{ display: 'flex', gap: '10px' }}>
                        <button 
                          className="btn" 
                          onClick={handleSimulate}
                          disabled={simulating || selectedScenario.is_materialized}
                        >
                          {simulating ? <RefreshCw className="spin" size={14} /> : <Play size={14} />}
                          <span>{selectedScenario.is_materialized ? 'Simulated' : 'Simulate Failure'}</span>
                        </button>
                        
                        <button 
                          className="btn btn-primary"
                          onClick={handleInvestigate}
                          disabled={investigating || !selectedScenario.is_materialized}
                        >
                          {investigating ? <RefreshCw className="spin" size={14} /> : <Activity size={14} />}
                          <span>Investigate Alert</span>
                        </button>
                      </div>
                    </div>

                    {/* Failure Mode Explanation is now embedded in the left directory list item */}

                    {!selectedScenario.is_materialized && (
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', backgroundColor: 'var(--accent-rose-dim)', border: '1px solid rgba(192, 104, 99, 0.1)', padding: '12px 16px', borderRadius: '12px', color: 'var(--accent-rose)', fontSize: '0.82rem', marginTop: '16px' }}>
                        <AlertTriangle size={15} />
                        <span>Database files missing. Click <strong>Simulate Failure</strong> to run the point-in-time generation.</span>
                      </div>
                    )}
                  </div>
                )}

                {/* State graph progression */}
                {(investigating || agentStatus !== 'idle') && (
                  <div className="glass-card">
                    <span className="section-label">LangGraph Agent Pipeline Execution</span>
                    <div className="state-visualizer">
                      <div className={`state-node ${['hypotheses', 'evidence', 'eliminate', 'synthesize', 'completed'].includes(agentStatus) ? (agentStatus === 'hypotheses' ? 'active' : 'completed') : ''}`}>
                        <div className="node-dot">1</div>
                        <span className="node-label">Hypotheses</span>
                      </div>
                      <div className={`state-node ${['evidence', 'eliminate', 'synthesize', 'completed'].includes(agentStatus) ? (agentStatus === 'evidence' ? 'active' : 'completed') : ''}`}>
                        <div className="node-dot">2</div>
                        <span className="node-label">Evidence</span>
                      </div>
                      <div className={`state-node ${['eliminate', 'synthesize', 'completed'].includes(agentStatus) ? (agentStatus === 'eliminate' ? 'active' : 'completed') : ''}`}>
                        <div className="node-dot">3</div>
                        <span className="node-label">Eliminate</span>
                      </div>
                      <div className={`state-node ${['synthesize', 'completed'].includes(agentStatus) ? (agentStatus === 'synthesize' ? 'active' : 'completed') : ''}`}>
                        <div className="node-dot">4</div>
                        <span className="node-label">Synthesize</span>
                      </div>
                    </div>

                    {/* Explains what the agent is currently doing in the stepper */}
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', marginTop: '20px', padding: '12px 16px', backgroundColor: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', color: 'var(--accent-text-secondary)', fontSize: '0.82rem' }}>
                      <Info size={16} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--accent-purple)' }} />
                      <p>{STEP_EXPLANATIONS[agentStatus] || 'Agent is idle. Trigger an investigation to start.'}</p>
                    </div>
                  </div>
                )}

                {/* Root Cause Synthesis Report */}
                {agentResult && agentStatus === 'completed' && (
                  <div className="glass-card">
                    <div className="diagnosis-header-card">
                      <div className="root-cause-display">
                        <span className="section-label">Root Cause Diagnostic</span>
                        <span className={`cause-badge ${getCauseColorClass(agentResult.cause)}`}>
                          {agentResult.cause.replace('_', ' ')}
                        </span>
                      </div>
                      {renderConfidenceGauge(agentResult.confidence)}
                    </div>
                    
                    <div style={{ marginTop: '20px' }}>
                      <p className="reasoning-text">{agentResult.reasoning}</p>
                    </div>

                    <div style={{ marginTop: '20px' }}>
                      <span className="section-label">Ruled Out Hypotheses</span>
                      <div className="ruled-out-list">
                        {agentResult.ruled_out.map((ro, idx) => (
                          <span key={idx} className="ruled-out-item">{ro.replace('_', ' ')}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Evidence Records list */}
                {logs.length > 0 && (
                  <div>
                    <span className="section-label">Findings & Evidence Logs</span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '10px' }}>
                      {logs.map((log) => {
                        const currentIdx = activeEvIdxs[log.investigation_id] || 0;
                        const ev = log.evidence[currentIdx];
                        
                        return (
                          <div key={log.investigation_id} className="glass-card" style={{ borderTop: `4px solid var(--accent-${getCauseColorClass(log.detected_root_cause)})` }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '12px' }}>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                <span style={{ fontSize: '0.8rem', color: 'var(--accent-text-primary)', fontWeight: 700 }}>
                                  Root Cause: {log.detected_root_cause.replace('_', ' ')}
                                </span>
                                <span style={{ fontSize: '0.72rem', color: 'var(--accent-text-muted)', fontFamily: 'var(--font-mono)' }}>
                                  ID: {log.investigation_id} | {log.timestamp}
                                </span>
                              </div>
                              
                              {/* Swiper Chevron Navigation Controls */}
                              {log.evidence.length > 1 && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                  <button 
                                    className="carousel-nav-btn" 
                                    disabled={currentIdx === 0}
                                    onClick={() => setActiveEvIdxs({ ...activeEvIdxs, [log.investigation_id]: currentIdx - 1 })}
                                    title="Previous Evidence"
                                  >
                                    <ChevronLeft size={16} />
                                  </button>
                                  <span style={{ fontSize: '0.78rem', color: 'var(--accent-text-secondary)', fontWeight: 700, fontFamily: 'var(--font-mono)', minWidth: '40px', textAlign: 'center' }}>
                                    {currentIdx + 1} / {log.evidence.length}
                                  </span>
                                  <button 
                                    className="carousel-nav-btn"
                                    disabled={currentIdx === log.evidence.length - 1}
                                    onClick={() => setActiveEvIdxs({ ...activeEvIdxs, [log.investigation_id]: currentIdx + 1 })}
                                    title="Next Evidence"
                                  >
                                    <ChevronRight size={16} />
                                  </button>
                                </div>
                              )}
                            </div>

                            {/* Render single active evidence swipe card */}
                            {ev && (
                              <div className="carousel-card-wrapper" key={ev.evidence_id}>
                                <div className="evidence-title-bar">
                                  {ev.evidence_type === 'git_commit' && <GitCommit size={15} color="var(--accent-rose)" />}
                                  {ev.evidence_type === 'feature_drift' && <TrendingDown size={15} color="var(--accent-amber)" />}
                                  {ev.evidence_type === 'model_registry_diff' && <GitBranch size={15} color="var(--accent-purple)" />}
                                  {ev.evidence_type === 'label_drift' && <TrendingDown size={15} color="var(--accent-sage)" />}
                                  {ev.evidence_type === 'feature_staleness' && <Clock size={15} color="var(--accent-purple)" />}
                                  <span className="evidence-title-text">{ev.hypothesis}</span>
                                  <span className="drift-badge normal" style={{ marginLeft: 'auto', fontSize: '0.7rem' }}>
                                    {ev.evidence_type}
                                  </span>
                                </div>
                                <p style={{ fontSize: '0.88rem', color: 'var(--accent-text-secondary)', marginBottom: '12px', lineHeight: '1.4' }}>
                                  {ev.evidence_summary}
                                </p>

                                {/* Feature Drift render */}
                                {ev.evidence_type === 'feature_drift' && ev.evidence_data && (
                                  <div>
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', backgroundColor: 'var(--accent-amber-dim)', padding: '10px 14px', borderRadius: '8px', color: 'var(--accent-amber)', fontSize: '0.8rem', marginBottom: '10px' }}>
                                      <Info size={14} style={{ flexShrink: 0 }} />
                                      <span>The Kolmogorov-Smirnov test measures if serving inputs deviate from training distributions. p-value &lt; 0.05 indicates statistical drift.</span>
                                    </div>
                                    <table className="data-table">
                                      <thead>
                                        <tr>
                                          <th>Feature</th>
                                          <th>KS Statistic</th>
                                          <th>P-Value</th>
                                          <th>Status</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        <tr>
                                          <td style={{ fontWeight: 600 }}>{ev.evidence_data.feature_name}</td>
                                          <td>{ev.evidence_data.ks_statistic?.toFixed(4)}</td>
                                          <td>{ev.evidence_data.p_value?.toFixed(6)}</td>
                                          <td>
                                            <span className={`drift-badge ${ev.evidence_data.is_drifted ? 'drifted' : 'normal'}`}>
                                              {ev.evidence_data.is_drifted ? 'DRIFTED' : 'NORMAL'}
                                            </span>
                                          </td>
                                        </tr>
                                      </tbody>
                                    </table>
                                  </div>
                                )}

                                {/* Git Commit render */}
                                {ev.evidence_type === 'git_commit' && ev.evidence_data && (
                                  <div>
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', backgroundColor: 'var(--accent-rose-dim)', padding: '10px 14px', borderRadius: '8px', color: 'var(--accent-rose)', fontSize: '0.8rem', marginBottom: '10px' }}>
                                      <Info size={14} style={{ flexShrink: 0 }} />
                                      <span>Commit scanner scans repository logs within target timeframe. Red (-) and green (+) diff lines show changed pre-processing functions.</span>
                                    </div>
                                    <div style={{ marginTop: '10px' }}>
                                      {ev.evidence_data.diffs && ev.evidence_data.diffs.map((d, dIdx) => (
                                        <div key={dIdx} style={{ marginBottom: '12px' }}>
                                          <p style={{ fontSize: '0.78rem', color: 'var(--accent-text-secondary)', marginBottom: '4px' }}>
                                            Modified File: {d.file_path}
                                          </p>
                                          {renderDiff(d.diff_text)}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {/* Model Registry render */}
                                {ev.evidence_type === 'model_registry_diff' && ev.evidence_data && (
                                  <div>
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', backgroundColor: 'var(--accent-purple-dim)', padding: '10px 14px', borderRadius: '8px', color: 'var(--accent-purple)', fontSize: '0.8rem', marginBottom: '10px' }}>
                                      <Info size={14} style={{ flexShrink: 0 }} />
                                      <span>Model registry queries hyperparameter discrepancies and training evaluation metrics between version iterations in MLflow.</span>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '10px' }}>
                                      <div className="glass-card" style={{ padding: '14px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: 'none' }}>
                                        <h4 style={{ fontSize: '0.8rem', marginBottom: '8px', color: 'var(--accent-text-secondary)', fontWeight: 700 }}>
                                          Baseline (Previous Version)
                                        </h4>
                                        <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent-text-primary)' }}>
                                          {JSON.stringify(ev.evidence_data.previous_version_metrics, null, 2)}
                                        </pre>
                                      </div>
                                      <div className="glass-card" style={{ padding: '14px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: 'none' }}>
                                        <h4 style={{ fontSize: '0.8rem', marginBottom: '8px', color: 'var(--accent-text-secondary)', fontWeight: 700 }}>
                                          Deployed (Latest Version)
                                        </h4>
                                        <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent-text-primary)' }}>
                                          {JSON.stringify(ev.evidence_data.latest_version_metrics, null, 2)}
                                        </pre>
                                      </div>
                                    </div>
                                  </div>
                                )}

                                {/* Label Drift render */}
                                {ev.evidence_type === 'label_drift' && ev.evidence_data && (
                                  <div>
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', backgroundColor: 'var(--accent-sage-dim)', padding: '10px 14px', borderRadius: '8px', color: 'var(--accent-sage)', fontSize: '0.8rem', marginBottom: '10px' }}>
                                      <Info size={14} style={{ flexShrink: 0 }} />
                                      <span>Label proportion test compares macro true classification ratios before and after the alert event. Reductions &gt; 15% indicate drift.</span>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '10px' }}>
                                      <div className="glass-card" style={{ padding: '16px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: 'none' }}>
                                        <h4 style={{ fontSize: '0.8rem', color: 'var(--accent-text-secondary)', marginBottom: '4px' }}>Pre-Alert Approval Rate</h4>
                                        <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-sage)' }}>
                                          {(ev.evidence_data.baseline_mean * 100).toFixed(2)}%
                                        </div>
                                      </div>
                                      <div className="glass-card" style={{ padding: '16px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: 'none' }}>
                                        <h4 style={{ fontSize: '0.8rem', color: 'var(--accent-text-secondary)', marginBottom: '4px' }}>Post-Alert Approval Rate</h4>
                                        <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent-rose)' }}>
                                          {(ev.evidence_data.target_mean * 100).toFixed(2)}%
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                )}

                                {/* Feature Staleness render */}
                                {ev.evidence_type === 'feature_staleness' && ev.evidence_data && (
                                  <div>
                                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', backgroundColor: 'var(--accent-purple-dim)', padding: '10px 14px', borderRadius: '8px', color: 'var(--accent-purple)', fontSize: '0.8rem', marginBottom: '10px' }}>
                                      <Info size={14} style={{ flexShrink: 0 }} />
                                      <span>Measures time latency delta between model inference requests and feature store updates. Average lags &gt; 0.5 days are flagged.</span>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', marginTop: '10px' }}>
                                      <div className="glass-card" style={{ padding: '14px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: 'none' }}>
                                        <h4 style={{ fontSize: '0.75rem', color: 'var(--accent-text-secondary)', marginBottom: '4px' }}>Matched Entries</h4>
                                        <div style={{ fontSize: '1.2rem', fontWeight: 800 }}>{ev.evidence_data.matched_count}</div>
                                      </div>
                                      <div className="glass-card" style={{ padding: '14px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: 'none' }}>
                                        <h4 style={{ fontSize: '0.75rem', color: 'var(--accent-text-secondary)', marginBottom: '4px' }}>Average Age Lag</h4>
                                        <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-rose)' }}>
                                          {ev.evidence_data.mean_lag_days.toFixed(2)} days
                                        </div>
                                      </div>
                                      <div className="glass-card" style={{ padding: '14px', backgroundColor: 'rgba(255, 255, 255, 0.02)', border: 'none' }}>
                                        <h4 style={{ fontSize: '0.75rem', color: 'var(--accent-text-secondary)', marginBottom: '4px' }}>Maximum Lag</h4>
                                        <div style={{ fontSize: '1.2rem', fontWeight: 800 }}>{ev.evidence_data.max_lag_days.toFixed(2)} days</div>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Pagination dots */}
                            {log.evidence.length > 1 && (
                              <div className="carousel-dots-container">
                                {log.evidence.map((_, dotIdx) => (
                                  <button 
                                    key={dotIdx} 
                                    className={`carousel-dot ${currentIdx === dotIdx ? 'active' : ''}`}
                                    onClick={() => setActiveEvIdxs({ ...activeEvIdxs, [log.investigation_id]: dotIdx })}
                                  />
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {activeTab === 'evaluations' && (
          <>
            {/* Evaluations Suite Hub */}
            <div className="evaluations-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span className="section-label">Agent Validation Suite</span>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Harness Evaluations</h2>
              </div>
              
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <label style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--accent-text-muted)' }}>Evaluation Limit</label>
                  <select 
                    value={evalLimit} 
                    onChange={(e) => setEvalLimit(Number(e.target.value))}
                    style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border-card)', backgroundColor: '#0c081e', color: 'var(--accent-text-primary)', fontSize: '0.85rem', fontWeight: 500 }}
                  >
                    <option value={7}>First 7 Scenarios</option>
                    <option value={14}>First 14 Scenarios</option>
                    <option value={40}>All 40 Scenarios</option>
                  </select>
                </div>
                
                <button 
                  className="btn btn-primary"
                  onClick={handleRunEvaluation}
                  disabled={evalLoading}
                  style={{ alignSelf: 'flex-end', height: '36px' }}
                >
                  {evalLoading ? <RefreshCw className="spin" size={14} /> : <BarChart2 size={14} />}
                  <span>Run Evaluations</span>
                </button>
              </div>
            </div>

            {evalLoading && (
              <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '60px' }}>
                <RefreshCw className="spin" size={32} color="var(--accent-sage)" />
                <p style={{ marginTop: '16px', color: 'var(--accent-text-secondary)', fontSize: '0.9rem' }}>Executing test suite on scenarios dynamically...</p>
              </div>
            )}

            {evalResults && !evalLoading && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                
                <div className="summary-metric-cards">
                  <div className="summary-metric-card">
                    <span className="summary-metric-label">Evaluated Scenarios</span>
                    <span className="summary-metric-value">{evalResults.summary.total_evaluated}</span>
                  </div>
                  <div className="summary-metric-card" style={{ borderTop: '4px solid var(--accent-sage)' }}>
                    <span className="summary-metric-label">Diagnosis Accuracy</span>
                    <span className="summary-metric-value" style={{ color: 'var(--accent-sage)' }}>
                      {evalResults.summary.overall_accuracy.toFixed(2)}%
                    </span>
                  </div>
                  <div className="summary-metric-card">
                    <span className="summary-metric-label">Evidence correctness</span>
                    <span className="summary-metric-value" style={{ color: 'var(--accent-purple)' }}>
                      {evalResults.summary.evidence_correctness.toFixed(2)}%
                    </span>
                  </div>
                  <div className="summary-metric-card">
                    <span className="summary-metric-label">Average Latency</span>
                    <span className="summary-metric-value">{evalResults.summary.mean_latency.toFixed(2)}s</span>
                  </div>
                </div>

                <div className="glass-card">
                  <h3 style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '16px' }}>Failure Mode Performance Breakdown</h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Failure Type</th>
                        <th>Scenarios Count</th>
                        <th>Diagnosis Accuracy</th>
                        <th>Evidence Correctness</th>
                        <th>Avg Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalResults.breakdown.map((row, idx) => (
                        <tr key={idx}>
                          <td style={{ fontWeight: 600, textTransform: 'capitalize' }}>{row.failure_type.replace('_', ' ')}</td>
                          <td>{row.count}</td>
                          <td style={{ color: row.accuracy === 100 ? 'var(--accent-sage)' : 'var(--accent-text-primary)', fontWeight: 700 }}>
                            {row.accuracy.toFixed(2)}%
                          </td>
                          <td>{row.evidence_correctness.toFixed(2)}%</td>
                          <td>{row.avg_latency.toFixed(2)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="glass-card">
                  <h3 style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '16px' }}>Detailed Scenarios Run List</h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Scenario ID</th>
                        <th>Expected Cause</th>
                        <th>Predicted Cause</th>
                        <th>Outcome</th>
                        <th>Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalResults.details.map((detail, idx) => (
                        <tr key={idx}>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{detail.scenario_id}</td>
                          <td style={{ textTransform: 'capitalize' }}>{detail.failure_mode.replace('_', ' ')}</td>
                          <td style={{ textTransform: 'capitalize' }}>{detail.predicted_cause.replace('_', ' ')}</td>
                          <td>
                            {detail.is_correct ? (
                              <span style={{ color: 'var(--accent-sage)', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.85rem', fontWeight: 700 }}>
                                <CheckCircle size={14} /> Correct
                              </span>
                            ) : (
                              <span style={{ color: 'var(--accent-rose)', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.85rem', fontWeight: 700 }}>
                                <AlertTriangle size={14} /> Failed
                              </span>
                            )}
                          </td>
                          <td>{detail.latency_sec.toFixed(2)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

              </div>
            )}

          </>
        )}

      </div>
    </div>
  );
}
