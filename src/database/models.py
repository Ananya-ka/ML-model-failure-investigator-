from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
from src.config import DATABASE_URL

Base = declarative_base()

class ServingLog(Base):
    """
    Stores individual prediction events during serving.
    Used for monitoring model performance and feature/label drift.
    """
    __tablename__ = "serving_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    income_band = Column(Integer, nullable=False)
    credit_score = Column(Integer, nullable=False)
    debt_to_income = Column(Float, nullable=False)
    employment_years = Column(Integer, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    true_label = Column(Integer, nullable=True)  # Ground truth label, populated after some time

class InvestigationLog(Base):
    """
    Logs each failure investigation triggered by a performance drop.
    """
    __tablename__ = "investigation_logs"
    
    investigation_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    alert_metric = Column(String, nullable=False)      # e.g., "accuracy" or "f1_score"
    alert_value = Column(Float, nullable=False)       # The value that triggered the alert
    status = Column(String, default="running")        # running, completed, failed
    detected_root_cause = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    ruled_out = Column(JSON, nullable=True)           # List of hypotheses ruled out with reason

class EvidenceRecord(Base):
    """
    Stores evidence gathered for hypotheses during an investigation.
    """
    __tablename__ = "evidence_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    investigation_id = Column(String, ForeignKey("investigation_logs.investigation_id"), nullable=False)
    hypothesis = Column(String, nullable=False)       # The hypothesis being tested
    status = Column(String, nullable=False)           # "proven", "eliminated", "unresolved"
    evidence_type = Column(String, nullable=False)    # "feature_drift", "model_registry_diff", "git_commit"
    evidence_summary = Column(String, nullable=False) # A description of the evidence found
    evidence_data = Column(JSON, nullable=True)       # The raw data retrieved (e.g. stats, commit dict)

def init_db(db_url: str = DATABASE_URL):
    """
    Initializes the database, creating all tables if they do not exist.
    """
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
