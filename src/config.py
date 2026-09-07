import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ml_failure_investigator.db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Ensure data directories exist
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
