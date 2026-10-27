"""
Configuration management for the Joint Input Robustness experiment.
Handles environment variables, paths, model names, budgets, and random seeds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Load .env file from project root first, then workspace root
if (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env")
elif (WORKSPACE_ROOT / ".env").exists():
    load_dotenv(WORKSPACE_ROOT / ".env")
else:
    load_dotenv()

# Standard subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = RESULTS_DIR / "cache"

# Ensure runtime directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip().strip('"').strip("'")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest").strip()

# Safety & Budget Parameters
MAX_API_BUDGET = int(os.getenv("MAX_API_BUDGET", "150"))
MAX_TOKEN_BUDGET = int(os.getenv("MAX_TOKEN_BUDGET", "500000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "5.0"))
RATE_LIMIT_PACING_SECONDS = float(os.getenv("RATE_LIMIT_PACING_SECONDS", "3.0"))

# Experiment Hyperparameters
RANDOM_SEED = 42
DEFAULT_TOP_K = 3
TASK_LIMIT = int(os.getenv("TASK_LIMIT", "0"))  # 0 means all tasks

def get_masked_api_key() -> str:
    """Return a masked representation of the API key for safe logging."""
    if not GEMINI_API_KEY:
        return "NOT_CONFIGURED"
    if len(GEMINI_API_KEY) <= 8:
        return "****"
    return f"{GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}"

def is_api_configured() -> bool:
    """Check if a valid API key is present."""
    return bool(GEMINI_API_KEY and len(GEMINI_API_KEY) > 10)
