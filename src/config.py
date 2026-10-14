from __future__ import annotations
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def _path_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:2b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
APPLICATION_DB_PATH = _path_env("APPLICATION_DB_PATH", "data/atmresolve.db")
CHECKPOINT_DB_PATH = _path_env("CHECKPOINT_DB_PATH", "data/checkpoints.db")
VECTOR_DB_PATH = _path_env("VECTOR_DB_PATH", "data/chroma")
ML_MODEL_PATH = _path_env("ML_MODEL_PATH", "artifacts/ml/intent_classifier.joblib")
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.35"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
