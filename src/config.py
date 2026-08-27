"""Configuration settings for the Aster & Row AI Support Agent."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DEFAULT_ORDERS_PATH = PROJECT_ROOT / "data" / "orders.json"
DEFAULT_KB_DIR = PROJECT_ROOT / "knowledge-base"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "evaluation"

ORDERS_DATA_PATH = Path(os.getenv("ORDERS_DATA_PATH", str(DEFAULT_ORDERS_PATH)))
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", str(DEFAULT_KB_DIR)))
EVALUATION_DIR = Path(os.getenv("EVALUATION_DIR", str(DEFAULT_EVAL_DIR)))

# Model & API settings
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Operational Constants
SNAPSHOT_AT: str = "2026-08-15T12:00:00Z"
ORDER_CANCELLATION_WINDOW_MINUTES: int = 30
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "yes")
