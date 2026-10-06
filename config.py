"""Central settings for the advisor bot.

Every tunable number and path lives here so the other files never hard-code
them. Secrets (the API key) are read from the .env file, never written here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read key=value pairs from .env into environment variables (if .env exists).
load_dotenv()

# --- Who the bot is for -----------------------------------------------------
UNIVERSITY = "The University of Texas at Austin"
MAJOR = "Chemical Engineering"

# --- Paths --------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
# Source documents. The subfolder a file sits in (Curriculum, Policies,
# Syllabi, ...) becomes its doc_type; files directly in here get "general".
RAW_DIR = PROJECT_DIR / "documents"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"  # vector database, built by ingest.py
COLLECTION_NAME = "advisor_docs"
EVAL_DIR = PROJECT_DIR / "eval"

# --- Chunking (Phase 1) ------------------------------------------------------
CHUNK_SIZE_WORDS = 350  # target chunk length (spec: about 300-400 words)
CHUNK_OVERLAP_WORDS = 50  # words repeated between neighboring chunks

# --- Embeddings and retrieval (Phase 2) --------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models search better when queries (not documents) start with this text.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 6  # how many chunks to retrieve per question

# --- Answering (Phase 3) -----------------------------------------------------
# If the best chunk's similarity score is below this, skip the LLM and say the
# documents don't cover the question. Tune this after Phase 2.
MIN_SCORE = 0.5
HISTORY_TURNS = 4  # past chat turns sent along for follow-up questions
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
MAX_ANSWER_TOKENS = 1024


def get_api_key() -> str:
    """Return the Anthropic API key, or stop with a clear message if it's missing.

    Called only when the LLM is actually needed, so ingestion and search work
    without a key.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and paste "
            "your key after ANTHROPIC_API_KEY="
        )
    return key
