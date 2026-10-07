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
# Files under RAW_DIR to leave out of the database. These PDFs are graphics
# whose text comes out scrambled; each has a hand-typed "(transcribed).md"
# version next to it that is used instead.
SKIP_FILES = [
    "Curriculum/2026-2028 CHE Suggested Arrangement of Courses.pdf",
    "Curriculum/CHE Prerequisite Flowchart.pdf",
    "Policies/Latest You Can Take Courses With Prereqs.pdf",
]
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"  # vector database, built by ingest.py
COLLECTION_NAME = "advisor_docs"
EVAL_DIR = PROJECT_DIR / "eval"

# --- Chunking (Phase 1) ------------------------------------------------------
CHUNK_SIZE_WORDS = 350  # target chunk length (spec: about 300-400 words)
CHUNK_OVERLAP_WORDS = 50  # words repeated between neighboring chunks
MIN_CHUNK_WORDS = 60  # chunks shorter than this get merged with a neighbor

# --- Embeddings and retrieval (Phase 2) --------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models search better when queries (not documents) start with this text.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 6  # how many chunks to retrieve per question
CANDIDATE_MULTIPLIER = 5  # each search method proposes TOP_K x this many chunks
RRF_K = 60  # rank-fusion constant; the standard value, rarely needs changing
KEYWORD_WEIGHT = 1.0  # how much the keyword ranking counts vs. the meaning ranking (1.0)
MAX_PER_SOURCE = 2  # at most this many chunks from any one file per answer
MAX_SCORE_GAP = 0.12  # drop chunks this much less similar than the best match
# Per-document-type limits on chunks per answer. There are 50+ syllabi with
# lots of repeated text, so they could otherwise fill every slot.
MAX_PER_DOC_TYPE = {"syllabi": 3}

# --- Answering (Phase 3) -----------------------------------------------------
# If the best chunk's similarity score is below this, skip the LLM and say the
# documents don't cover the question. In testing, real advising questions
# scored 0.68+ and off-topic ones (pizza, weather) 0.43-0.56.
MIN_SCORE = 0.6
HISTORY_TURNS = 4  # past chat turns sent along for follow-up questions
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5")
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
