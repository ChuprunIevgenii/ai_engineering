"""Shared configuration (ready — read, don't edit).

Importing this module only sets a few environment flags that quiet the
embedding libraries. It never loads data or models.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
for _name in ("sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# --- Models ---------------------------------------------------------------
MODEL = "openrouter:google/gemini-3.5-flash-lite"
JUDGE_MODEL = "openrouter:openai/gpt-6.1-terra"
ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Retrieval ------------------------------------------------------------
CHUNK_CHARS = 400   # policy page text is cut into ~400-char chunks, each keeps its page number
TOP_K = 8           # chunks returned by one search call
RRF_K = 60          # reciprocal rank fusion constant
RRF_DEPTH = 50      # how deep each ranker (BM25, dense) contributes to RRF

# --- Personas -------------------------------------------------------------
DEFAULT_STUDENT = "S-1041"   # the logged-in student when a question does not say otherwise

# --- Paths ----------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
HW_DIR = PKG_DIR.parent
DATA_DIR = HW_DIR / "data"
POLICY_META = DATA_DIR / "policies" / "meta.json"
PAGES_DIR = DATA_DIR / "pages"
STUDENTS_FILE = DATA_DIR / "api" / "students.json"
QUESTIONS_FILE = DATA_DIR / "questions.json"
CACHE_DIR = HW_DIR / ".cache"
RESULTS_DIR = HW_DIR / "results"
