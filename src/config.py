"""Configuration settings for the Resume Screening & Job Matching System."""
import os
from pathlib import Path

# Enable offline mode for Hugging Face to avoid latency and network timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "sample_data"
RESUMES_DIR = DATA_DIR / "resumes"
JOBS_DIR = DATA_DIR / "jobs"
MODELS_CACHE_DIR = BASE_DIR / ".cache"

# Embedding Model Configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# TF-IDF Configuration
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 5000
TFIDF_SUBLINEAR_TF = True

# Hybrid Matching Default Weights
# Semantic Similarity (SBERT) + Lexical Similarity (TF-IDF) + Hard Skill Overlap + Section Coverage
DEFAULT_WEIGHTS = {
    "semantic": 0.45,
    "tfidf": 0.25,
    "skills": 0.20,
    "experience": 0.10,
}

# Score Categorization Thresholds
SCORE_THRESHOLDS = {
    "strong_match": 0.75,
    "moderate_match": 0.55,
    "weak_match": 0.35,
}
