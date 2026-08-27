"""
Configuration for RQ2 (adaptive HB1 on the Unseen Stack Overflow 2025 set).

Self-contained — no cross-folder imports. Same shared constants as RQ1
(encoder, thresholds, judge, PIPELINES), plus RQ2-specific paths.

Override KB_ROOT and OPENROUTER_KEY_FILE via environment variables.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))            # .../RQ2/scripts
RQ_ROOT      = os.path.dirname(PROJECT_ROOT)                          # .../RQ2
DATA_DIR     = os.path.join(RQ_ROOT, "data")
RETRIEVAL_DIR = os.path.join(DATA_DIR, "retrieval")
GENERATION_DIR = os.path.join(DATA_DIR, "generation")
EVAL_DIR     = os.path.join(DATA_DIR, "evaluation")
LOGS_DIR     = os.path.join(RQ_ROOT, "logs")

# Knowledge base + embeddings (too large to ship — build via ../../datasets/)
KB_ROOT = os.environ.get("KB_ROOT", os.path.expanduser("~/Adaptive_HyDe_RAG"))
OVO_PATH              = os.path.join(KB_ROOT, "CODE_POST_OVERALL-EMBEDDINGS_DATA_V3.pkl")
ANSWERS_CONTEXT_PATH  = os.path.join(KB_ROOT, "answers_context.pkl")
ALL_FULL_ANS_EMB_PATH = os.path.join(KB_ROOT, "complete_answer_embeddings.pt")
ALL_SENT_TEXT_PATH    = os.path.join(KB_ROOT, "AnswerEmbedding/all_sentences.npy")
ALL_SENT_EMB_PATH     = os.path.join(KB_ROOT, "AnswerEmbedding/all_embeddings.pt")

# Test sets shipped in ../data/
UNSEEN_CSV        = os.path.join(DATA_DIR, "unseen_2025.csv")
UNSEEN_LEGACY_CSV = os.path.join(DATA_DIR, "unseen_5510.csv")

# OpenRouter API
OPENROUTER_API_KEY_PATH = os.environ.get(
    "OPENROUTER_KEY_FILE",
    os.path.join(RQ_ROOT, "..", "openrouter_api.txt"),
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GPT4O_MODEL         = "openai/gpt-4o"

# Encoder — same model as RQ1
ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_MODEL   = ENCODER_MODEL             # alias for compatibility with legacy imports
ENCODER_DIM   = 768

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
THRESHOLDS    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ADAPTIVE_THRS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
TOP_K = 10
SEED  = 42

# Same pipeline table as RQ1 (kept here for self-containment)
PIPELINES = {
    "QB1": {"query": "question",           "retrieval": "direct",   "granularity": "sentence"},
    "QB2": {"query": "question",           "retrieval": "direct",   "granularity": "full_answer"},
    "QB3": {"query": "question",           "retrieval": "indirect", "granularity": "sentence"},
    "QB4": {"query": "question",           "retrieval": "indirect", "granularity": "full_answer"},
    "HB1": {"query": "hyde",               "retrieval": "direct",   "granularity": "full_answer"},
    "HB2": {"query": "hyde",               "retrieval": "direct",   "granularity": "sentence"},
    "HYB": {"query": "question_then_hyde", "retrieval": "indirect", "granularity": "full_answer"},
}

# Optimal pipeline per model, from RQ1 (adaptive-avg winners)
OPTIMAL_PIPELINE = {
    "llama-3.1-8b":    "HB1",
    "gpt-4.1":         "HB1",
    "qwen3-8b":        "HB1",
    "mistral-7b":      "HYB",
    "deepseek-r1-70b": "HB1",
    "granite-3.1-8b":  "HB1",
}


def get_openrouter_api_key():
    with open(OPENROUTER_API_KEY_PATH, "r") as f:
        return f.read().strip()


for d in (DATA_DIR, RETRIEVAL_DIR, GENERATION_DIR, EVAL_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)
