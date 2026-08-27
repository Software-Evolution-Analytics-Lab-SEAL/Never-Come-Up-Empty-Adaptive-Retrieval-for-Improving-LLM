"""
Configuration for RQ1 (optimal pipeline on the 666-question synthetic set).

Paths default to a layout relative to this script; override with environment
variables (KB_ROOT, OPENROUTER_KEY_FILE) to point at your KB and API key.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))           # .../RQ1/scripts
RQ_ROOT      = os.path.dirname(PROJECT_ROOT)                         # .../RQ1
DATA_DIR     = os.path.join(RQ_ROOT, "data")
RETRIEVAL_DIR = os.path.join(DATA_DIR, "retrieval_results")
LOGS_DIR     = os.path.join(RQ_ROOT, "logs")

# Knowledge base + embeddings live outside the package (too large to ship).
# Point KB_ROOT at your local build (see ../../datasets/README.md).
KB_ROOT = os.environ.get("KB_ROOT", os.path.expanduser("~/Adaptive_HyDe_RAG"))
OVO_PATH              = os.path.join(KB_ROOT, "CODE_POST_OVERALL-EMBEDDINGS_DATA_V3.pkl")
ANSWER_EMB_PATH       = os.path.join(KB_ROOT, "CODE_POST_OVERALL-EMBEDDINGS_DATA_V3_complete_answer_embeddings.pkl")
ANSWERS_CONTEXT_PATH  = os.path.join(KB_ROOT, "answers_context.pkl")
ALL_SENT_EMB_PATH     = os.path.join(KB_ROOT, "AnswerEmbedding/all_embeddings.pt")
ALL_SENT_TEXT_PATH    = os.path.join(KB_ROOT, "AnswerEmbedding/all_sentences.npy")
ALL_FULL_ANS_EMB_PATH = os.path.join(KB_ROOT, "complete_answer_embeddings.pt")

# Synthetic Question Set (shipped in ../data/)
TEST_SET_PATH   = os.path.join(DATA_DIR, "synthetic_questions_n666.csv")
HYDE_CACHE_PATH = os.path.join(DATA_DIR, "hyde_cache.json")

# OpenRouter API (used for GPT-4o HyDE + LLM-as-a-Judge)
OPENROUTER_API_KEY_PATH = os.environ.get(
    "OPENROUTER_KEY_FILE",
    os.path.join(RQ_ROOT, "..", "openrouter_api.txt"),
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GPT4O_MODEL         = "openai/gpt-4o"

# Encoder (must match the one used to build cached embeddings)
ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"
ENCODER_DIM   = 768

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOP_K      = 10          # max retrieved items per query
N_TARGET   = 666         # synthetic-set size (99% confidence, 5% margin of error)
SEED       = 42

# Pipeline definitions
# query:       "question" | "hyde" | "question_then_hyde" (HYB stage-1 → stage-2)
# retrieval:   "direct"   (encode → search granularity pool)
#              "indirect" (stage-1 = question pool, stage-2 = re-rank within matched posts)
# granularity: "sentence" | "full_answer"
PIPELINES = {
    "QB1": {"query": "question",           "retrieval": "direct",   "granularity": "sentence"},
    "QB2": {"query": "question",           "retrieval": "direct",   "granularity": "full_answer"},
    "QB3": {"query": "question",           "retrieval": "indirect", "granularity": "sentence"},
    "QB4": {"query": "question",           "retrieval": "indirect", "granularity": "full_answer"},
    "HB1": {"query": "hyde",               "retrieval": "direct",   "granularity": "full_answer"},
    "HB2": {"query": "hyde",               "retrieval": "direct",   "granularity": "sentence"},
    "HYB": {"query": "question_then_hyde", "retrieval": "indirect", "granularity": "full_answer"},
}


def get_openrouter_api_key():
    with open(OPENROUTER_API_KEY_PATH, "r") as f:
        return f.read().strip()


for d in (DATA_DIR, RETRIEVAL_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)
