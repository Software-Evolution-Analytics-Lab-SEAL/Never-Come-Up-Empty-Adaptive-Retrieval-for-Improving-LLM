"""
Knowledge-base access — loaders for the KB pickles and the encoder wrapper.

All heavy artifacts (OVO data, full-answer pool, sentence pool, encoder) are
loaded once per process and cached. Cosine similarity handles fp16/fp32
mismatch so query (fp32) can be compared against fp16 sentence pool.
"""
import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

import config


_CACHE = {}


def _set_cuda_visible_devices(gpu_id=None):
    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# ---------------------------------------------------------------------------
# OVO_data — the per-post object array
# ---------------------------------------------------------------------------
def load_ovo_data():
    """Load CODE_POST_OVERALL-EMBEDDINGS_DATA_V3.pkl as a list of dicts.

    Each dict has keys:
        raw_question (str)
        question_embedding (torch.Tensor on CPU)
        raw_accepted_answer (str)
        answer_sentences (list[str])
        knowledge (?)
    """
    if "ovo" in _CACHE:
        return _CACHE["ovo"]
    print(f"[kb] Loading OVO_data from {config.OVO_PATH} ...")
    with open(config.OVO_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"[kb] OVO_data loaded: {len(data)} posts")
    _CACHE["ovo"] = data
    return data


def question_embeddings_matrix():
    """Stack all per-post question_embedding tensors into one (N, 768) matrix on CPU.

    Used by indirect-retrieval pipelines (QB3, QB4, HYB) for stage-1 search.
    """
    if "q_matrix" in _CACHE:
        return _CACHE["q_matrix"]
    ovo = load_ovo_data()
    print(f"[kb] Stacking {len(ovo)} question embeddings ...")
    q_emb = torch.stack([p["question_embedding"] for p in ovo]).cpu()
    print(f"[kb] question_matrix shape: {tuple(q_emb.shape)}")
    _CACHE["q_matrix"] = q_emb
    return q_emb


def raw_questions_list():
    if "raw_q" in _CACHE:
        return _CACHE["raw_q"]
    ovo = load_ovo_data()
    _CACHE["raw_q"] = [p["raw_question"] for p in ovo]
    return _CACHE["raw_q"]


def raw_accepted_answers_list():
    if "raw_a" in _CACHE:
        return _CACHE["raw_a"]
    ovo = load_ovo_data()
    _CACHE["raw_a"] = [p["raw_accepted_answer"] for p in ovo]
    return _CACHE["raw_a"]


def answer_sentences_list():
    if "ans_sents" in _CACHE:
        return _CACHE["ans_sents"]
    ovo = load_ovo_data()
    _CACHE["ans_sents"] = [p["answer_sentences"] for p in ovo]
    return _CACHE["ans_sents"]


# ---------------------------------------------------------------------------
# Full-answer embeddings (for QB2 / HB1 direct full-answer search)
# ---------------------------------------------------------------------------
def full_answer_pool():
    """Returns (texts: list[str], embeddings: torch.FloatTensor (N, 768)).

    Source: complete_answer_embeddings.pkl  +  answers_context.pkl
    """
    if "fa_pool" in _CACHE:
        return _CACHE["fa_pool"]
    print(f"[kb] Loading full-answer text from {config.ANSWERS_CONTEXT_PATH} ...")
    with open(config.ANSWERS_CONTEXT_PATH, "rb") as f:
        ans_text = pickle.load(f)
    print(f"[kb] Loading full-answer embeddings from {config.ALL_FULL_ANS_EMB_PATH} ...")
    ans_emb = torch.load(config.ALL_FULL_ANS_EMB_PATH, map_location="cpu", weights_only=False)
    if isinstance(ans_emb, list):
        ans_emb = torch.stack(ans_emb).cpu()
    elif hasattr(ans_emb, "cpu"):
        ans_emb = ans_emb.cpu()
    print(f"[kb] full_answer pool: {len(ans_text)} texts, embeddings shape {tuple(ans_emb.shape)}, dtype {ans_emb.dtype}")
    if len(ans_text) != ans_emb.shape[0]:
        raise RuntimeError(f"full_answer pool size mismatch: {len(ans_text)} texts vs {ans_emb.shape[0]} embeddings")
    _CACHE["fa_pool"] = (ans_text, ans_emb)
    return _CACHE["fa_pool"]


# ---------------------------------------------------------------------------
# Sentence embeddings (for QB1 / HB2 direct sentence search)
# ---------------------------------------------------------------------------
def sentence_pool():
    """Returns (texts: list[str], embeddings: torch.FloatTensor (S, 768)).

    Source: AnswerEmbedding/all_sentences.npy  +  AnswerEmbedding/all_embeddings.pt
    """
    if "sent_pool" in _CACHE:
        return _CACHE["sent_pool"]
    print(f"[kb] Loading sentences from {config.ALL_SENT_TEXT_PATH} ...")
    sent_text = np.load(config.ALL_SENT_TEXT_PATH, allow_pickle=True)
    sent_text = sent_text.tolist() if hasattr(sent_text, "tolist") else list(sent_text)
    print(f"[kb] Loading sentence embeddings from {config.ALL_SENT_EMB_PATH} ...")
    sent_emb = torch.load(config.ALL_SENT_EMB_PATH, map_location="cpu", weights_only=False)
    if isinstance(sent_emb, list):
        sent_emb = torch.stack(sent_emb).cpu()
    elif hasattr(sent_emb, "cpu"):
        sent_emb = sent_emb.cpu()
    # NOTE: sentence pool is fp16 on disk (~24 GB). Keep it fp16 to fit in GPU
    # memory; the query embedding will be cast to fp16 at cosine time (see
    # kb.cos_sim). This is the same behavior as torch's mixed
    # precision and produces results within 1e-3 of fp32.
    print(f"[kb] sentence pool: {len(sent_text)} sentences, embeddings shape {tuple(sent_emb.shape)}, dtype {sent_emb.dtype}")
    if len(sent_text) != sent_emb.shape[0]:
        raise RuntimeError(f"sentence pool size mismatch: {len(sent_text)} texts vs {sent_emb.shape[0]} embeddings")
    _CACHE["sent_pool"] = (sent_text, sent_emb)
    return _CACHE["sent_pool"]


def free_pool(name):
    """Free a cached pool to reclaim RAM."""
    _CACHE.pop(name, None)


# ---------------------------------------------------------------------------
# Sentence -> post_idx mapping (rebuilt from OVO_data['answer_sentences'])
# ---------------------------------------------------------------------------
_SENT_MAP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "sent_to_post_map.pkl")


def sent_to_post_map():
    """Return a numpy int32 array where entry [i] is the post_idx (into OVO)
    that sentence i in the sentence pool belongs to.

    Built by concatenating OVO_data[i]['answer_sentences'] in order, which is
    how the sentence pool was originally constructed.
    """
    if "sent_to_post" in _CACHE:
        return _CACHE["sent_to_post"]
    if os.path.exists(_SENT_MAP_CACHE):
        print(f"[kb] Loading cached sent_to_post_map from {_SENT_MAP_CACHE}")
        with open(_SENT_MAP_CACHE, "rb") as f:
            m = pickle.load(f)
        _CACHE["sent_to_post"] = m
        return m

    import numpy as _np
    ovo = load_ovo_data()
    print("[kb] Building sentence -> post_idx map from OVO ...")
    mapping = []
    for i, p in enumerate(ovo):
        mapping.extend([i] * len(p["answer_sentences"]))
    mapping = _np.array(mapping, dtype=_np.int32)
    print(f"[kb]   {len(mapping)} sentences mapped")
    os.makedirs(os.path.dirname(_SENT_MAP_CACHE), exist_ok=True)
    with open(_SENT_MAP_CACHE, "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    _CACHE["sent_to_post"] = mapping
    return mapping

_ENCODER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[kb] Loading encoder {config.ENCODER_MODEL} on {device} ...")
        _ENCODER = SentenceTransformer(config.ENCODER_MODEL).to(device)
    return _ENCODER


def encode_query(text):
    """Encode one or more query strings -> torch tensor on the encoder's device."""
    enc = get_encoder()
    if isinstance(text, str):
        return enc.encode(text, convert_to_tensor=True)
    return enc.encode(list(text), convert_to_tensor=True)


def cos_sim(a, b):
    """Cosine similarity. Returns shape (len(a), len(b)) tensor.

    Uses sentence_transformers.util.pytorch_cos_sim with one safety addition:
    if `a` and `b` have different dtypes (e.g. query=fp32 vs pool=fp16), cast `a`
    to match `b` so the matmul does not crash. The pool is the larger tensor so
    we never want to cast it up.
    """
    if a.dtype != b.dtype:
        a = a.to(b.dtype)
    return util.pytorch_cos_sim(a, b)
