"""
Shared LLM client code for RQ2 generation + judging.

MODELS registry (endpoint, id, fallback_ep, fallback_id):
    - openrouter endpoint uses OpenAI SDK against OPENROUTER_BASE_URL
    - ollama endpoint uses POST to local Ollama /api/chat

Optimal-pipeline-per-model map lives in config.OPTIMAL_PIPELINE (set in RQ2/config.py).
"""
import os
import sys
import time
import requests
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config


# Local models are served from an in-house Ollama instance on the A100 host.
# Only the two frontier LLMs (GPT-4.1 and DeepSeek-r1-70B) are accessed via OpenRouter.
MODELS = {
    "llama-3.1-8b":    ("ollama",     "llama3.1:8b-instruct-fp16",   None, None),
    "mistral-7b":      ("ollama",     "mistral:7b-instruct-v0.3",    None, None),
    "granite-3.1-8b":  ("ollama",     "granite3.1:8b-instruct",      None, None),
    "qwen3-8b":        ("ollama",     "qwen3:8b",                    None, None),
    "deepseek-r1-70b": ("openrouter", "deepseek/deepseek-r1",        None, None),
    "gpt-4.1":         ("openrouter", "openai/gpt-4.1",              None, None),
}


_or_client = None
def or_client():
    global _or_client
    if _or_client is None:
        api_key = config.get_openrouter_api_key()
        _or_client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
    return _or_client


def call_openrouter(model_id, system, user, max_tokens=512, temperature=0.7, retries=3):
    for attempt in range(retries):
        try:
            r = or_client().chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR] {e}"


def call_ollama(model_id, system, user, max_tokens=512, temperature=0.7, retries=3, timeout=1800):
    payload = {
        "model": model_id, "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    for attempt in range(retries):
        try:
            r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR] {e}"


def call_model(model_slug, system, user, max_tokens=512, temperature=0.7):
    ep, mid, fep, fid = MODELS[model_slug]
    if ep == "openrouter":
        out = call_openrouter(mid, system, user, max_tokens=max_tokens, temperature=temperature)
    else:
        out = call_ollama(mid, system, user, max_tokens=max_tokens, temperature=temperature)
    # Fallback on error
    if isinstance(out, str) and out.startswith("[ERROR]") and fep is not None:
        if fep == "openrouter":
            out = call_openrouter(fid, system, user, max_tokens=max_tokens, temperature=temperature)
        else:
            out = call_ollama(fid, system, user, max_tokens=max_tokens, temperature=temperature)
    return out
