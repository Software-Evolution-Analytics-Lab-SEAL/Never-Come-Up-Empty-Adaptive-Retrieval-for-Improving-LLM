# Upstream Self-RAG

The wrapper scripts in this directory drive the official Self-RAG retrieval
pipeline. To reproduce, clone the upstream repo into `./upstream/`:

```
git clone https://github.com/AkariAsai/self-rag upstream
```

Then use `encode_so_kb.sh` to build the Stack Overflow index and `run_retrieval.py`
to run retrieval over the RQ1 synthetic set. See `../README.md`
for exact commit hashes and reproduction details.
