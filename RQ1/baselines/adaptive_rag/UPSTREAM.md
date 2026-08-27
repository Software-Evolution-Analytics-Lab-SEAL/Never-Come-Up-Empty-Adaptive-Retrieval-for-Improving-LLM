# Upstream Adaptive-RAG

The wrapper scripts in this directory (`run_retrieval.py`, `build_predict_json.py`,
`predict_on_n666.sh`, `train_classifier.sh`, `merge_zeroshot_path_a.py`) drive the
official Adaptive-RAG codebase.

To reproduce, clone the upstream repo into `./upstream/`:

```
git clone https://github.com/starsuzi/Adaptive-RAG upstream
```

Then follow `../README.md` for the classifier training and
retrieval-prediction steps. Our wrappers expect the upstream directory layout as
of commit hash reported in that document.
