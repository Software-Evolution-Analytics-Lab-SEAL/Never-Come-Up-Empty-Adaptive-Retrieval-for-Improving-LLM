# Datasets — knowledge base construction and preprocessing

The full knowledge base and embeddings are too large to ship (5.3 GB and 24 GB
respectively). This folder contains the crawler and offline embedding code so
you can rebuild both from the public Stack Overflow data dump.

## Contents

| File                              | Purpose |
|-----------------------------------|---------|
| `SOclawerOnline.ipynb`            | Online SO crawler (2025 posts for the Unseen SO test set) |
| `Embedding_Offline_code.py`       | Sentence-tokenize + T5-paraphrase + SentenceTransformer-encode accepted answers |
| `Embedding_Offline_code_completeAnswer.ipynb` | Full-answer embedding pipeline |
| `Embedding_Offline_AnswerSentence.ipynb` | Sentence-level embedding pipeline |

## Knowledge base construction

Source: [Stack Overflow data dump](https://archive.org/details/stackexchange),
snapshot covering January 2008 – December 2024 (~90 GB).

Filtering:

1. Keep only posts tagged `[java]` or `[python]`.
2. Keep only questions with an accepted answer.
3. Exclude Stack Overflow duplicates via the `DuplicateOfId` field
   (retain only the original in each duplicate group).
4. Remove questions with identical titles after normalization
   (lowercasing + whitespace trimming).

Result: **3,428,217** unique Java/Python posts with accepted answers.

Pre-processing (per accepted answer):

1. Strip HTML tags and Markdown syntax to produce plain text.
2. Sentence-tokenize (NLTK Punkt) for sentence-level retrieval variants.
3. Encode with `sentence-transformers/all-mpnet-base-v2` at both full-answer
   and sentence granularity.

Each knowledge base entry consists of the question title and its accepted
answer only; the question body is not indexed. Developer queries in practice
are short and title-like, and question bodies often carry stack traces, code
dumps, and environment details that add retrieval noise.

## Data-splitting protocol

- **Knowledge base:** all posts with `creation_date < 2025-01-01`.
- **Synthetic Question Set:** 666 questions sampled from the KB and paraphrased
  with GPT-4o. Not used as ground truth; retrieval targets are known to lie in
  the KB, so this set benchmarks pipeline design, not real-world generalization.
- **Unseen Stack Overflow 2025:** 3,376 questions crawled with
  `creation_date ≥ 2025-01-01`. No `post_id` overlap with the KB; all titles
  unique within the set; every question has an accepted answer.
- **Unseen GitHub Discussions 2025:** 1,732 threads from the top-100 Java and
  top-100 Python public repositories by stars, `creation_date ≥ 2025-01-01`,
  marked-answer ≥ 100 characters, filtered for question signals, majority class
  undersampled for language balance.

All Unseen 2025 questions post-date every evaluated LLM's training cutoff.

## Reproduce the KB

```bash
# 1. Download the SO data dump (Posts.xml, ~90 GB)
wget https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z

# 2. Filter to Java/Python + accepted-answer posts + de-dup by DuplicateOfId + title
python filter_kb.py --posts Posts.xml --out stack_overflow_python_java_kb.json

# 3. Embed accepted answers (~14 h on an A100 80 GB)
python Embedding_Offline_code.py --input stack_overflow_python_java_kb.json \
                                 --out complete_answer_embeddings.pt \
                                 --model sentence-transformers/all-mpnet-base-v2
```
