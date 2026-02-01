# Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support
# Abstract
In recent years, Large Language Models (LLMs) have shown promise in assisting developers with code-related questions; however, LLMs carry the risk of generating unreliable answers, including hallucinations or answers that exceed their knowledge limits. To address this, Retrieval-Augmented Generation (RAG) has been proposed to reduce LLM unreliability. However, the design of effective RAG pipelines remains challenging due to numerous design choices. In this paper, we construct a retrieval corpus of over 3 million Java and Python-related Stack Overflow posts with accepted answers, and explore various RAG pipeline designs to answer developer questions, evaluating their effectiveness in generating accurate and reliable responses. Furthermore, we propose an adaptive thresholding approach that improves the generated responses when available knowledge is limited. More specifically, we (1) design and evaluate 7 different RAG pipelines and 63 pipeline variants to answer questions that have historically similar matches and (2) address new questions without any close prior matches by dynamically changing the similarity threshold during retrieval, thereby increasing the chance of finding partially relevant context and improving coverage when fixed thresholds are not sufficient. We find that an RAG pipeline combining hypothetical document embedding (HyDE) with full-answer retrieval maintains high answer quality while avoiding retrieval failures by using adaptive thresholding when no close matches are available. Furthermore, we apply our optimal RAG pipeline to multiple LLMs and compare their performance against their zero-shot baselines. Our findings show that our optimal RAG pipeline consistently outperforms zero-shot baselines across most evaluated LLMs, achieving higher scores for helpfulness, correctness, and detail.

# Experimental Workflow

![Experimental Workflow](images/Overall_Workflow-v2.PNG)


## Extract and Process Stack Overflow Dump

Please download the Stack Overflow data dump from [Archive.org Stack Exchange Data Dump](https://archive.org/details/stackexchange).

Specifically, extract the `Posts.xml` file and place it **in the same directory as this project (\datasets)**.

The `datasets` folder contains the following notebooks:

- **xmlProcessing_python_java_posts.ipynb**  
  This script extracts **Java** and **Python** posts from the `Posts.xml` file.  
  It pairs each **question** with its corresponding **accepted answer**, and performs **text cleaning** to prepare the data for downstream tasks.

- **Embedding.ipynb**  
  This notebook builds the **RAG knowledge dataset** by embedding the paired question-answer data.  
  The generated file is:  
  `CODE_POST_OVERALL-EMBEDDINGS_DATA_V3.pkl`  
  This dataset will be used in the following research questions (RQs).

- **Embedding_AnswerSentence.ipynb**  
  This script embeds the **answer content at the sentence level**, to support experiments that require **sentence-level content granularity**.

- **Embedding_Full_Answer.ipynb**  
  This script embeds the **full answer content**, to support experiments using **full-answer granularity**.

> **Note:**  
> Due to the large file size, we do not upload intermediate processed files. These files will be generated automatically when you run the pipeline.


## RQ1: Which retrieval approach configurations yield the highest response quality in LLM-generated answers?

This folder contains the code, synthetic data, and evaluation scripts for **RQ1**, which evaluates whether Retrieval-Augmented Generation (RAG) improves response generation over standard LLM prompting.

#### `synthetic_testing/`
This subfolder includes Jupyter notebooks for running different RAG pipelines and baselines on a **synthetic question set** derived from our knowledge base.

- **HB1.ipynb / HB2.ipynb / HYB.ipynb**  
  Run **HyDE-based RAG pipelines** with variations in retrieval target and granularity.
  
- **QB1.ipynb / QB2.ipynb / QB3.ipynb / QB4.ipynb**  
  Run **Question-based RAG pipelines**, using the input question directly for retrieval. Each file represents a different pipeline variant.
  
- **Synthetic_old_question.csv**  
  Contains the synthetic questions and their corresponding accepted answers, generated from the knowledge base for controlled testing.

- **synthetic_question_generation.ipynb**  
  Script to generate the synthetic questions automatically from historical Stack Overflow data.

#### `llm_judge_results/`
This subfolder stores the **evaluation results** from the LLM-as-a-Judge framework. Each answer is scored for **helpfulness, correctness, and level of detail** on a 1–10 scale.

---

### Other Scripts in `RQ1/`

- **llm_as_judge.ipynb**  
  Uses an LLM to evaluate generated answers. The LLM rates each response anonymously to reduce bias.

- **llm_score_calculation.ipynb**  
  Aggregates and analyzes the scores produced by `llm_as_judge.ipynb`. This includes statistical tests and group comparisons.

---

### Purpose

This folder enables replication of our **RQ1 experiments**:
- Compare zero-shot prompting to different RAG configurations.
- Evaluate the impact of HyDE-based retrieval versus question-based retrieval.
- Perform controlled testing using synthetic Stack Overflow questions with known ground truth answers.

For methodology details, please refer to Section 3 of our paper.

## RQ2: How well does adaptive HyDE retrieval perform on novel questions outside the training corpus?

This folder contains code, data, and results for **RQ2**:  
**"How well does adaptive HyDE retrieval perform on novel questions outside the training corpus?"**

The goal of this experiment is to test whether **HyDE-based retrieval (HB1 pipeline)** can effectively handle **unseen questions** by dynamically adjusting the **similarity threshold** during retrieval. This helps evaluate the robustness of HyDE when the test questions have no close matches in the existing knowledge base.

---

#### `unseen_HB1_results_decrease_thres/`
This folder stores **retrieval and generation results for HB1** at multiple retrieval thresholds (0.5 to 0.9).

- **HB1_0.5.csv – HB1_0.9.csv**  
  Generated answers for the unseen test set using different cosine similarity thresholds for retrieval.  
  Lower thresholds allow broader retrieval, while higher thresholds are more selective.

- **llm_as_judge_unseen_data.ipynb**  
  Runs LLM-as-a-Judge to evaluate the generated answers at each threshold.

- **unseen_data_llm_judge_scores.csv**  
  Contains LLM-as-a-Judge scores (helpfulness, correctness, detail) for all thresholds.

---

#### `unseen_data_create/`
This folder prepares the **unseen 2024 question set** used in the experiment.

- **SAMPLE_combined_LATEST_TESTSET_COMPLETED.csv**  
  Final combined dataset of unseen Java and Python questions for evaluation.

- **java_questions_2024_completed.json / python_questions_2024_completed.json**  
  Raw unseen question sets crawled from Stack Overflow, post-2024.

- **SOclawerOnline.ipynb**  
  Script to collect recent Stack Overflow questions online.

- **processMerge.ipynb**  
  Merges Java and Python unseen questions into a single dataset.

---

### Other Scripts in `RQ2/`

- **HB1.ipynb**  
  Runs the **HB1 pipeline** (HyDE-based retrieval with full answer retrieval) on the unseen data at the default threshold.

- **HB1_decrease_thres.csv**  
  Records retrieval statistics (e.g., number of retrieved documents) at each threshold level.

- **HB1_decrease_thres_llm_judge_results.csv**  
  Stores LLM-as-a-Judge evaluations for different threshold settings.

- **decrease_threshold.ipynb**  
  Automates experiments across multiple thresholds for HB1, enabling **adaptive retrieval testing**.

- **llm_as_judge_unseen_data_decrease_thres.ipynb**  
  Runs LLM-as-a-Judge specifically for the **decreasing threshold experiments**.

---

### Purpose

This folder supports replication of **RQ2 experiments**, including:

- Testing **HyDE-based retrieval (HB1)** on unseen questions from 2024.
- Evaluating the effect of **adaptive retrieval thresholds** on response quality.
- Using **LLM-as-a-Judge** to assess generated answers for **helpfulness, correctness, and detail**.

For detailed methodology, refer to **Section 3 of the paper**.


## RQ3: How does our proposed RAG pipeline perform across different LLMs?

This folder contains scripts and results for **RQ3**, which investigates:

**"Can our adaptive RAG pipelines generalize across different Large Language Models (LLMs)?"**

The goal of RQ3 is to evaluate whether the best-performing RAG pipelines from RQ1 and RQ2 (specifically **HB1 and RAG4**) still improve answer quality when applied to **different open-source LLMs**, beyond the models originally tested.

---

#### Baseline Results (Zero-shot prompting without retrieval)

- **baseline_granite.csv**  
- **baseline_llama.csv**  
- **baseline_mistral.csv**  
- **baseline_qwen.csv**
- **baseline_gpt4.csv**
- **baseline_deepseek.csv** 

These files store the **zero-shot generation results** for each model without any RAG support. They serve as baselines for comparison.

---

#### Dynamic RAG Results (Adaptive RAG4 pipeline)

- **dynamic_RAG4_granite_results.csv**  
- **dynamic_RAG4_llama_results.csv**  
- **dynamic_RAG4_mistral_results.csv**  
- **dynamic_RAG4_qwen_results.csv**
- **dynamic_RAG4_gpt4_results.csv**
- **dynamic_RAG4_deepseek_results.csv**   

These files contain the results of applying **RAG4 (HyDE-based retrieval with filtering)** to each LLM using a **dynamic threshold**.

---

#### HB1 Dynamic Threshold Application

- **apply_HB1_dynamic_thres_granite.ipynb**  
- **apply_HB1_dynamic_thres_llama.ipynb**  
- **apply_HB1_dynamic_thres_mistral.ipynb**  
- **apply_HB1_dynamic_thres_qwen.ipynb**
- **apply_HB1_dynamic_thres_gpt4.ipynb**
- **apply_HB1_dynamic_thres_deepseek.ipynb**   

Jupyter notebooks that run **HB1 pipeline (HyDE + full answer retrieval)** with adaptive thresholds for each LLM.

---

#### Zero-shot Scripts

- **zeroshot_Granite.ipynb**  
- **zeroshot_Llama.ipynb**  
- **zeroshot_Mistral.ipynb**  
- **zeroshot_Qwen3.ipynb**
- **zeroshot_gpt4.ipynb**
- **zeroshot_deepseek.ipynb**   

Run baseline zero-shot generation for each LLM without retrieval.

---

#### Other Files

- **RQ2_more_LLMs_applied_RAG4.csv**  
  Aggregated results of applying RAG4 to multiple LLMs as part of extended RQ2/RQ3 testing.

- **llm_as_judge_more_LLMs_applied_HB1.ipynb**  
  Runs **LLM-as-a-Judge** to evaluate the generated answers across all tested LLMs for HB1 and RAG4.

---

### Purpose

This folder enables replication of **RQ3 experiments**, including:

- Testing whether **HyDE-based retrieval (HB1)** and **dynamic RAG4 pipelines** generalize to other models like **Granite, LLaMA, Mistral, and Qwen3**.
- Comparing **zero-shot generation vs. RAG-enhanced generation** for each model.
- Using **LLM-as-a-Judge** to evaluate helpfulness, correctness, and detail across LLMs.

For full methodology details, refer to **Section 3 of the paper**.


## Extra

_**implication**_ folder contains supplemental analysis files to support the **Implications and Discussion** section of the paper.

- **HB1_vs_ZeroShot_TopCases.xlsx**  
  Contains selected examples for the **qualitative insights section**. This highlights representative cases where **HB1 significantly outperforms zero-shot prompting**.

- **random_sample_for_alignment_testing_54.xlsx**  
  Contains a **random sample of judged answers** used to manually verify the **alignment between LLM-as-a-Judge scores and human evaluation**.

These files are intended for deeper inspection of model behavior and evaluation consistency.
