# QIME

**Ontology-grounded, question-based interpretable embeddings for biomedical text.**

This repository accompanies [*Asking the Right Questions: Ontology-Grounded Interpretable Embeddings for Biomedical Text*](https://arxiv.org/abs/2603.01690). The method names below follow paper version **v3**.

## Main method and comparison variants

The main **QIME** method combines:

1. **Ontology-Grounded Question Generation (OGQG):** discover discriminative biomedical yes/no questions using corpus clusters and UMLS grounding.
2. **Training-free embedding construction:** encode documents and questions with a pretrained encoder, compute their cosine similarities, and select question dimensions using **Maximal Marginal Relevance (MMR)**. Selected dimensions are set to 1; all others are set to 0.

The main configuration in the paper (§3.3 and §4.1) uses **8,855 questions**, **k = 256**, and **MMR λ = 0.7**. The implementation uses `abhinand/MedEmbed-large-v0.1` as the document/question encoder. The paper reports Qwen3-30B for question generation; this is separate from the embedding encoder.

| Name | Embedding construction | Repository entry point | Role in paper v3 |
| --- | --- | --- | --- |
| **QIME (TF+MMR)** | Cosine relevance + MMR selection; sparse binary coordinates | `TF/code/mmr_topk_model.py` | Main method |
| **QIME w/o MMR** | Top-k cosine relevance; sparse binary coordinates | `TF/code/test_model.py` | MMR ablation |
| **QIME w/o ontology** | TF+MMR with an ungrounded question bank | `eval_mteb_mmr.py` with `linear_questions.json` | Ontology ablation |
| **QIME_CLS** | Trained per-question classification heads on a shared encoder | `framework/imec_model.py`, `demo.py` | Classifier comparison (§4.3, Appendix C) |
| Cosine-valued sparse representation | Top-k selection with cosine-valued coordinates | `TF/code/test_model_cos_mmr.py` | Additional implementation variant |

**Start with the TF+MMR quick start below to use the paper's main method.** The root `demo.py` and `framework/run_qime_pipeline.py` use the classifier branch, **QIME_CLS**. Existing checkpoint paths such as `checkpoints/QIME/` retain their original names.

Training-free refers to embedding construction using a fixed question bank and a pretrained encoder. Generating a new question bank still requires the OGQG pipeline and an LLM. With the supplied bank, inference needs no per-question classifier training, QA supervision, or test-time LLM calls. Each active coordinate represents similarity-based semantic activation of its question, rather than a calibrated yes/no answer probability.

## Installation

Clone the repository and create the supplied environment:

```bash
git clone https://github.com/L1nzh/QIME.git
cd QIME
conda env create -f environment.yml
conda activate qime
```

The supplied environment targets Linux/CUDA and includes dependencies for question generation, classifier training, and evaluation. A CUDA GPU is recommended; the TF encoder also selects CPU when CUDA is unavailable. First use downloads the pretrained encoder from Hugging Face.

An LLM service/API or local generation backend and UMLS resources are needed when generating new questions. They are not needed for TF+MMR inference with `TF/data/questions.json`.

## Quick start: QIME (main method, TF+MMR)

Run this example **from the repository root** in the configured environment:

```python
import sys
import numpy as np

sys.path.insert(0, "TF/code")
from mmr_topk_model import EncoderModel

model = EncoderModel(
    topk=256,
    mmr_diversity=0.7,
    que_path="TF/data/questions.json",
)

texts = [
    "Patient presents with severe chest pain.",
    "Treatment involves daily insulin injections.",
]
embeddings = model.encode(texts, batch_size=32)

# The supplied bank has 8,855 questions, each defining one fixed coordinate.
print("Embedding shape:", embeddings.shape)  # (2, 8855)
print("Active dimensions per text:", embeddings.sum(axis=1))  # 256 each

# Binary coordinates have equal values. Inspect all selected questions;
# their coordinate indices do not represent a relevance ranking.
for text, embedding in zip(texts, embeddings):
    print("\nText:", text)
    for index in np.flatnonzero(embedding):
        print(f"[{index}] {model.questions[index]}")
```

The returned NumPy array has shape `[num_texts, num_questions]`. It is sparse in its values but stored as a dense array. With `k <= num_questions`, exactly `k` distinct dimensions are active per text. The vector dimension is the size of the question bank, **not** `k`.

`mmr_diversity` is the code's name for the paper's **λ**, the weight on document relevance:

```text
MMR score = λ × cosine(document, question)
            - (1 - λ) × redundancy with already selected questions
```

Increasing λ places more weight on relevance; decreasing it places more weight on diversity. Set `topk=256` explicitly: the encoder constructor's default is 32. The released code clamps the redundancy penalty to be non-negative; consult `mmr_indices()` for that implementation detail relative to the equation in §3.3.

## Evaluate QIME (main method)

Run the MMR evaluation script **from `TF/scripts/`** because its imports and default paths are relative to that directory:

```bash
cd TF/scripts
python eval_mteb_mmr.py --topks 256 --questions-path ../data/questions.json --gpus 0
```

This runner imports `mmr_topk_model.EncoderModel` and uses its default `mmr_diversity=0.7`. It has no `--mmr-diversity` command-line flag. Use `--topks 256` for the paper's main sparsity setting; `--topks 128 256` runs a sparsity comparison. `--gpus` distributes top-k configurations across GPUs; it does not split a single top-k configuration across devices.

Tasks are loaded from `TF/dataset/datasets.yaml`, and results are written under `TF/results/`. The supplied task list also includes SciFact and ArguAna, beyond the biomedical tasks in the paper's main tables. Record the actual question bank, task configuration, model/dependency versions, and output representation when comparing results with the paper. These commands select the main implementation and hyperparameters; they do not establish numerical reproduction of the reported tables.

### Ablations and additional TF variants

Run these commands from `TF/scripts/`. The two top-k runners (`eval_mteb.py` and `eval_mteb_ablation.py`) additionally require `TF/dataset/train_data/labeled_cor_ids.json`: `test_model.py` loads it unconditionally, even when encoding text directly. That file is not included in this checkout, so those two commands require the missing artifact or a code fix before they can run. The main TF+MMR encoder does not have this dependency.

```bash
# QIME w/o MMR: same grounded bank, top-k binary activation
python eval_mteb.py --topks 256 --questions-path ../data/questions.json --gpus 0

# QIME w/o ontology: MMR activation with the ungrounded question bank
python eval_mteb_mmr.py --topks 256 --questions-path ../data/linear_questions.json --gpus 0

# Top-k binary activation with the ungrounded bank: removes both components
python eval_mteb_ablation.py --topks 256 --questions-path ../data/linear_questions.json

# MMR lambda sweep at fixed k
python eval_mteb_mmr_lambda.py --topk 256 --lambdas 0.1 0.3 0.5 0.7 0.9
```

`eval_mteb_mmr_ablation.py` currently constructs the encoder with its default grounded bank, despite its filename. Use the explicit ungrounded-bank command above for the ontology ablation.

Despite its filename, `test_model_cos_mmr.py` currently uses cosine **top-k** selection without MMR and stores the selected cosine scores instead of binary 1s. It is not the paper's main binary TF+MMR representation.

### Released question banks

| File in `TF/data/` | Questions | Use |
| --- | ---: | --- |
| `questions.json` | 8,855 | Default grounded bank for the main quick start |
| `linear_questions.json` | 3,726 | Ungrounded question bank for ontology ablations |
| `questions_gpt_5_mini.json` | 8,757 | Alternative question-generation LLM |
| `questions_llama.json` | 8,701 | Alternative question-generation LLM |

Keep question ordering fixed: coordinate `j` always refers to question `j` in the selected JSON file. Changing banks changes the coordinate system and may also change its dimension.

## QIME_CLS: classifier comparison

This branch trains per-question classifier heads and is the **QIME_CLS** comparison in paper v3 (§4.3, Appendix C). It does not apply the main method's cosine-based MMR selection.

### Pretrained classifier demo

Download the classifier checkpoint and matching ordered question bank from the [released assets](https://drive.google.com/drive/folders/1YPlQrg_L6U5jS7iHJQOpIcRcpu5L9e4u), placing them at:

```text
checkpoints/QIME/qime_model_base.pt
checkpoints/QIME/questions.json
```

Then run from the repository root:

```bash
python demo.py
```

`demo.py` loads `IMECClassifier`, calls its forward method, and prints **raw per-question logits** and the ten highest-scoring heads. These logits are continuous scores from binary classifiers, not binary embedding values or probabilities.

For custom classifier inference:

```python
import json
import torch
from framework.imec_model import IMECClassifier

with open("checkpoints/QIME/questions.json", "r") as handle:
    questions = json.load(handle)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = IMECClassifier(
    num_labels=len(questions),
    backbone="abhinand/MedEmbed-large-v0.1",
)
model.load_state_dict(torch.load(
    "checkpoints/QIME/qime_model_base.pt", map_location=device,
))
model.to(device)
model.eval()

texts = ["Patient presents with severe chest pain."]
with torch.no_grad():
    logits = model(texts)  # IMECClassifier exposes forward(), not encode().
    probabilities = torch.sigmoid(logits)
    binary_embeddings = (probabilities > 0.5).float()
```

`framework/eval_mteb_medical.py` uses `IMECMTEBModelWrapper` with `use_sigmoid=True`, `is_binary=True`, and `binary_threshold=0.5`. Its classifier representation is therefore `sigmoid(logit) > 0.5`, equivalent to `logit > 0`. It can activate a variable number of dimensions; it does not select a fixed `k=256`.

Run the classifier/baseline evaluator from the repository root:

```bash
python framework/eval_mteb_medical.py
```

The script's existing internal label `QIME` refers to **QIME_CLS**. It also evaluates CQG-MBQA, QAEmb-MBQA, LDIR-UAE-500, and dense baselines. Label results from this branch as `QIME_CLS`, and distinguish raw-logit, sigmoid, and binary representations when reporting them.

### Train QIME_CLS on your corpus

The `framework/run_qime_pipeline.py` pipeline combines OGQG with classifier training. Prepare your corpus as a JSON list of document strings. This training workflow is for QIME_CLS; main-method TF+MMR inference can use the released question bank directly. Run the example below from the repository root after configuring the generation backend and UMLS resources.

```python
import json
import sys
sys.path.insert(0, "framework")
from ogqg import OntologyGroundedQuestionGeneration
from imec import IMEC

# Load your corpus
with open("data/your_corpus.json", "r") as f:
    doc_texts = json.load(f)

# 1. Generate Questions (OGQG)
ogqg = OntologyGroundedQuestionGeneration(
    corpus=doc_texts,
    LLM="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", # Or your preferred LLM
    use_vllm=True,
    temp_folder="./temp",
    name="MyMedicalModel"
)
ogqg.generate_questions()

# 2. Train Model (IMEC)
imec = IMEC(
    corpus=doc_texts,
    LLM="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    temp_folder="./temp",
    output_folder="./output",
    name="MyMedicalModel",
    backbone="abhinand/MedEmbed-large-v0.1"
)
imec.collect_training_data_with_ogqg()
imec.train_model()
```


## Directory structure

```text
QIME/
├── TF/                              # Main QIME implementation and TF ablations
│   ├── code/
│   │   ├── mmr_topk_model.py         # Main QIME: MMR-selected binary embeddings
│   │   ├── test_model.py             # QIME w/o MMR: top-k binary embeddings
│   │   └── test_model_cos_mmr.py     # Additional cosine-valued top-k variant
│   ├── scripts/
│   │   ├── eval_mteb_mmr.py         # Main QIME evaluation
│   │   ├── eval_mteb.py             # Top-k binary evaluation
│   │   ├── eval_mteb_mmr_ablation.py # Legacy helper; uses the default grounded bank
│   │   ├── eval_mteb_ablation.py     # Top-k + ungrounded question bank
│   │   └── eval_mteb_mmr_lambda.py   # MMR lambda sweep
│   ├── dataset/datasets.yaml        # Evaluation task configuration
│   ├── data/                        # Released question banks
│   └── results/                     # Generated TF evaluation outputs
├── framework/
│   ├── ogqg.py                      # Ontology-grounded question generation
│   ├── imec.py                      # QIME_CLS training
│   ├── imec_model.py                # Classifier and evaluation wrapper
│   ├── run_qime_pipeline.py         # OGQG + QIME_CLS training pipeline
│   └── eval_mteb_medical.py          # QIME_CLS and baseline evaluation
├── demo.py                          # Pretrained QIME_CLS raw-logit demo
├── checkpoints/                     # External classifier checkpoints
└── data/                            # Corpus and generation resources
```

Large classifier checkpoints and `data/pubmed_documents_5M.json` are distributed through the [released assets](https://drive.google.com/drive/folders/1YPlQrg_L6U5jS7iHJQOpIcRcpu5L9e4u). Main-method inference uses the included `TF/data/questions.json` and the pretrained MedEmbed encoder, without a QIME_CLS checkpoint.
