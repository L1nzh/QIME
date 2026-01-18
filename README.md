# QIME: Quality & Interpretability optimized Medical Embeddings

🎯 **Generate high-quality interpretable semantic text embeddings by creating and answering discriminative yes/no questions.** 

This project implements the QIME framework, featuring:
*   **OGQG (Ontology-Grounded Question Generation)**: Generates discriminative yes/no questions grounded in medical ontologies (UMLS).
*   **IMEC (Interpretable Medical Embedding Construction)**: Trains a multi-task binary classifier to answer these questions, producing interpretable embeddings.

## 🌟 Key Features
- 😎 **Interpretable Semantic Text Embeddings**: Each dimension of the embedding corresponds to a specific medical question (e.g., "Does this text discuss a cardiovascular disease?").
- 🛠️ **General Framework**: Customizable for specific medical domains or general biomedical text.
- 🚀 **Easy to Use**: Integrated pipeline for generation and training.
- 📚 **No Labels Required**: Unsupervised training using plain text documents.

## Directory Structure
```
QIME/
├── checkpoints/       # Pre-trained models (QIME, OGQG, QAEmb-IMEC)
├── data/              # Datasets (PubMed documents, terms)
├── framework/         # Core source code
│   ├── ogqg.py        # Ontology-Grounded Question Generation
│   ├── imec.py        # Interpretable Medical Embedding Construction
│   ├── imec_model.py  # Model architecture
│   ├── run_qime_pipeline.py # Main training pipeline
│   └── eval_mteb_medical.py # Evaluation script
└── results/           # Evaluation outputs
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (recommended)
- OpenAI API Key (for question generation)

### Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training Your Own Model
To train a model from scratch using your own corpus (e.g., PubMed documents), use the `run_qime_pipeline.py` script.

**Data Format**: Prepare your corpus as a JSON file containing a list of strings (documents).

**Example Training Script**:
```python
from framework.ogqg import OntologyGroundedQuestionGeneration
from framework.imec import IMEC
import json

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

### 2. Evaluating Models
We provide a comprehensive evaluation script `framework/eval_mteb_medical.py` that evaluates models on the MTEB Medical benchmark (Retrieval, STS, Clustering).

```bash
python framework/eval_mteb_medical.py
```

This script evaluates:
*   **QIME** (Ours)
*   **CQBA-MBQA** (Baseline)
*   **QAEmb-IMEC** (Baseline)
*   **LDIR-UAE-500** (Baseline)
*   Various general domain models (BERT, BioLORD, etc.)

## Using Pretrained Models for Inference

A simple `demo.py` script is provided in the root directory to showcase how to use a pretrained QIME model for inference and interpret its embeddings.

To run the demo:
```bash
python demo.py
```

The script will:
1.  Load the pretrained QIME model.
2.  Encode a sample input text.
3.  Print the embedding result.
4.  Identify and print the top 10 highest-scoring heads (questions) from the embedding, providing an interpretation of the text.

You can also load a trained model directly using `IMECClassifier` or `IMECMTEBModelWrapper` for custom inference:

```python
from framework.imec_model import IMECClassifier
import torch
import json

# Load questions (defining the dimensions)
with open("checkpoints/QIME/questions.json", "r") as f:
    questions = json.load(f)

# Initialize model
model = IMECClassifier(num_labels=len(questions), backbone="abhinand/MedEmbed-large-v0.1")
model.load_state_dict(torch.load("checkpoints/QIME/qime_model_base.pt"))
model.eval()

# Encode documents
docs = ["Patient presents with severe chest pain.", "Treatment involves daily insulin injections."]
embeddings = model.encode(docs)

# embeddings shape: [2, num_questions]
```


