#!/usr/bin/env python3
"""
Medical MTEB Evaluation Script for QIME Project
Evaluates the following models:
1. CQG-MBQA (Baseline)
2. QAEmb-MBQA (Baseline)
3. QIME (Ours)
4. LDIR-UAE-500 (Baseline)
And various general/medical baseline models.
"""

import os
import sys
import json
import torch
import mteb
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Ensure framework modules can be imported
sys.path.append(os.path.dirname(__file__))

from imec_model import IMECClassifier, IMECMTEBModelWrapper
from anchor_sim_model import cos_simi_model
from sentence_transformers import SentenceTransformer
from utils import BagOfTokenEncoder

# ==============================================================================
# Model Wrapper Classes
# ==============================================================================

class SBERTEncodingModelGeneral:
    """SBERT encoder wrapper for MTEB evaluation."""
    def __init__(self, model, device="cuda"):
        self.model = SentenceTransformer(model, device=device)

    def encode(self, sentences, **kwargs):
        # Filter unsupported arguments
        unsupported_keys = [
            'task_name', 'task_category', 'data_split', 'max_seq_length',
            'return_numpy', 'batch_size', 'normalize_embeddings', 'prompt_type'
        ]
        for key in unsupported_keys:
            if key in kwargs:
                kwargs.pop(key)

        kwargs.setdefault('show_progress_bar', False)
        convert_to_tensor = kwargs.pop('convert_to_tensor', True)
        convert_to_numpy = kwargs.pop('convert_to_numpy', False)

        embeddings = self.model.encode(sentences, convert_to_tensor=True, **kwargs)

        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu()

        if convert_to_numpy or not convert_to_tensor:
            return embeddings.numpy() if hasattr(embeddings, 'numpy') else embeddings
        else:
            return embeddings

class LDIRModelWrapper:
    """LDIR model wrapper for MTEB evaluation."""
    def __init__(self, anchor_file_path, backbone_model="WhereIsAI/UAE-Large-V1", batch_size=256):
        if not os.path.exists(anchor_file_path):
            raise FileNotFoundError(f"Anchor file not found: {anchor_file_path}")

        with open(anchor_file_path, "r") as f:
            anchors_dict = json.load(f)

        self.anchors_list = list(anchors_dict.values())

        class ModelArgs:
            def __init__(self, model_name_or_path):
                self.model_name_or_path = model_name_or_path

        model_args = ModelArgs(backbone_model)
        self.model = cos_simi_model(model_args, anchors=self.anchors_list, batch_size=batch_size, is_binary=False)

    def encode(self, sentences, **kwargs):
        # Pass through to the underlying model
        embeddings = self.model.encode(sentences)
        return embeddings

# ==============================================================================
# Model Loading Logic
# ==============================================================================

def get_model_wrapper(model_name):
    """Load model wrapper for MTEB evaluation."""
    dirname = os.path.dirname(__file__)
    checkpoints_dir = os.path.join(dirname, "../checkpoints")

    # 1. LDIR-UAE-500
    if model_name == "LDIR-UAE-500":
        anchor_file = os.path.join(checkpoints_dir, "LDIR-UAE-500/anchors.json")
        print(f"[INFO] Loading LDIR-UAE-500 from {anchor_file}")
        return LDIRModelWrapper(
            anchor_file_path=anchor_file, 
            backbone_model="WhereIsAI/UAE-Large-V1", 
            batch_size=256
        )

    # 2. QIME Models (CQG-MBQA, QAEmb-MBQA, QIME)
    model_config = {
        "CQG-MBQA": {
            "path": "CQG-MBQA", # Keeping original path for now
            "model_file": "mbqa_model.pt", # Keeping original filename
            "backbone": "WhereIsAI/UAE-Large-V1"
        },
        "QAEmb-MBQA": {
            "path": "QAEmb-MBQA", # Keeping original path
            "model_file": "qaemb_model.pt",
            "backbone": "WhereIsAI/UAE-Large-V1"
        },
        "QIME": {
            "path": "QIME",
            "model_file": "qime_model_base.pt",
            "backbone": "abhinand/MedEmbed-large-v0.1"
        }
    }

    if model_name in model_config:
        config = model_config[model_name]
        base_path = os.path.join(checkpoints_dir, config["path"])
        questions_file = os.path.join(base_path, "questions.json")
        model_file = os.path.join(base_path, config["model_file"])

        print(f"[INFO] Loading {model_name}...")
        print(f"  - Questions: {questions_file}")
        print(f"  - Model: {model_file}")
        print(f"  - Backbone: {config['backbone']}")

        if not os.path.exists(questions_file):
            raise ValueError(f"Questions file not found: {questions_file}")
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")

        # Load Questions
        with open(questions_file, "r") as f:
            linear_questions = json.load(f)

        # Load Model
        model = IMECClassifier(num_labels=len(linear_questions), backbone=config["backbone"])
        model.load_state_dict(torch.load(model_file, map_location="cuda:0"))
        model.to("cuda")
        model.eval()

        return IMECMTEBModelWrapper(
            model, linear_questions,
            is_binary=True, 
            is_sparse=False,
            binary_threshold=0.5, 
            use_sigmoid=True
        )
    
    # 3. Bag of Words
    if model_name == "bag_of_words":
        print(f"[INFO] Loading BagOfTokenEncoder")
        return BagOfTokenEncoder()

    # Fallback for general SBERT models
    print(f"[INFO] Loading general SBERT model: {model_name}")
    return SBERTEncodingModelGeneral(model_name)

# ==============================================================================
# Evaluation Logic
# ==============================================================================

def get_medical_tasks():
    """Get list of medical MTEB tasks for evaluation."""
    return [
        # STS
        mteb.get_task("BIOSSES", languages=["eng"]),
        # Retrieval
        mteb.get_task("R2MEDIIYiClinicalRetrieval", languages=["eng"]),
        mteb.get_task("R2MEDPMCClinicalRetrieval", languages=["eng"]),
        mteb.get_task("NFCorpus", languages=["eng"]),
        mteb.get_task("PublicHealthQA", languages=["eng"]),
        mteb.get_task("MedicalQARetrieval", languages=["eng"]),
        mteb.get_task("TRECCOVID", languages=["eng"]),
        # Clustering
        mteb.get_task("BiorxivClusteringP2P", languages=["eng"]),
        mteb.get_task("BiorxivClusteringS2S", languages=["eng"]),
        mteb.get_task("MedrxivClusteringP2P", languages=["eng"]),
        mteb.get_task("MedrxivClusteringS2S", languages=["eng"]),
        mteb.get_task("ClusTREC-Covid", languages=["eng"]),
    ]

def evaluate_single_model(model_name, tasks):
    """Evaluate a single model on medical MTEB tasks."""
    print(f"\n[INFO] Evaluating {model_name}")

    try:
        model_wrapper = get_model_wrapper(model_name)
        
        output_folder = os.path.join(os.path.dirname(__file__), f"../results/evaluation/model_outputs/{model_name.replace('/', '_')}")
        os.makedirs(output_folder, exist_ok=True)

        completed_tasks = []

        for task in tasks:
            task_name = task.metadata.name
            print(f"  [INFO] Evaluating task: {task_name}")

            # Reset task context if supported
            if hasattr(model_wrapper, 'set_current_task'):
                model_wrapper.set_current_task(task_name)

            single_task_evaluation = mteb.MTEB(tasks=[task])
            single_task_evaluation.run(model_wrapper, output_folder=output_folder)
            completed_tasks.append(task_name)

        print(f"  [INFO] Completed: {len(completed_tasks)}/{len(tasks)} tasks")

        # Cleanup to free GPU memory
        if hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'to'):
            model_wrapper.model.to('cpu')
        torch.cuda.empty_cache()

        return completed_tasks

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

def check_evaluation_completeness(models, tasks):
    """Check if all models have been evaluated on all tasks."""
    base_dir = os.path.join(os.path.dirname(__file__), "../results/evaluation/model_outputs")
    
    missing_evaluations = []
    
    for model_name in models:
        model_dir_name = f"{model_name.replace('/', '_')}"
        model_path = os.path.join(base_dir, model_dir_name, "no_model_name_available", "no_revision_available")
        
        if not os.path.exists(model_path):
            missing_evaluations.append(f"{model_name}: Directory not found")
            continue
            
        for task in tasks:
            task_name = task.metadata.name
            task_file = os.path.join(model_path, f"{task_name}.json")
            
            if not os.path.exists(task_file):
                missing_evaluations.append(f"{model_name}: Missing {task_name}")
    
    return missing_evaluations

def generate_results_csv(models, tasks, metric_name='ndcg_at_10', output_suffix=''):
    """Generate CSV with model scores.
    
    If metric_name is 'main_metric', it selects the appropriate metric per task:
    - STS (BIOSSES): spearman
    - Clustering: v_measure
    - Retrieval: ndcg_at_10
    """
    if metric_name == 'main_metric':
        print(f"  [INFO] Generating CSV for Main Metrics (Mixed)")
    else:
        metric_display = metric_name.upper().replace('_', '@') if '@' not in metric_name else metric_name
        print(f"  [INFO] Generating CSV for {metric_display}")

    base_dir = os.path.join(os.path.dirname(__file__), "../results/evaluation/model_outputs")
    results_data = []

    for model_name in models:
        model_dir_name = f"{model_name.replace('/', '_')}"
        model_path = os.path.join(base_dir, model_dir_name, "no_model_name_available", "no_revision_available")

        row = {"Model": model_name}
        task_scores = []

        for task in tasks:
            task_name = task.metadata.name
            task_type = task.metadata.type # e.g. "Retrieval", "STS", "Clustering" 
            
            # Determine metric for this task
            if metric_name == 'main_metric':
                if task_name == "BIOSSES" or task_type == "STS":
                    target_metric = "spearman"
                elif "Clustering" in task_name or task_type == "Clustering":
                    target_metric = "v_measure"
                else:
                    target_metric = "ndcg_at_10"
            else:
                target_metric = metric_name

            task_file = os.path.join(model_path, f"{task_name}.json")
            custom_score = 'N/A'
            
            if os.path.exists(task_file):
                try:
                    with open(task_file, 'r') as f:
                        task_results = json.load(f)

                    if 'scores' in task_results:
                        scores = task_results['scores']
                        # Try common split keys
                        for score_key in ['test', 'dev', 'all']:
                            if score_key in scores:
                                target_scores = scores[score_key]
                                if isinstance(target_scores, list) and len(target_scores) > 0:
                                    target_scores = target_scores[0]

                                if isinstance(target_scores, dict):
                                    # Handle cos_sim nesting for STS
                                    if target_metric == "spearman" and "cos_sim" in target_scores:
                                        if "spearman" in target_scores["cos_sim"]:
                                            custom_score = target_scores["cos_sim"]["spearman"]
                                            break
                                    
                                    # Standard metric lookup
                                    if target_metric in target_scores and target_scores[target_metric] is not None:
                                        custom_score = target_scores[target_metric]
                                        break
                                        
                except Exception as e:
                    print(f"  [WARN] Error reading {task_file}: {e}")
                    row[task_name] = 'Error'
            else:
                row[task_name] = 'Missing'

            row[task_name] = custom_score
            if isinstance(custom_score, (int, float)):
                task_scores.append(custom_score)

        if task_scores:
            row["Average"] = round(sum(task_scores) / len(task_scores), 4)
        else:
            row["Average"] = 'N/A'

        results_data.append(row)

    df = pd.DataFrame(results_data)
    task_columns = [task.metadata.name for task in tasks]
    column_order = ["Model"] + task_columns + ["Average"]
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    output_dir = os.path.join(os.path.dirname(__file__), "../results/medical_mteb_retrieval_metrics")
    os.makedirs(output_dir, exist_ok=True)

    if metric_name == 'main_metric':
        csv_filename = "medical_mteb_main_results.csv"
    elif output_suffix:
        csv_filename = f"retrieval_{output_suffix}.csv"
    else:
        csv_filename = f"results_{metric_name}.csv"

    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    return df

def generate_all_metrics(models, tasks):
    """Generate summary CSVs."""
    # 1. Main Mixed Metrics Table
    generate_results_csv(models, tasks, metric_name='main_metric')
    
    # 2. Specific Metric Tables
    metrics = [
        ('ndcg_at_10', 'ndcg_at_10'),
        ('mrr_at_10', 'mrr_at_10'),
        ('recall_at_10', 'recall_at_10'),
    ]
    
    for metric_name, suffix in metrics:
        generate_results_csv(models, tasks, metric_name, suffix)

def main():
    print("=" * 80)
    print("QIME Medical MTEB Evaluation")
    print("=" * 80)
    
    tasks = get_medical_tasks()
    print(f"\n[INFO] Target Tasks ({len(tasks)}): {[t.metadata.name for t in tasks]}")
    
    models = [
        "CQG-MBQA",
        "QAEmb-MBQA",
        "QIME",
        "LDIR-UAE-500",
        "bag_of_words",
        
        'google-bert/bert-base-uncased',
        'sentence-transformers/average_word_embeddings_glove.6B.300d',
        'princeton-nlp/unsup-simcse-bert-base-uncased',
        'princeton-nlp/sup-simcse-bert-base-uncased',
        'BMRetriever/BMRetriever-410M',
        'abhinand/MedEmbed-large-v0.1',
        'ncbi/MedCPT-Query-Encoder',
        'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        'FremyCompany/BioLORD-2023',
        'NeuML/pubmedbert-base-embeddings',
        'sentence-transformers/embeddinggemma-300m-medical',
    ]
    print(f"\n[INFO] Target Models ({len(models)}): {models}")
    
    # Check evaluation status
    print("\n[INFO] Checking evaluation status...")
    missing_evaluations = check_evaluation_completeness(models, tasks)
    
    if missing_evaluations:
        print(f"[INFO] Found {len(missing_evaluations)} missing evaluations. Starting execution...")
        for model_name in models:
            evaluate_single_model(model_name, tasks)
    else:
        print("[INFO] All evaluations are already complete!")
        
    # Generate Reports
    print("\n[INFO] Generating Reports...")
    generate_all_metrics(models, tasks)
    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()