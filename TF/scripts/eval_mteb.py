import mteb
import yaml
import argparse
import multiprocessing
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys
sys.path.append("../code")
from test_model import EncoderModel
import json
import os

TASK_NAME_MAP = {
    "biorxiv_clustering_p2p": "BiorxivClusteringP2P",
    "biorxiv_clustering_s2s": "BiorxivClusteringS2S",
    "medrxiv_clustering_p2p": "MedrxivClusteringP2P",
    "medrxiv_clustering_s2s": "MedrxivClusteringS2S",
    "clustrec_covid": "ClusTREC-Covid",
    "biosses": "BIOSSES",
    "r2medii_clinical_retrieval": "R2MEDIIYiClinicalRetrieval",
    "r2medpmc_clinical_retrieval": "R2MEDPMCClinicalRetrieval",
    "nfcorpus": "NFCorpus",
    "public_health_qa": "PublicHealthQA",
    "medical_qa": "MedicalQARetrieval",
    "scifact": "SciFact",
    "arguana": "ArguAna",
    "trec_covid": "TRECCOVID",
}

def load_task_names(config_path="../dataset/datasets.yaml"):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["datasets"]
    names = []
    for item in config:
        mapped = TASK_NAME_MAP.get(item["name"])
        if mapped:
            names.append(mapped)
    return names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=["binary", "hf"], default="binary")
    parser.add_argument("--model-name", default="abhinand/MedEmbed-large-v0.1")  # required only for hf
    parser.add_argument("--topks", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--tasks-config", default="../dataset/datasets.yaml")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--questions-path", default="../data/questions.json")
    args = parser.parse_args()
    return args

def validate_args(args):
    if args.model_kind == "hf" and not args.model_name:
        raise ValueError("model-name is required when model-kind is hf")
    if not args.gpus:
        count = torch.cuda.device_count()
        args.gpus = list(range(count)) if count else [args.gpu_id]
    return args

def get_index_set(model):
    indices_list = model.one_indices
    index_set = set()
    for sub in indices_list:
        for idx in sub:
            index_set.add(idx)
    return index_set

def save_dim_meta(model, task_name, que_stem):
    path = "../results/{model_name}/{que_stem}/no_model_name_available/no_revision_available/indices.json".format(model_name=model.model_name, que_stem=que_stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            indice_json = json.load(handle)
    else:
        indice_json = {}
    index_set = get_index_set(model)
    indice_json[task_name] = {"dim": len(index_set), "index_set": list(index_set), "detail": model.one_indices}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(indice_json, handle)

def choose_device(gpu_id):
    if torch.cuda.is_available():
        return "cuda:{0}".format(gpu_id)
    return "cpu"

def run_topk_worker(topk_vals, task_list, gpu_id, que_path):
    device = choose_device(gpu_id)
    que_stem = os.path.splitext(os.path.basename(que_path))[0]
    for topk_val in topk_vals:
        model = EncoderModel(topk=topk_val, device=device, que_path=que_path)
        for task_name in tqdm(task_list, desc="topk {0}".format(topk_val)):
            result_path = "../results/{model_name}/{que_stem}/no_model_name_available/no_revision_available/{task_name}.json".format(
                model_name=model.model_name, que_stem=que_stem, task_name=task_name)
            if os.path.exists(result_path):
                print(f"Skipping {task_name} (already exists)")
                continue
            model.one_indices = []
            evaluator = mteb.MTEB(tasks=[task_name])
            evaluator.run(model, output_folder="../results/{model_name}/{que_stem}".format(model_name=model.model_name, que_stem=que_stem))
            save_dim_meta(model, task_name, que_stem)

def split_topks(topks, gpus):
    splits = []
    step = (len(topks) + len(gpus) - 1) // len(gpus)
    start = 0
    for gpu_id in gpus:
        end = min(start + step, len(topks))
        splits.append((gpu_id, topks[start:end]))
        start = end
    return splits

def run_binary(topks, task_list, gpus, que_path):
    ctx = multiprocessing.get_context("spawn")
    processes = []
    splits = split_topks(topks, gpus)
    for gpu_id, subset in tqdm(splits, desc="launch topk workers"):
        if not subset:
            continue
        process = ctx.Process(target=run_topk_worker, args=(subset, task_list, gpu_id, que_path))
        process.start()
        processes.append(process)
    for process in tqdm(processes, desc="topk workers done"):
        process.join()

def run_hf(evaluator, model_name):
    model = SentenceTransformer(model_name)
    evaluator.run(model, output_folder="../results/")

def resolve_model_name(args):
    if args.model_kind == "binary":
        return "top{0}_binary".format(args.topks)
    return args.model_name

def main():
    args = validate_args(parse_args())
    task_names = load_task_names(args.tasks_config)
    model_name = resolve_model_name(args)
    print(f"Questions path: {args.questions_path}")
    print(f"Evaluating {model_name}")
    if args.model_kind == "binary":
        run_binary(args.topks, task_names, args.gpus, args.questions_path)
    else:
        evaluator = mteb.MTEB(tasks=mteb.get_tasks(tasks=task_names))
        run_hf(evaluator, args.model_name)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()


