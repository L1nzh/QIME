from argparse import ArgumentParser
from json import dump
from multiprocessing import get_context, set_start_method
from pathlib import Path
from sys import path as python_path

import mteb
import torch
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
TASKS_CONFIG_PATH = REPO_ROOT / "dataset" / "datasets.yaml"
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.json"
RESULTS_DIR = REPO_ROOT / "rebuttal" / "results"
INNER_RESULTS_DIR = Path("no_model_name_available") / "no_revision_available"
DEFAULT_LAMBDAS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_TOPK = 256

python_path.append(str(CODE_DIR))
from mmr_topk_model import EncoderModel  # noqa: E402


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


def load_task_names(config_path=TASKS_CONFIG_PATH):
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        dataset_config = yaml.safe_load(config_file)["datasets"]
    task_names = []
    for task_record in dataset_config:
        task_name = TASK_NAME_MAP[task_record["name"]]
        task_names.append(task_name)
    return task_names


def parse_arguments(argument_values=None):
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    argument_parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
    )
    argument_parser.add_argument("--tasks-config", type=Path, default=TASKS_CONFIG_PATH)
    argument_parser.add_argument("--gpu-id", type=int, default=1)
    argument_parser.add_argument("--gpus", nargs="+", type=int, default=None)
    argument_parser.add_argument("--questions-path", type=Path, default=QUESTIONS_PATH)
    return argument_parser.parse_args(argument_values)


def validate_arguments(parsed_arguments):
    if parsed_arguments.gpus is None:
        parsed_arguments.gpus = [parsed_arguments.gpu_id]
    return parsed_arguments


def get_index_set(model):
    index_set = set()
    for index_record in model.one_indices:
        for index_value in index_record:
            index_set.add(index_value)
    return index_set


def build_output_folder(model_name, questions_path):
    return RESULTS_DIR / model_name / Path(questions_path).stem


def build_indices_path(model_name, questions_path):
    return build_output_folder(model_name, questions_path) / INNER_RESULTS_DIR / "indices.json"


def save_dim_meta(model, task_name, questions_path, index_summary):
    index_set = get_index_set(model)
    index_summary[task_name] = {
        "dim": len(index_set),
        "index_set": list(index_set),
        "detail": model.one_indices,
    }
    indices_path = build_indices_path(model.model_name, questions_path)
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    with indices_path.open("w", encoding="utf-8") as indices_file:
        dump(index_summary, indices_file)


def choose_device(gpu_id):
    return f"cuda:{gpu_id}"


def evaluate_task(model, task_name, output_folder):
    evaluator = mteb.MTEB(tasks=[task_name])
    if task_name == "PublicHealthQA":
        evaluator.run(
            model,
            output_folder=str(output_folder),
            eval_subsets=["english"],
            overwrite_results=False,
        )
    else:
        evaluator.run(
            model,
            output_folder=str(output_folder),
            overwrite_results=False,
        )


def run_lambda_worker(lambda_values, task_names, gpu_id, topk, questions_path):
    device = choose_device(gpu_id)
    for lambda_value in lambda_values:
        model = EncoderModel(
            topk=topk,
            mmr_diversity=lambda_value,
            device=device,
            que_path=str(questions_path),
        )
        output_folder = build_output_folder(model.model_name, questions_path)
        index_summary = {}
        for task_name in tqdm(task_names, desc=f"lambda {lambda_value}"):
            model.one_indices = []
            evaluate_task(model, task_name, output_folder)
            save_dim_meta(model, task_name, questions_path, index_summary)


def split_lambdas(lambda_values, gpu_ids):
    lambda_splits = []
    split_size = (len(lambda_values) + len(gpu_ids) - 1) // len(gpu_ids)
    split_start = 0
    for gpu_id in gpu_ids:
        split_end = min(split_start + split_size, len(lambda_values))
        lambda_splits.append((gpu_id, lambda_values[split_start:split_end]))
        split_start = split_end
    return lambda_splits


def run_lambda_sweep(lambda_values, task_names, gpu_ids, topk, questions_path):
    process_context = get_context("spawn")
    processes = []
    lambda_splits = split_lambdas(lambda_values, gpu_ids)
    for gpu_id, lambda_subset in tqdm(lambda_splits, desc="launch lambda workers"):
        if not lambda_subset:
            continue
        process = process_context.Process(
            target=run_lambda_worker,
            args=(lambda_subset, task_names, gpu_id, topk, questions_path),
        )
        process.start()
        processes.append(process)
    for process in tqdm(processes, desc="lambda workers done"):
        process.join()


def run_evaluation():
    parsed_arguments = validate_arguments(parse_arguments())
    task_names = load_task_names(parsed_arguments.tasks_config)
    print(f"Questions path: {parsed_arguments.questions_path}")
    print(f"Topk: {parsed_arguments.topk}")
    print(f"Lambdas: {parsed_arguments.lambdas}")
    run_lambda_sweep(
        parsed_arguments.lambdas,
        task_names,
        parsed_arguments.gpus,
        parsed_arguments.topk,
        parsed_arguments.questions_path,
    )


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    run_evaluation()
