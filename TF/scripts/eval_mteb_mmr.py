"""Evaluate native QIME TF+MMR, with a pinned paper-12 protocol via --paper."""
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys

import mteb
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import yaml

TF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TF_ROOT / "code"))
from mmr_topk_model import EncoderModel

BACKBONE_REVISION = "963121bfb9c625475f65b08fb54990ce9c4e7a1a"
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


def load_task_specs(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    specs = []
    for item in config["datasets"]:
        if item["name"] not in TASK_NAME_MAP:
            raise ValueError(f"Unknown task alias: {item['name']}")
        specs.append({**item, "task_name": TASK_NAME_MAP[item["name"]]})
    if len({s["task_name"] for s in specs}) != len(specs):
        raise ValueError("Duplicate tasks in configuration")
    return config, specs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true", help="Use only the 12 paper-main tasks and pinned data revisions")
    parser.add_argument("--list-tasks", action="store_true", help="Inspect task/subset/revision configuration without loading a model or datasets")
    parser.add_argument("--model-kind", choices=["binary", "hf"], default="binary")
    parser.add_argument("--model-name", default="abhinand/MedEmbed-large-v0.1", help="Dense baseline only when --model-kind hf")
    parser.add_argument("--topks", nargs="+", type=int)
    parser.add_argument("--tasks-config", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--questions-path", type=Path, default=TF_ROOT / "data/questions.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmr-diversity", type=float, default=0.7)
    parser.add_argument("--backbone-revision", default=BACKBONE_REVISION)
    parser.add_argument("--save-indices", action="store_true", help="Also save every activation index; these files can be large")
    args = parser.parse_args(argv)
    args.topks = args.topks or ([256] if args.paper else [128, 256])
    args.tasks_config = args.tasks_config or TF_ROOT / "dataset" / ("paper12.yaml" if args.paper else "datasets.yaml")
    if args.batch_size < 1 or not 0 < args.mmr_diversity <= 1:
        parser.error("batch-size must be positive and mmr-diversity must be in (0, 1]")
    if args.paper and (args.model_kind != "binary" or args.topks != [256] or args.mmr_diversity != 0.7):
        parser.error("--paper selects the main TF+MMR model: binary, k=256, lambda=0.7")
    if not args.gpus:
        args.gpus = list(range(torch.cuda.device_count())) or [args.gpu_id]
    return args


def prepare_task(spec, seed):
    # PublicHealthQA's unfiltered default contains eight languages.
    task = mteb.get_tasks(tasks=[spec["task_name"]], languages=["eng"])[0]
    task.seed = seed
    if "revision" in spec:
        if task.metadata.dataset["path"] != spec["id"]:
            raise ValueError(f"Dataset path mismatch for {spec['task_name']}")
        task.metadata.dataset["revision"] = spec["revision"]
    if spec.get("metric") and task.metadata.main_score != spec["metric"]:
        raise ValueError(f"Metric mismatch for {spec['task_name']}")
    subsets = spec.get("eval_subsets", task.hf_subsets)
    if any(subset not in task.hf_subsets for subset in subsets):
        raise ValueError(f"Unknown subset for {spec['task_name']}: {subsets}")
    return task, subsets


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BatchedMMREncoder:
    """Bound dense encoding and MMR batches without changing the native rule."""
    def __init__(self, model, save_indices=False):
        self.model = model
        self.save_indices = save_indices
        self.one_indices = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode(self, sentences, batch_size=128, **kwargs):
        sentences = [sentences] if isinstance(sentences, str) else list(sentences)
        output = np.empty((len(sentences), len(self.model.questions)), dtype=np.float32)
        for start in range(0, len(sentences), batch_size):
            self.model.one_indices = []
            chunk = self.model.encode(sentences[start:start + batch_size], batch_size=batch_size, **kwargs)
            output[start:start + len(chunk)] = chunk
            if self.save_indices:
                self.one_indices.extend(self.model.one_indices)
        self.model.one_indices = []
        return output


def output_folder(args, model_name):
    root = TF_ROOT / "results"
    if args.paper:
        root /= "paper12"
    return root / model_name / args.questions_path.stem / f"seed{args.seed}_batch{args.batch_size}_{args.backbone_revision[:12]}"


def run_topk_worker(topks, specs, gpu_id, args):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    versions = {p: importlib.metadata.version(p) for p in [
        "torch", "sentence-transformers", "transformers", "huggingface-hub",
        "numpy", "mteb", "datasets", "scikit-learn", "scipy",
    ]}
    for topk in topks:
        seed_all(args.seed)
        native = EncoderModel(topk=topk, mmr_diversity=args.mmr_diversity,
                              device=device, que_path=str(args.questions_path),
                              backbone_revision=args.backbone_revision)
        model = BatchedMMREncoder(native, save_indices=args.save_indices)
        folder = output_folder(args, model.model_name)
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {
            "method": "QIME-TF-MMR", "topk": topk, "mmr_lambda": args.mmr_diversity,
            "backbone": "abhinand/MedEmbed-large-v0.1", "backbone_revision": args.backbone_revision,
            "questions_sha256": hashlib.sha256(args.questions_path.read_bytes()).hexdigest(),
            "num_questions": len(model.questions), "seed": args.seed, "batch_size": args.batch_size,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "versions": versions, "tasks": [],
        }
        prepared = []
        for spec in specs:
            task, subsets = prepare_task(spec, args.seed)
            prepared.append((task, subsets))
            manifest["tasks"].append({"name": task.metadata.name, "dataset": task.metadata.dataset,
                                      "eval_subsets": subsets, "eval_splits": ["test"],
                                      "metric": task.metadata.main_score})
        manifest_path = folder / "run_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Existing results have a different protocol: {folder}")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        for task, subsets in tqdm(prepared, desc=f"k={topk}, lambda={args.mmr_diversity}"):
            seed_all(args.seed)
            model.one_indices = []
            mteb.MTEB(tasks=[task]).run(model, output_folder=str(folder), eval_splits=["test"],
                                      eval_subsets=subsets, encode_kwargs={"batch_size": args.batch_size},
                                      raise_error=True, co2_tracker=False)
            if args.save_indices:
                (folder / f"{task.metadata.name}.indices.json").write_text(json.dumps(model.one_indices))


def split_topks(topks, gpus):
    step = (len(topks) + len(gpus) - 1) // len(gpus)
    return [(gpu, topks[i * step:(i + 1) * step]) for i, gpu in enumerate(gpus)]


def main(argv=None):
    args = parse_args(argv)
    _, specs = load_task_specs(args.tasks_config)
    if args.paper:
        expected = yaml.safe_load((TF_ROOT / "dataset/paper12.yaml").read_text())
        if specs != load_task_specs(TF_ROOT / "dataset/paper12.yaml")[1]:
            raise ValueError("--paper requires the supplied paper12 task definitions")
        bank_sha = hashlib.sha256(args.questions_path.read_bytes()).hexdigest()
        if bank_sha != expected["questions_sha256"]:
            raise ValueError("--paper requires the released ordered question bank")
        if args.backbone_revision != expected["backbone_revision"]:
            raise ValueError("--paper requires the pinned MedEmbed backbone revision")
    if args.list_tasks:
        for spec in specs:
            task, subsets = prepare_task(spec, args.seed)
            print(json.dumps({"task": task.metadata.name, "dataset": task.metadata.dataset,
                              "eval_subsets": subsets, "eval_splits": ["test"]}))
        return
    # spawn starts children with this hash seed before PublicHealthQA builds IDs from sets.
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.model_kind == "hf":
        seed_all(args.seed)
        kwargs = {"revision": args.backbone_revision} if args.model_name == "abhinand/MedEmbed-large-v0.1" else {}
        model = SentenceTransformer(args.model_name, **kwargs)
        for spec in specs:
            task, subsets = prepare_task(spec, args.seed)
            seed_all(args.seed)
            mteb.MTEB(tasks=[task]).run(model, output_folder=str(TF_ROOT / "results"),
                                      eval_splits=["test"], eval_subsets=subsets,
                                      encode_kwargs={"batch_size": args.batch_size})
        return
    context = multiprocessing.get_context("spawn")
    processes = []
    for gpu_id, topks in split_topks(args.topks, args.gpus):
        if topks:
            process = context.Process(target=run_topk_worker, args=(topks, specs, gpu_id, args))
            process.start()
            processes.append(process)
    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(process.exitcode)
    if failed:
        raise SystemExit(f"Evaluation worker(s) failed: exit codes {failed}")


if __name__ == "__main__":
    main()
