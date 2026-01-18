from ogqg import OntologyGroundedQuestionGeneration
from imec import IMEC
import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("[INFO] Starting QIME Training Pipeline")

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    
    # Load corpus
    corpus_path = os.path.join(dirname, "../data/pubmed_documents_5M.json")
    logger.info(f"[INFO] Loading corpus from {corpus_path}")
    with open(corpus_path, "r") as f:
        doc_texts = json.load(f)
    logger.info(f"[INFO] Loaded {len(doc_texts)} documents")

    # Path to pre-extracted medical terms
    terms_path = os.path.join(dirname, "../data/pubmedqa_hunflair_terms_complete.json")

    # 1. Ontology-Grounded Question Generation (OGQG)
    logger.info("[INFO] Initializing OGQG module")
    ogqg = OntologyGroundedQuestionGeneration(
        corpus=doc_texts,
        LLM="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        use_vllm=True,
        tensor_parallel_size=1,
        batch_size=512,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.8,
        temp_folder="./temp",
        output_folder="./output",
        name="QIME-Training",
        encoder="abhinand/MedEmbed-large-v0.1",
        k=2500,
        theta=0.8,
        use_umls=True,
        use_preextracted_terms=True,
        preextracted_terms_path=terms_path,
        enable_semtype_filtering=False
    )
    ogqg.generate_questions()

    # 2. Interpretable Medical Embedding Construction (IMEC)
    logger.info("[INFO] Initializing IMEC module")
    imec = IMEC(
        corpus=doc_texts,
        LLM="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        use_vllm=True,
        tensor_parallel_size=1,
        batch_size=512,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.8,
        temp_folder="./temp",
        output_folder="./output",
        name="QIME-Training",
        backbone="abhinand/MedEmbed-large-v0.1",
        enable_dpos_filtering=False,
        num_steps=4000000,
        checkpoint_steps=[2500000, 3000000, 3500000, 4000000]
    )
    imec.collect_training_data_with_ogqg()
    imec.train_model()