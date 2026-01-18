#!/usr/bin/env python3
"""
Extract medical terms from PubMed corpus using HunFlair2 NER.
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from flair.data import Sentence
from flair.nn import Classifier
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_pubmed_documents(file_path: str, start_idx: int = 0, limit: int = None) -> tuple:
    """Load documents from PubMed corpus."""
    logger.info(f"Loading documents from {file_path}...")
    
    if not os.path.exists(file_path):
        logger.error(f"Corpus file not found: {file_path}")
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    total_docs = len(documents)
    logger.info(f"Total documents in corpus: {total_docs}")
    
    if limit is not None:
        end_idx = min(start_idx + limit, total_docs)
        documents_slice = documents[start_idx:end_idx]
        logger.info(f"Processing documents {start_idx} to {end_idx-1} ({len(documents_slice)} documents)")
    else:
        documents_slice = documents[start_idx:]
        logger.info(f"Processing documents {start_idx} to {total_docs-1} ({len(documents_slice)} documents)")
    
    return documents_slice, start_idx

def initialize_hunflair_model(gpu_id: int = None):
    """Initialize HunFlair2 NER model."""
    logger.info("Initializing HunFlair2 NER model...")
    
    device_name = "cpu"
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
            device_name = f'cuda:{gpu_id}'
        else:
            device = torch.device('cuda')
            device_name = 'cuda'
        
        # Set flair device
        import flair
        flair.device = device
        logger.info(f"[INFO] Using GPU: {device_name} ({torch.cuda.get_device_name(device)})")
    else:
        logger.info("[INFO] Using CPU mode")
    
    # Load HunFlair2 NER model
    start_time = time.time()
    ner_tagger = Classifier.load("hunflair2")
    load_time = time.time() - start_time
    logger.info(f"[INFO] NER model loaded in {load_time:.1f}s")
    
    return ner_tagger

def extract_medical_terms_batch(
    documents: List[str], 
    ner_tagger,
    batch_size: int = 128,
) -> List[List[str]]:
    """Extract medical terms from documents in batches."""
    
    all_terms = []
    total_docs = len(documents)
    
    logger.info(f"Processing {total_docs} documents in batches of {batch_size}")
    
    for i in tqdm(range(0, total_docs, batch_size), desc="Extracting terms"):
        batch_docs = documents[i:i + batch_size]
        
        # Create batch sentences with truncation
        batch_sentences = []
        for doc in batch_docs:
            truncated_text = doc[:2000] if len(doc) > 2000 else doc
            batch_sentences.append(Sentence(truncated_text))
        
        # Parallel prediction using Flair's batch processing
        ner_tagger.predict(batch_sentences, mini_batch_size=batch_size)
        
        # Extract terms from all sentences in batch
        batch_terms_list = []
        for sentence in batch_sentences:
            terms = []
            for entity in sentence.get_spans('ner'):
                entity_text = entity.text.strip()
                if entity_text and len(entity_text) > 1:  # Filter very short terms
                    terms.append(entity_text)
            # Remove duplicates while preserving order
            unique_terms = list(dict.fromkeys(terms))
            batch_terms_list.append(unique_terms)
        
        all_terms.extend(batch_terms_list)
        
        # Progress info - removed detailed rate logging to keep it concise in tqdm
    
    return all_terms

def main():
    parser = argparse.ArgumentParser(description="Extract medical terms from PubMed corpus using HunFlair2")
    parser.add_argument(
        "--corpus_path", 
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/pubmed_documents_5M.json"),
        help="Path to PubMed corpus"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting document index"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Number of documents to process (None for all)"
    )
    parser.add_argument(
        "--output_file", 
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/pubmed_hunflair_terms_complete.json"),
        help="Final output file for extracted terms"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=None,
        help="Specific GPU ID to use"
    )
    
    args = parser.parse_args()
    
    logger.info("[INFO] HunFlair2 Medical Term Extraction from PubMed Documents")
    
    # Load documents
    documents, start_idx = load_pubmed_documents(
        args.corpus_path, 
        args.start_idx, 
        args.limit
    )
    
    if not documents:
        logger.warning("[WARN] No documents to process. Exiting.")
        return
    
    # Initialize HunFlair2 model
    ner_tagger = initialize_hunflair_model(args.gpu_id)
    
    try:
        # Extract medical terms
        logger.info(f"[INFO] Starting HunFlair2 medical term extraction for {len(documents)} documents...")
        start_time = time.time()
        
        all_terms = extract_medical_terms_batch(
            documents, 
            ner_tagger,
            args.batch_size,
        )
        
        total_time = time.time() - start_time
        
        # Create output directory and save final results
        output_dir = os.path.dirname(args.output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        final_results = {
            "corpus_path": args.corpus_path,
            "method": "HunFlair2_NER",
            "start_index": start_idx,
            "end_index": start_idx + len(documents),
            "total_documents_processed": len(documents),
            "total_time_seconds": total_time,
            "medical_terms": all_terms
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Statistics
        total_terms = sum(len(terms) for terms in all_terms)
        docs_with_terms = sum(1 for terms in all_terms if terms)
        avg_terms_per_doc = total_terms / len(documents) if documents else 0
        
        logger.info(f"[INFO] HunFlair2 extraction complete.")
        logger.info(f"  Documents processed: {len(documents)}")
        logger.info(f"  Documents with terms: {docs_with_terms}")
        logger.info(f"  Total terms extracted: {total_terms}")
        logger.info(f"  Average terms per document: {avg_terms_per_doc:.1f}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Results saved to: {args.output_file}")
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[INFO] GPU memory cleaned up.")

if __name__ == "__main__":
    main()
