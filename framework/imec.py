from imec_model import IMECClassifier
from utils import OpenAIWrapper, format_qa_prompt, call_batch_api, wait_for_batch_api_completion, parse_response, QwenVLLMWrapper, call_vllm_batch_api, wait_for_vllm_completion
from enum import Enum

import numpy as np
import random
import json
import os
import pickle
import logging
import torch
import time
from torch.optim import Adam

from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, task_ids, labels):
        self.texts = texts
        self.task_ids = task_ids
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'task_ids': torch.tensor(self.task_ids[idx]),  # List of task IDs relevant for this sample
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)  # List of labels for each task
        }

class StepIMEC(Enum):
    NOT_STARTED = 0
    REQUEST_JSON_CREATED = 1
    REQUEST_SENT = 2
    REQUEST_COMPLETED = 3
    RESULTS_RETRIEVED = 4
    FINAL_QUESTIONS_ARTICLE_PAIRS_GENERATED = 5
    
class IMEC:
    def __init__(self, **kwargs):
        self.corpus = kwargs.get("corpus", None)
        self.temp_folder = kwargs.get("temp_folder", "./temp")
        self.output_folder = kwargs.get("output_folder", "./output")
        self.name = kwargs.get("name", None)
        assert self.name is not None, "Name is required"
        self.temp_folder = os.path.join(self.temp_folder, self.name)
        self.output_folder = os.path.join(self.output_folder, self.name)
        os.makedirs(self.temp_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.LLM = kwargs.get("LLM", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
        self.use_vllm = kwargs.get("use_vllm", True)  # Use VLLM by default
        self.openai_api_key = kwargs.get("openai_api_key", None)
        
        # Initialize model client based on use_vllm flag
        if self.use_vllm:
            self.vllm_wrapper = QwenVLLMWrapper(
                model_name=self.LLM,
                tensor_parallel_size=kwargs.get("tensor_parallel_size", 8),
                batch_size=kwargs.get("batch_size", 512),  # Conservative default for memory efficiency
                gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
                enable_expert_parallel=kwargs.get("enable_expert_parallel", True),  # Enable for MoE models
                max_model_len=kwargs.get("max_model_len", 15000)  # Configurable max length
            )
            self.client = self.vllm_wrapper.client
        else:
            self.client = OpenAIWrapper(API_KEY=self.openai_api_key).client
        
        self.device = kwargs.get("device", "cuda")
        self.backbone = kwargs.get("backbone", "abhinand/MedEmbed-large-v0.1")
        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.num_steps = kwargs.get("num_steps", 3000000)
        default_checkpoints = list(range(2500000, 4000001, 250000))
        self.checkpoint_steps = sorted(set(kwargs.get("checkpoint_steps", default_checkpoints)))
        
        # D_pos filtering configuration
        self.enable_dpos_filtering = kwargs.get("enable_dpos_filtering", True)
        self.dpos_threshold = kwargs.get("dpos_threshold", 10)  # Questions with D_pos <= 10 will be skipped
        self.dpos_source_name = kwargs.get("dpos_source_name", self.name)  # Source folder for question terms
        
    def _log_progress(self, progress):
        with open(os.path.join(self.temp_folder, "imec_progress.txt"), "w") as f:
            f.write(str(progress))

    def _get_progress(self):
        if not os.path.exists(os.path.join(self.temp_folder, "imec_progress.txt")):
            return StepIMEC.NOT_STARTED.value
        with open(os.path.join(self.temp_folder, "imec_progress.txt"), "r") as f:
            return int(f.read())

    def _filter_questions_by_dpos(self, questions):
        """Filter questions based on D_pos values from question_terms.json"""
        if not self.enable_dpos_filtering:
            logger.info("D_pos filtering disabled, using all questions")
            # Create identity mapping: {0:0, 1:1, 2:2, ..., n:n}
            identity_mapping = {i: i for i in range(len(questions))}
            return questions, list(range(len(questions))), identity_mapping
        
        logger.info(f"Filtering questions with D_pos <= {self.dpos_threshold}")
        
        # Load question_terms.json to get D_pos values
        question_terms_file = f"results/mbqa_question_terms/{self.dpos_source_name}/question_terms.json"
        
        if not os.path.exists(question_terms_file):
            logger.warning(f"Question terms file not found: {question_terms_file}")
            logger.warning("D_pos filtering disabled due to missing file")
            return questions, list(range(len(questions))), {}
        
        try:
            with open(question_terms_file, 'r') as f:
                question_terms_data = json.load(f)
            
            results = question_terms_data.get("results", {})
            
            # Create mapping of question index to D_pos
            dpos_mapping = {}
            for question_idx_str, data in results.items():
                question_idx = int(question_idx_str)
                if question_idx < len(questions):
                    dpos_mapping[question_idx] = data.get("D_pos", 0)
            
            # Filter questions
            valid_questions = []
            valid_indices = []
            question_mapping = {}  # old_index -> new_index
            
            filtered_count = 0
            for i, question in enumerate(questions):
                d_pos = dpos_mapping.get(i, 0)
                
                if d_pos > self.dpos_threshold:
                    new_index = len(valid_questions)
                    valid_questions.append(question)
                    valid_indices.append(i)
                    question_mapping[i] = new_index
                else:
                    filtered_count += 1
                    logger.info(f"Filtered question {i} (D_pos={d_pos}): {question[:60]}...")
            
            logger.info(f"D_pos filtering results:")
            logger.info(f"  • Original questions: {len(questions):,}")
            logger.info(f"  • Filtered out: {filtered_count:,} (D_pos <= {self.dpos_threshold})")
            logger.info(f"  • Remaining: {len(valid_questions):,}")
            
            # Save the filtering results
            filter_results = {
                "config": {
                    "dpos_threshold": self.dpos_threshold,
                    "enable_dpos_filtering": self.enable_dpos_filtering
                },
                "statistics": {
                    "original_count": len(questions),
                    "filtered_count": filtered_count,
                    "remaining_count": len(valid_questions)
                },
                "question_mapping": question_mapping,  # old_index -> new_index
                "valid_indices": valid_indices  # list of original indices that were kept
            }
            
            os.makedirs(f"results/dpos_filtering", exist_ok=True)
            with open(f"results/dpos_filtering/{self.name}_dpos_filter_results.json", 'w') as f:
                json.dump(filter_results, f, indent=2)
            
            logger.info(f"D_pos filtering results saved to results/dpos_filtering/{self.name}_dpos_filter_results.json")
            
            return valid_questions, valid_indices, question_mapping
            
        except Exception as e:
            logger.error(f"Error during D_pos filtering: {e}")
            logger.warning("Falling back to using all questions")
            return questions, list(range(len(questions))), {}

    def _save_checkpoint(self, model, step):
        checkpoint_path = os.path.join(self.output_folder, f"imec_model_step_{step}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save both filtered and original questions at checkpoint
        filtered_questions_path = os.path.join(self.output_folder, "questions_filtered.json")
        if os.path.exists(filtered_questions_path):
            with open(filtered_questions_path, "r") as f:
                filtered_questions = json.load(f)
            # Save checkpoint with filtered questions (this will be the primary questions file)
            with open(os.path.join(self.output_folder, f"questions_step_{step}.json"), "w") as f:
                json.dump(filtered_questions, f)
        else:
            # Fallback: save original questions if filtered don't exist yet
            questions_path = os.path.join(self.output_folder, "questions.json")
            if os.path.exists(questions_path):
                with open(questions_path, "r") as f:
                    questions = json.load(f)
                with open(os.path.join(self.output_folder, f"questions_step_{step}.json"), "w") as f:
                    json.dump(questions, f)
                    
        logger.info(f"Checkpoint saved at step {step}: {checkpoint_path}")
        
    def collect_training_data_with_ogqg(self):
        with open(os.path.join(self.temp_folder, "deduped_questions.json"), "r") as f:
            deduped_questions = json.load(f)
            
        with open(os.path.join(self.temp_folder, "questions.json"), "r") as f:
            questions = json.load(f)
        
        with open(os.path.join(self.temp_folder, "cluster_assignments.json"), "r") as f:
            cluster_assignments = json.load(f)
            cluster_assignments = np.array(cluster_assignments)
        
        with open(os.path.join(self.temp_folder, "cluster_distances.npy"), "rb") as f:
            cluster_distances = np.load(f)
        
        logger.info(f"Start collecting training data...")
        
        # 1. Sample questions from each cluster and pack them into multiple of 20 to save cost
        if self._get_progress() < StepIMEC.REQUEST_JSON_CREATED.value:
            logger.info("Sampling questions from each cluster... ")
            linear_questions = []
            original_questions_id = []

            selected_docs_in_clusters = {}

            selected_positive_articles = {}
            docs_frequency = {}

            logger.info(f"Processing {len(deduped_questions)} clusters...")
            for i in range(len(deduped_questions)):
                if str(i) not in deduped_questions:
                    continue

                if i % 100 == 0:
                    logger.info(f"  Processed {i}/{len(deduped_questions)} clusters, {len(linear_questions)} questions so far")
                picked_questions_cluster = deduped_questions[str(i)]
                generated_questions_cluster = questions[str(i)]
                picked_questions_cluster_text = [generated_questions_cluster[q] for q in picked_questions_cluster]
                
                # Modified sampling: sample up to 300 documents from current cluster
                cluster_indices = np.where(cluster_assignments == i)[0]
                if len(cluster_indices) > 300:
                    cluster_indices = random.sample(list(cluster_indices), 300)
                else:
                    cluster_indices = list(cluster_indices)
                selected_docs_in_clusters[str(i)] = cluster_indices
                for j in range(len(picked_questions_cluster)):
                    linear_questions.append(picked_questions_cluster_text[j])
                    original_questions_id.append((i, picked_questions_cluster[j]))

                    question_id = len(linear_questions) - 1
                    
                    selected_positive_articles[question_id] = []
                    
                    for k in cluster_indices:
                        selected_positive_articles[question_id].append(k)
                        if k not in docs_frequency:
                            docs_frequency[k] = 0
                        docs_frequency[k] += 1
            
            # Modified hard negative sampling: sample from 5 nearest clusters until reaching 1000 total
            selected_hard_negatives = {}
            closest_clusters_top5 = np.argsort(cluster_distances, axis=1)[:, 1:6]

            logger.info(f"Sampling negatives from nearest clusters for {len(original_questions_id)} questions...")
            for q_id in range(len(original_questions_id)):
                if q_id % 1000 == 0:
                    logger.info(f"  Processed {q_id}/{len(original_questions_id)} questions")

                i = original_questions_id[q_id][0]

                # Calculate how many negatives we need (fixed 500 negatives)
                num_positives = len(selected_positive_articles[q_id])
                num_negatives_needed = 500

                # get nearest 5 clusters of this cluster
                closest_clusters_of_i = closest_clusters_top5[i]

                selected_hard_negatives[q_id] = []

                # Collect all available docs from 5 nearest clusters
                available_docs_per_cluster = []
                for j in range(len(closest_clusters_of_i)):
                    closest_cluster_id = str(closest_clusters_of_i[j])
                    # Skip clusters that don't have selected docs (empty clusters from CQG)
                    if closest_cluster_id not in selected_docs_in_clusters:
                        continue
                    available_docs_per_cluster.append(list(selected_docs_in_clusters[closest_cluster_id]))

                # Sample evenly from each cluster
                if len(available_docs_per_cluster) > 0:
                    docs_per_cluster = num_negatives_needed // len(available_docs_per_cluster)
                    remainder = num_negatives_needed % len(available_docs_per_cluster)

                    for cluster_idx, cluster_docs in enumerate(available_docs_per_cluster):
                        # Each cluster gets equal share, first few get +1 for remainder
                        num_to_sample = docs_per_cluster + (1 if cluster_idx < remainder else 0)

                        if len(cluster_docs) > num_to_sample:
                            sampled_docs = random.sample(cluster_docs, num_to_sample)
                        else:
                            sampled_docs = cluster_docs

                        for k in sampled_docs:
                            selected_hard_negatives[q_id].append(k)
                            if k not in docs_frequency:
                                docs_frequency[k] = 0
                            docs_frequency[k] += 1
            
            # Add easy negative sampling (200 random documents not in positive or hard negative sets)
            num_easy_negatives_needed = 200
            if len(self.corpus) > num_easy_negatives_needed: # Ensure corpus is large enough
                # Combine positive and hard negative indices for exclusion
                # selected_positive_articles[q_id] is a set
                # selected_hard_negatives[q_id] is a list until later converted to set
                excluded_indices = selected_positive_articles[q_id] | set(selected_hard_negatives[q_id])
                
                easy_negatives_sampled_count = 0
                while easy_negatives_sampled_count < num_easy_negatives_needed:
                    random_doc_index = random.randint(0, len(self.corpus) - 1)
                    if random_doc_index not in excluded_indices:
                        selected_hard_negatives[q_id].append(random_doc_index)
                        if random_doc_index not in docs_frequency:
                            docs_frequency[random_doc_index] = 0
                        docs_frequency[random_doc_index] += 1
                        easy_negatives_sampled_count += 1
                        excluded_indices.add(random_doc_index) # Add to excluded to prevent re-sampling
            
            # Convert to sets for reorganization
            selected_hard_negatives = {k: set(selected_hard_negatives[k]) for k in selected_hard_negatives}
            selected_positive_articles = {k: set(selected_positive_articles[k]) for k in selected_positive_articles}
            # reorganize the dic to be doc_index -> list of question_id

            docs_to_question_ids = {}

            logger.info(f"Reorganizing document-question pairs...")
            for q_id in range(len(original_questions_id)):
                if q_id % 1000 == 0:
                    logger.info(f"  Processed {q_id}/{len(original_questions_id)} questions")

                for doc_index in selected_positive_articles[q_id]:
                    if doc_index not in docs_to_question_ids:
                        docs_to_question_ids[doc_index] = []
                    docs_to_question_ids[doc_index].append(q_id)

                for doc_index in selected_hard_negatives[q_id]:
                    if doc_index not in docs_to_question_ids:
                        docs_to_question_ids[doc_index] = []
                    docs_to_question_ids[doc_index].append(q_id)
                    
            len(docs_to_question_ids)

            # count the frequency of each doc in the selected docs that are not multiple of 20

            num_non_20 = 0

            for doc_index in docs_to_question_ids:
                if len(docs_to_question_ids[doc_index]) % 20 != 0:
                    num_non_20 += 1
                    
            logger.info(f"Number of documents that are failed to pack to multiple of 20: {num_non_20}")
            
            docs_to_question_ids_int = {int(k): [int(vi) for vi in v] for k, v in docs_to_question_ids.items()}
            
            with open(os.path.join(self.temp_folder, "docs_to_question_ids.json"), "w") as f:
                json.dump(docs_to_question_ids_int, f)
                
            with open(os.path.join(self.output_folder, "questions.json"), "w") as f:
                json.dump(linear_questions, f)
            
            with open(os.path.join(self.temp_folder, "original_questions_id.json"), "w") as f:
                json.dump(original_questions_id, f)

            batch_jsons_get_training_data = []

            logger.info(f"Preparing batch API requests for {len(docs_to_question_ids)} documents...")
            doc_list = list(docs_to_question_ids.keys())
            for idx, doc_index in enumerate(doc_list):
                if idx % 10000 == 0:
                    logger.info(f"  Prepared {idx}/{len(doc_list)} documents")
                doc_text = self.corpus[doc_index][:10000]
                
                questions_ids_for_doc = docs_to_question_ids[doc_index]
                
                questions_texts = [linear_questions[q_id] for q_id in questions_ids_for_doc]
                
                batch_size = 20
                
                for i in range(0, len(questions_texts), batch_size):
                    batch = questions_texts[i:i+batch_size]
                    prompt = format_qa_prompt(doc_text, batch)
                    req_obj = {
                            "custom_id": f"{doc_index}_{i}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.LLM,
                                "messages": [
                                    {"role": "system", "content": "You are an expert in biomedical analysis. Answer each question with exactly 'yes' or 'no'."},
                                    {"role": "user", "content": prompt}
                                ]
                            }
                        }
                    batch_jsons_get_training_data.append(req_obj)
                    
            logger.info(f"Number of requests to get training data: {len(batch_jsons_get_training_data)}")
            
            MAX_BATCH_SIZE = 40000
            
            for i in range(0, len(batch_jsons_get_training_data), MAX_BATCH_SIZE):
                batch = batch_jsons_get_training_data[i:i+MAX_BATCH_SIZE]
                with open(os.path.join(self.temp_folder, f"batch_jsons_get_training_data_{i}.json"), "w") as f:
                    for req in batch:
                        json.dump(req, f)
                        f.write("\n")
            
            self._log_progress(StepIMEC.REQUEST_JSON_CREATED.value)
            logger.info(f"Requests to get training data prepared")
            
        
        
        if self._get_progress() < StepIMEC.REQUEST_SENT.value:
            # 2. call the batch API
            logger.info("Calling the batch API to get training data...")
            
            json_files = os.listdir(self.temp_folder)
            json_files = [f for f in json_files if f.startswith("batch_jsons_get_training_data_")]
            json_files = [os.path.join(self.temp_folder, f) for f in json_files]
            
            if self.use_vllm:
                # Use VLLM batch processing for MBQA QAEmb training data (yes/no answers)
                from utils import call_vllm_batch_api
                output_files = call_vllm_batch_api(
                    self.vllm_wrapper, 
                    json_files, 
                    self.temp_folder, 
                    "batch_results_get_training_data",
                    task_type="mbqa_answers"
                )
                # Save output file list for compatibility
                with open(os.path.join(self.temp_folder, "vllm_output_files_training_data.json"), "w") as f:
                    json.dump(output_files, f)
            else:
                call_batch_api(self.client, json_files, os.path.join(self.temp_folder, "batch_ids_get_training_data.json"))
            
            self._log_progress(StepIMEC.REQUEST_SENT.value)
            logger.info(f"Requests to get training data sent")
            
        if self._get_progress() < StepIMEC.REQUEST_COMPLETED.value:
            # 3. wait for the batch API to complete
            logger.info("Waiting for the batch API to complete...")
            
            if self.use_vllm:
                # VLLM processing is already complete (synchronous)
                logger.info("VLLM batch processing completed synchronously")
            else:
                with open(os.path.join(self.temp_folder, "batch_ids_get_training_data.json"), "r") as f:
                    batch_ids = json.load(f)
                wait_for_batch_api_completion(self.client, batch_ids)
            
            self._log_progress(StepIMEC.REQUEST_COMPLETED.value)
            logger.info(f"Requests to get training data completed")
            
        if self._get_progress() < StepIMEC.RESULTS_RETRIEVED.value:
            # 4. retrieve the results
            logger.info("Retrieving the results...")
            
            if self.use_vllm:
                # Results are already saved by VLLM wrapper, just verify they exist
                with open(os.path.join(self.temp_folder, "vllm_output_files_training_data.json"), "r") as f:
                    output_files = json.load(f)
                from utils import wait_for_vllm_completion
                wait_for_vllm_completion(output_files)
            else:
                with open(os.path.join(self.temp_folder, "batch_ids_get_training_data.json"), "r") as f:
                    batch_ids = json.load(f)
                
                for batch_id in batch_ids:
                    batch = self.client.batches.retrieve(batch_id)
                    batch_output_file_id = batch.output_file_id
                    
                    with open(os.path.join(self.temp_folder, f"batch_results_get_training_data_{batch_id}.json"), "wb") as f:
                        f.write(self.client.files.content(batch_output_file_id).content)
                    
            self._log_progress(StepIMEC.RESULTS_RETRIEVED.value)
            logger.info(f"Results retrieved")
            
        if self._get_progress() < StepIMEC.FINAL_QUESTIONS_ARTICLE_PAIRS_GENERATED.value:
            # 5. generate final questions article pairs
            logger.info("Generating final questions article pairs...")
            
            with open(os.path.join(self.temp_folder, "docs_to_question_ids.json"), "r") as f:
                docs_to_question_ids = json.load(f)
                
            with open(os.path.join(self.temp_folder, "deduped_questions.json"), "r") as f:
                deduped_questions = json.load(f)
                
            result_files = os.listdir(self.temp_folder)
            result_files = [f for f in result_files if f.startswith("batch_results_get_training_data_")]
            
            training_data = {}
            for response_file in result_files:
                for line in open(os.path.join(self.temp_folder, response_file), "r"):
                    response = json.loads(line)
                    custom_id = response["custom_id"]
                    doc_index, start_ind = custom_id.split("_")
                    doc_index = doc_index
                    start_ind = int(start_ind)
                    
                    req_answers = parse_response(response["response"]["body"]["choices"][0]["message"]["content"])
                    if len(req_answers) + start_ind > len(docs_to_question_ids[doc_index]):
                        logger.warning(f"Warning: in file {response_file} and id {custom_id}, number of answers not equal to number of questions. This error is handled but you should not see a lot of it.")
                        continue
                    if doc_index not in training_data:
                        training_data[doc_index] = {}

                    for i in range(len(req_answers)):
                        training_data[doc_index][docs_to_question_ids[doc_index][start_ind + i]] = req_answers[i]
            
            with open(os.path.join(self.temp_folder, "training_data.json"), "w") as f:
                json.dump(training_data, f)
            with open(os.path.join(self.output_folder, "questions.json"), "w") as f:
                json.dump(deduped_questions, f)
            
            self._log_progress(StepIMEC.FINAL_QUESTIONS_ARTICLE_PAIRS_GENERATED.value)
            logger.info(f"Final questions article pairs generated")
        
        logger.info(f"Finished collecting training data")
    
    def train_model(self, max_steps=None):
        logger.info(f"Start training model...")
        with open(os.path.join(self.temp_folder, "training_data.json"), "r") as f:
            training_data = json.load(f)
        
        with open(os.path.join(self.output_folder, "questions.json"), "r") as f:
            original_questions = json.load(f)
        
        # Apply D_pos filtering
        filtered_questions, valid_indices, question_mapping = self._filter_questions_by_dpos(original_questions)
        
        # Save filtered questions for later use
        with open(os.path.join(self.output_folder, "questions_filtered.json"), "w") as f:
            json.dump(filtered_questions, f)
        
        logger.info(f"Using {len(filtered_questions):,} questions after D_pos filtering")
        
        training_texts = []
        training_task_ids = []
        training_labels = []

        for doc_id in training_data:
            original_task_ids = list(training_data[doc_id].keys())
            original_task_labels = []
            for q_id in original_task_ids:
                original_task_labels.append(training_data[doc_id][q_id])
            
            # Filter tasks to only include valid questions (those not filtered by D_pos)
            filtered_task_ids = []
            filtered_task_labels = []
            
            for i, q_id in enumerate(original_task_ids):
                q_id_int = int(q_id)
                # Only keep if this question was not filtered out
                if q_id_int in question_mapping:
                    new_q_id = question_mapping[q_id_int]  # Get new index after filtering
                    filtered_task_ids.append(new_q_id)
                    filtered_task_labels.append(original_task_labels[i])
            
            # Skip if no valid tasks remain after filtering  
            if len(filtered_task_ids) == 0 or len(filtered_task_labels) == 0:
                continue  # Silently skip documents with no valid questions
                
            training_texts.append(self.corpus[int(doc_id)])
            training_task_ids.append(filtered_task_ids)
            training_labels.append(filtered_task_labels)

        if len(training_labels) == 0:
            logger.error("No training data available! Cannot train model.")
            raise ValueError("No training data found. Please check if question generation succeeded.")

        if type(training_labels[0][0]) == str:
            training_labels = [[1 if "yes" in label else 0 for label in labels]for labels in training_labels]
            
        train_texts, val_texts, train_task_ids, val_task_ids, train_labels, val_labels = train_test_split(training_texts, training_task_ids, training_labels, test_size=0.1, random_state=42)
        
        train_dataset = MyDataset(
            train_texts,
            train_task_ids,
            train_labels
        )

        val_dataset = MyDataset(
            val_texts,
            val_task_ids,
            val_labels
        )
        
        data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        label_freq = {
            0: 0,
            1: 0
        }

        for labels in training_labels:
            for label in labels:
                label_freq[label] += 1
                
        weight = torch.tensor([label_freq[0] / label_freq[1]]).to(self.device)
        model = IMECClassifier(num_labels=len(filtered_questions), backbone=self.backbone)
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weight)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        
        logger.info(f"Model training in progress...")
        num_steps = 0
        max_checkpoint = max(self.checkpoint_steps) if self.checkpoint_steps else 0
        base_target = max_steps if max_steps is not None else self.num_steps
        target_steps = max(base_target, max_checkpoint)
        pending_checkpoints = [step for step in self.checkpoint_steps if step >= num_steps]
        while True:
            if num_steps >= target_steps:
                break
            model.train()
            total_loss = 0
            losses = []
            t0 = time.time()
            for batch in data_loader:
                text = batch['text'][0]  # Assuming batch size of 1 for simplicity
                task_ids = batch['task_ids'][0]
                labels = batch['labels'][0]
                
                labels = labels.to(self.device)

                optimizer.zero_grad()
                if len(task_ids) < 2:
                    continue
                logits = model([text], task_ids=task_ids)

                # Calculate loss only for the active tasks
                loss = 0
                # print(logits.shape)
                # print(labels.shape)
                if logits.shape != labels.shape:
                    print("warning: logits shape is not equal to labels shape")
                    print(logits.shape, labels.shape)
                    continue
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_steps += 1

                while pending_checkpoints and num_steps >= pending_checkpoints[0]:
                    step_to_save = pending_checkpoints.pop(0)
                    self._save_checkpoint(model, step_to_save)

                if num_steps % 10000 == 0:
                    print(f"Average loss: {total_loss / 10000}", f"Time elapsed: {time.time() - t0}")
                    t0 = time.time()
                    losses.append(total_loss / 10000)
                    total_loss = 0
            
            model.eval()

            pred_labels = []
            gt_labels = []

            for batch in val_data_loader:
                text = batch['text'][0]  # Assuming batch size of 1 for simplicity
                task_ids = batch['task_ids'][0]
                labels = batch['labels'][0]
                
                labels = labels.to(self.device)
                if len(task_ids) == 0:
                    continue
                logits = model([text], task_ids=task_ids)
                pred_labels.append(logits.cpu().detach().numpy())
                gt_labels.append(labels.cpu().detach().numpy())
                
            pred_labels = np.concatenate(pred_labels, axis=0)
            gt_labels = np.concatenate(gt_labels, axis=0)

            logger.info(classification_report(gt_labels, pred_labels > 0))
            
        torch.save(model.state_dict(), os.path.join(self.output_folder, f"imec_model.pt"))
        
        # Save the final filtered questions as the primary questions file
        # (This will be used by evaluation scripts)
        with open(os.path.join(self.output_folder, "questions.json"), "w") as f:
            json.dump(filtered_questions, f)
            
        # Also save the original questions for reference
        with open(os.path.join(self.output_folder, "questions_original.json"), "w") as f:
            json.dump(original_questions, f)
            
        logger.info(f"Training completed with {len(filtered_questions):,} questions (filtered from {len(original_questions):,})")