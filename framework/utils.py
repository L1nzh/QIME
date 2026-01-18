import re
import json
import time
from openai import OpenAI
import logging
import os
from transformers import AutoTokenizer
import numpy as np
from vllm import LLM, SamplingParams
import torch
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def parse_response(response):
    try:
        # Split the response into lines
        lines = response.split('\n')
        
        # Initialize an empty list to store questions
        questions = []
        
        # Loop through each line and extract the questions
        for line in lines:
            # Use regex to extract the question part
            match = re.match(r'\d+\.\s+(.*)', line)
            if match:
                questions.append(match.group(1))
        
        return questions
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return []
    
    
def format_qa_prompt(chunk, questions):
    txt = '''Evaluate the following text chunk based on the yes/no questions provided.

**Text Chunk:** '''
    txt += chunk + "\n\n"
    txt += '''**Questions:**
'''
    for i, question in enumerate(questions):
        txt += f"{i + 1}. {question}\n"
    
    txt += '''

**Instruction for the model:** Please read the provided text chunk and answer each of the questions with either "yes" or "no".Format the responses as follows:
1. yes/no
2. yes/no'''
        
    return txt

def call_batch_api(client, json_files, request_id_save_file_name):
    batch_ids = []
    for json_file in json_files:
        batch_file = client.files.create(
            file=open(json_file, 'rb'),
            purpose="batch"
        )
        batch_file_id = batch_file.id
        batch_request = client.batches.create(
            input_file_id = batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Batch request for file " + json_file
            }
        )
        batch_ids.append(batch_request.id)
    
    with open(request_id_save_file_name, "w") as f:
        json.dump(batch_ids, f)
        
        
def wait_for_batch_api_completion(client, batch_ids):
    for batch_id in batch_ids:
        while True:
            batch = client.batches.retrieve(batch_id)
            if batch.status == "completed":
                break
            time.sleep(60)

def call_vllm_batch_api(vllm_wrapper, json_files, temp_folder, output_prefix, task_type="question_generation"):
    """Process batch requests using VLLM wrapper"""
    all_requests = []
    
    # Load all requests from json files
    for json_file in json_files:
        with open(json_file, "r") as f:
            for line in f:
                all_requests.append(json.loads(line))
    
    # Process all requests at once with appropriate task type
    output_files = vllm_wrapper.process_batch_requests(all_requests, temp_folder, output_prefix, task_type)
    
    # Return output file paths for compatibility with existing code
    return [os.path.join(temp_folder, f) for f in output_files]

def wait_for_vllm_completion(output_files):
    """VLLM processing is synchronous, so no waiting needed"""
    # Check if all output files exist
    for output_file in output_files:
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"VLLM output file not found: {output_file}")
    logger.info("VLLM batch processing completed")
            

class QwenVLLMWrapper:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QwenVLLMWrapper, cls).__new__(cls)
            cls._instance._initialize_model(**kwargs)
        return cls._instance
    
    def _initialize_model(self, **kwargs):
        """Initialize the VLLM Qwen model"""
        self.model_name = kwargs.get("model_name", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 8)
        self.batch_size = kwargs.get("batch_size", 512)  # Reduced for memory efficiency
        self.enable_expert_parallel = kwargs.get("enable_expert_parallel", True)  # Enable by default for MoE models
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)  # Reduced for memory efficiency
        self.max_model_len = kwargs.get("max_model_len", 15000)  # Configurable max length
        
        logger.info(f"Initializing VLLM Qwen model: {self.model_name}")
        logger.info(f"Expert parallel enabled: {self.enable_expert_parallel}")
        
        # VLLM configuration optimized for CQG/MBQA workloads with conservative memory usage
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "auto",
            
            # Context length optimized for medical articles
            "max_model_len": self.max_model_len,  # Configurable max length
            
            # Conservative memory settings
            "gpu_memory_utilization": self.gpu_memory_utilization,  # Configurable utilization
            "swap_space": 8,  # Reduced swap space
            
            # Batch processing optimizations for memory
            "max_num_seqs": min(self.batch_size // 4, 64),  # Much smaller concurrent sequences
            "max_num_batched_tokens": min(self.batch_size // 4, 64) * 512,  # Smaller batch tokens
            
            # Performance optimizations  
            "enforce_eager": kwargs.get("enforce_eager", True),   # Default to eager mode to avoid compilation
            "enable_prefix_caching": True,  # Cache common prompt prefixes
            "disable_log_stats": False,
            "disable_custom_all_reduce": kwargs.get("disable_custom_all_reduce", False),  # Configurable compilation
            
            # Additional optimizations for FP8 model
            "quantization": None,  # FP8 is already quantized
            "enable_lora": False,  # Not using LoRA
        }
        
        # Add expert parallel for MoE models (Qwen3 is a MoE model)
        if self.enable_expert_parallel:
            llm_kwargs["enable_expert_parallel"] = True
            logger.info("Expert parallelism enabled for MoE model optimization")
        
        self.llm = LLM(**llm_kwargs)
        
        # Sampling parameters optimized for different use cases
        # Will be updated dynamically based on task type
        self.question_generation_params = SamplingParams(
            temperature=0.7,        # Balanced creativity for question generation
            top_p=0.9,             # High diversity for varied questions
            top_k=50,              # Reasonable variety
            max_tokens=4096,       # Increased for UMLS context + 10 questions + formatting
            stop=None,             # Remove all stop sequences to allow complete generation
            frequency_penalty=0.1,  # Light penalty to prevent repetitive questions
            presence_penalty=0.05,  # Very light penalty to encourage diverse topics
            repetition_penalty=1.05 # Minimal repetition control
        )
        
        self.quality_probing_params = SamplingParams(
            temperature=0.3,        # Lower temperature for consistent yes/no answers
            top_p=0.8,             # Focused sampling
            top_k=20,              # More deterministic
            max_tokens=200,        # Short answers only
            stop=["\n\n", "**", "---", "Question", "Answer"],
            frequency_penalty=0.0,  # No penalty needed for short answers
            presence_penalty=0.0,
            repetition_penalty=1.0
        )
        
        self.mbqa_answer_params = SamplingParams(
            temperature=0.2,        # Very consistent for training data
            top_p=0.7,             # Focused sampling
            top_k=10,              # Highly deterministic
            max_tokens=800,        # Enough for 100+ yes/no answers (each ~5-6 tokens)
            stop=["\n\n", "**", "---", "Question"],
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0
        )
        
        # Default to question generation parameters
        self.sampling_params = self.question_generation_params
        
        logger.info(f"VLLM Qwen model initialized successfully")
        logger.info(f"Configuration: {self.tensor_parallel_size} GPUs, batch_size={self.batch_size}, max_len={llm_kwargs['max_model_len']}")
        
        # Create mock client for compatibility
        self.client = self
    
    def process_batch_requests(self, requests: List[Dict[str, Any]], temp_folder: str, output_prefix: str, task_type: str = "question_generation") -> List[str]:
        """Process batch requests using VLLM and save results in OpenAI batch format"""
        logger.info(f"Processing {len(requests)} requests with VLLM batch size {self.batch_size}")
        logger.info(f"Task type: {task_type}")
        
        # Select appropriate sampling parameters based on task type
        if task_type == "question_generation":
            sampling_params = self.question_generation_params
            logger.info("Using question generation parameters (creative, diverse)")
        elif task_type == "quality_probing":
            sampling_params = self.quality_probing_params
            logger.info("Using quality probing parameters (consistent yes/no)")
        elif task_type == "mbqa_answers":
            sampling_params = self.mbqa_answer_params
            logger.info("Using MBQA answer parameters (deterministic yes/no)")
        else:
            sampling_params = self.question_generation_params
            logger.warning(f"Unknown task type {task_type}, using default parameters")
        
        # Extract messages and custom_ids for chat format
        message_batches = []
        custom_ids = []
        for req in requests:
            messages = req["body"]["messages"]
            message_batches.append(messages)
            custom_ids.append(req["custom_id"])
        
        # Adaptive batch size based on task type
        actual_batch_size = self.batch_size
        if task_type == "question_generation":
            # Large prompts (42 samples), reduce batch size
            actual_batch_size = min(self.batch_size // 2, 256)
        elif task_type == "quality_probing":
            # Medium prompts, normal batch size
            actual_batch_size = self.batch_size
        elif task_type == "mbqa_answers":
            # Small prompts, can increase batch size
            actual_batch_size = min(self.batch_size * 2, 1024)
        
        logger.info(f"Using adaptive batch size: {actual_batch_size} for {task_type}")
        
        # Process in batches
        all_responses = []
        for i in range(0, len(message_batches), actual_batch_size):
            batch_messages = message_batches[i:i + actual_batch_size]
            batch_custom_ids = custom_ids[i:i + actual_batch_size]
            
            logger.info(f"Processing batch {i//actual_batch_size + 1}/{(len(message_batches)-1)//actual_batch_size + 1} "
                       f"({len(batch_messages)} requests)")
            
            # Generate responses using chat format for better instruction following
            responses = self.llm.chat(batch_messages, sampling_params)
            
            # Format responses in OpenAI batch format
            for j, response in enumerate(responses):
                custom_id = batch_custom_ids[j]
                output_text = response.outputs[0].text
                
                # Format as OpenAI batch response
                formatted_response = {
                    "custom_id": custom_id,
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": output_text
                                    }
                                }
                            ]
                        }
                    }
                }
                all_responses.append(formatted_response)
        
        # Save results to temp files
        output_files = []
        for i in range(0, len(all_responses), 40000):  # Match MAX_BATCH_SIZE
            batch_responses = all_responses[i:i + 40000]
            output_file = os.path.join(temp_folder, f"{output_prefix}_{i}.json")
            
            with open(output_file, "w") as f:
                for response in batch_responses:
                    json.dump(response, f)
                    f.write("\n")
            
            output_files.append(f"{output_prefix}_{i}.json")
        
        return output_files

class OpenAIWrapper:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OpenAIWrapper, cls).__new__(cls)
            API_KEY = kwargs.get("API_KEY", None)
            if API_KEY is not None:
                os.environ["OPENAI_API_KEY"] = API_KEY
            cls._instance.client = OpenAI()
            try:
                test_response = cls._instance.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, how are you?"}
                    ]
                )
                logger.info(f"Connected to OpenAI")
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                logger.error(f"Error connecting to OpenAI: {e}\nNote we currently only support OpenAI models. Make sure you have setup the API key correctly or have passed the API key in as parameter. Make sure you have sufficient credits. Refer to https://platform.openai.com/docs/quickstart for more details on how to setup the API key.")
                raise e
        return cls._instance
    
class BagOfTokenEncoder:
    def __init__(self, tokenizer="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
    def encode(self, sentences, **kwargs):
        input_ids = self.tokenizer(sentences, add_special_tokens=False, padding=False, truncation=True, return_tensors="np")['input_ids']
        total_dims = self.tokenizer.vocab_size
        output = np.zeros((len(sentences), total_dims), dtype=np.float32)
        for i in range(len(sentences)):
            # Ensure input_ids are integers
            unique, counts = np.unique(input_ids[i].astype(np.int32), return_counts=True)
            # Ensure unique indices are within bounds
            valid_indices = unique < total_dims
            output[i, unique[valid_indices]] = counts[valid_indices]
        return output

def get_adaptive_t(cluster_size):
    """Calculate dynamic t-value based on cluster size."""
    if cluster_size < 1000:      return 2
    elif cluster_size < 2000:    return 4  
    elif cluster_size < 3000:    return 6
    else:                        return 8