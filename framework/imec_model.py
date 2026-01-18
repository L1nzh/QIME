from torch import nn
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import os
from tqdm import tqdm

class IMECClassifier(nn.Module):
    def __init__(self, num_labels=10000, backbone="abhinand/MedEmbed-large-v0.1"):
        super(IMECClassifier, self).__init__()
        self.transformer = SentenceTransformer(backbone, device="cpu")
        self.hidden_size = self.transformer.get_sentence_embedding_dimension()
        # Define an MLP for each classifier head
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, 8),  # First linear layer
                nn.ReLU(),                       # ReLU activation function
                nn.Linear(8, 1)                  # Output layer
            ) for _ in range(num_labels)
        ])
        
    def forward(self, texts, task_ids=None):
        embeddings = self.transformer.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        # Clone embeddings to avoid inference tensor issues
        embeddings = embeddings.clone()
        if task_ids is not None:
            logits = [self.classifiers[task_id](embeddings).squeeze(-1) for task_id in task_ids]
        else:
            logits = [classifier(embeddings) for classifier in self.classifiers]
        # Ensure all tensors in logits have at least one dimension
        logits = [l.unsqueeze(-1) if l.dim() == 0 else l for l in logits]
        logits = torch.cat(logits, dim=-1)
        
        # Only squeeze if we have more than one task to avoid converting to scalar
        if logits.shape[-1] > 1:
            logits = logits.squeeze(-1)
        else:
            # Keep as (batch_size, 1) for single task
            pass

        return logits

class IMECMTEBModelWrapper():
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))   
    
    def __init__(self, model, questions, is_binary=False, binary_threshold=0.5, is_sparse=False, use_sigmoid=False,
                 enable_task_specific_masking=False,
                 task_masking_config_file="task_specific_masking_config.json"):
        self.model = model
        self.is_binary = is_binary
        self.binary_threshold = binary_threshold
        self.is_sparse = is_sparse
        self.questions = questions
        self.use_sigmoid = use_sigmoid
        self.current_task = None

        # Task-specific masking (new approach)
        self.enable_task_specific_masking = enable_task_specific_masking
        self.task_noise_heads = self._load_task_masking_config(task_masking_config_file) if enable_task_specific_masking else None
    


    def _load_task_masking_config(self, task_masking_config_file):
        """Load task-specific masking configuration.

        Returns:
            dict: Maps task_name -> set of noise_head_indices
        """
        if not os.path.exists(task_masking_config_file):
            print(f"Task-specific masking config not found: {task_masking_config_file}")
            print(f"Task-specific masking will be disabled.")
            return None

        try:
            with open(task_masking_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            task_noise_heads = {}
            for task_name, task_config in config.items():
                noise_indices = task_config.get('noise_head_indices', [])
                task_noise_heads[task_name] = set(noise_indices)
                print(f"Loaded masking config for {task_name}: {len(noise_indices)} noise heads")

            print(f"Task-specific masking enabled for {len(task_noise_heads)} tasks")
            return task_noise_heads

        except Exception as e:
            print(f"Error loading task-specific masking config: {e}")
            print(f"Task-specific masking will be disabled.")
            return None


    
    def set_current_task(self, task_name):
        """Set the current task. Should be called by MTEB before evaluation."""
        self.current_task = task_name
        if self.enable_task_specific_masking and self.task_noise_heads:
            if task_name in self.task_noise_heads:
                print(f"Current task set: {task_name} (masking enabled: {len(self.task_noise_heads[task_name])} heads)")
            else:
                print(f"Current task set: {task_name} (masking disabled: task not in config)")
        else:
            # print(f"Current task set: {task_name}")
            pass
    


    def _apply_masking(self, binary_logits):
        """Apply head masking to binary embeddings."""
        if not self.enable_task_specific_masking or not self.task_noise_heads or not self.current_task:
            return binary_logits

        if self.current_task not in self.task_noise_heads:
            return binary_logits

        noise_heads_to_mask = self.task_noise_heads[self.current_task]
        masked_logits = binary_logits.copy()

        # No filtering - use direct indexing
        for i in range(binary_logits.shape[1]):  # For each question dimension
            if i in noise_heads_to_mask:
                # Mask this dimension (set to 0)
                masked_logits[:, i] = 0

        return masked_logits

    def encode(self, sentences, **kwargs):
        # Extract task_name from kwargs if provided (for MTEB integration)
        task_name = kwargs.get('task_name', None)
        
        # Auto-detect task name from common MTEB kwargs if not explicitly provided
        if not task_name:
            # MTEB often passes task metadata in kwargs
            if 'task_category' in kwargs:
                category = kwargs.get('task_category')
                if category:
                    # Store for potential task name inference
                    pass
            
            # Try to infer from other parameters that MTEB might pass
            for key in ['data_split', 'prompt_type']:
                if key in kwargs:
                    # These suggest MTEB is calling us, but don't give task name
                    pass
        
        if task_name and task_name != self.current_task:
            self.set_current_task(task_name)
        
        encoded = []
        batch_size = 64
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        
        desc = f"IMEC encoding"
        
        for i in tqdm(range(0, len(sentences), batch_size), 
                      desc=desc, 
                      total=total_batches):
            batch = sentences[i:i+batch_size]
            logits = self.model(batch).cpu().detach().numpy()
            
            # Apply sigmoid if requested
            if self.use_sigmoid:
                logits = self.sigmoid(logits)
            
            # Apply binary conversion if requested
            if self.is_binary:
                binary_logits = np.array(logits > self.binary_threshold, dtype=np.float32)
                # Apply head masking after binarization
                binary_logits = self._apply_masking(binary_logits)
                encoded.append(binary_logits)
            else:
                encoded.append(np.array(logits))
                
        stack_ed = np.vstack(encoded)
        return stack_ed
    
    def explain(self, embedding1, embedding2, num_explanations=None, verbose=False):
        # normalize the embeddings
        norm_embedding1 = embedding1 / np.linalg.norm(embedding1)
        norm_embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # compute the element-wise product
        product = norm_embedding1 * norm_embedding2
        
        absolute_product = np.abs(product)
        
        # count non-zero elements
        non_zero_count = np.count_nonzero(absolute_product)
        count = 0
        if verbose:
            for dim in range(len(product)):
                if product[dim] != 0:

                    count += 1
                    if num_explanations and count >= num_explanations:
                        break
        return non_zero_count
