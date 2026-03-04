import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

class EncoderModel:
    def __init__(self, topk=32, mmr_diversity=0.7, device=None, que_path='../data/questions.json'):
        self.topk = topk
        self.mmr_diversity = mmr_diversity
        self.model_name = f'top{self.topk}_{self.mmr_diversity}_binary_mmr'
        self.que_path = que_path
        # self.que_path = '../data/linear_questions.json'
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.__inner_encoder = SentenceTransformer('abhinand/MedEmbed-large-v0.1', device=device)
        # Initialize list to store indices of '1's
        self.one_indices = []
        # Load data
        self.init_questions()

    def init_questions(self):
        with open(self.que_path, 'r') as f:
            self.questions = json.load(f)
        print('Questions used:', self.que_path)
        q_embs = self.__inner_encoder.encode(
            self.questions,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False
        )
        self.question_embeddings = torch.nn.functional.normalize(q_embs, p=2, dim=1)

    def mmr_indices(self, sentences_embeddings):
        """
        Batched Maximal Marginal Relevance.
        """
        batch_size = sentences_embeddings.shape[0]
        
        # 1. Relevance Scores
        doc_q_sim = util.pytorch_cos_sim(sentences_embeddings, self.question_embeddings)
        
        selected_indices = torch.zeros((batch_size, self.topk), dtype=torch.long, device=self.device)
        
        # 2. Redundancy Tracker
        candidate_max_sim_to_selected = torch.full(
            (batch_size, len(self.questions)), -1.0, device=self.device
        )
        
        working_scores = doc_q_sim.clone()
        batch_rows = torch.arange(batch_size, device=self.device)

        # 3. Selection Loop
        for i in range(self.topk):
            redundancy_penalty = torch.clamp(candidate_max_sim_to_selected, min=0.0)
            
            mmr_scores = (self.mmr_diversity * working_scores) - \
                         ((1 - self.mmr_diversity) * redundancy_penalty)
            
            best_val, best_ind = torch.max(mmr_scores, dim=1)
            selected_indices[:, i] = best_ind
            
            # Mask selected so they aren't picked again
            working_scores[batch_rows, best_ind] = -float('inf')
            
            # Update Redundancy
            newly_selected_emb = self.question_embeddings[best_ind]
            new_sim_to_all = torch.matmul(
                newly_selected_emb.unsqueeze(1), 
                self.question_embeddings.T
            ).squeeze(1)
            
            candidate_max_sim_to_selected = torch.max(candidate_max_sim_to_selected, new_sim_to_all)

        return selected_indices

    def encode(self, sentences, batch_size=128, **kwargs):
        
        sentences_embeddings = self.__inner_encoder.encode(
            sentences, 
            convert_to_tensor=True, 
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        sentences_embeddings = torch.nn.functional.normalize(sentences_embeddings, p=2, dim=1)
        
        selected_indices = self.mmr_indices(sentences_embeddings)
        
        # Save indices
        self.one_indices.extend(selected_indices.cpu().numpy().tolist())
        
        # Create Binary Embeddings
        binary_embeddings = torch.zeros(
            sentences_embeddings.shape[0], 
            len(self.questions), 
            device=selected_indices.device
        )
        binary_embeddings.scatter_(1, selected_indices, 1.0)
        
        return binary_embeddings.cpu().numpy()