import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class EncoderModel:
    def __init__(self, topk=32, device=None):
        self.topk = topk
        self.model_name = f'top{self.topk}_cosine_mmr'
        self.que_path = '../data/questions.json'
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.__inner_encoder = SentenceTransformer('abhinand/MedEmbed-large-v0.1', device=device)
        self.one_indices = []   # Store the indices of the topk elements for each sentence  
        self.init_questions()

    def init_questions(self):
        with open(self.que_path, 'r') as f:
            self.questions = json.load(f)
        self.question_embeddings = self.__inner_encoder.encode(
            self.questions,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False
        )


    def topk_indices(self, sentences_embeddings):
        # Compute the similarity between the sentences and questions
        similarities = util.pytorch_cos_sim(sentences_embeddings, self.question_embeddings)
        self.latest_cosine_similarity_matrix = similarities
        topk_indices = torch.topk(similarities, self.topk, dim=1).indices

        return topk_indices

    def encode(self, sentences, batch_size=256, **kwargs):
        
        sentences_embeddings = self.__inner_encoder.encode(
            sentences, 
            convert_to_tensor=True, 
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        topk_indices = self.topk_indices(sentences_embeddings)
        self.one_indices.extend(topk_indices.cpu().numpy().tolist())
        num_sentences = sentences_embeddings.shape[0]
        num_questions = self.question_embeddings.shape[0]
        cosine_embeddings = torch.zeros(num_sentences, num_questions, device=topk_indices.device)
        topk_cosine_similarity_values = self.latest_cosine_similarity_matrix.gather(1, topk_indices)
        cosine_embeddings.scatter_(
            1,
            topk_indices,
            topk_cosine_similarity_values
        )
        
        return cosine_embeddings.cpu().numpy()