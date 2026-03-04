import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm

class EncoderModel:
    def __init__(self, topk=32, device=None, que_path='../data/questions.json'):
        self.topk = topk
        self.model_name = f'top{self.topk}_binary'
        self.que_path = que_path
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.__inner_encoder = SentenceTransformer('abhinand/MedEmbed-large-v0.1', device=device)
        self.one_indices = []   # Store the indices of the topk elements for each sentence  
        with open("../dataset/train_data/labeled_cor_ids.json", 'r') as f:
            self.labeled_cor_ids = json.load(f)
        self.init_questions()

    def init_questions(self):
        with open(self.que_path, 'r') as f:
            self.questions = json.load(f)
        print('Questions used:', self.que_path)
        self.question_embeddings = self.__inner_encoder.encode(
            self.questions,
            convert_to_tensor=True,
            batch_size=512,
            show_progress_bar=False
        )


    def topk_indices(self, sentences_embeddings):
        # Compute the similarity between the sentences and questions
        similarities = util.pytorch_cos_sim(sentences_embeddings, self.question_embeddings)
        topk_indices = torch.topk(similarities, self.topk, dim=1).indices

        return topk_indices

    def encode(self, sentences, batch_size=1024, sentences_embeddings_path=None, **kwargs):
        if sentences_embeddings_path is None:
            sentences_embeddings = self.__inner_encoder.encode(
                sentences, 
                convert_to_tensor=True, 
                batch_size=batch_size,
                show_progress_bar=True
            )
        else:
            sentences_embeddings = torch.as_tensor(np.load(sentences_embeddings_path), device=self.device)
            sentences_embeddings = sentences_embeddings[self.labeled_cor_ids]
        
        topk_indices = self.topk_indices(sentences_embeddings)
        self.one_indices.extend(topk_indices.cpu().numpy().tolist())
        num_sentences = sentences_embeddings.shape[0]
        num_questions = self.question_embeddings.shape[0]
        binary_embeddings = torch.zeros(num_sentences, num_questions, device=topk_indices.device)
        binary_embeddings.scatter_(1, topk_indices, 1.0)
        
        return binary_embeddings.cpu().numpy()


    # # To Save GPU Memory
    # def encode(self, sentences, batch_size=1024, sentences_embeddings_path=None, **kwargs):
    #     # 1. Load or Compute Embeddings
    #     if sentences_embeddings_path is None:
    #         sentences_embeddings = self.__inner_encoder.encode(
    #             sentences, 
    #             convert_to_tensor=True, 
    #             batch_size=batch_size,
    #             show_progress_bar=True
    #         )
    #     else:
    #         # Load to CPU first to save GPU memory
    #         sentences_embeddings = torch.as_tensor(np.load(sentences_embeddings_path))

    #     num_sentences = sentences_embeddings.shape[0]
    #     num_questions = self.question_embeddings.shape[0]

    #     # 2. Prepare Output container on CPU (System RAM)
    #     # WARNING: This still requires 165GB of System RAM if dense. 
    #     # If you run out of RAM, use scipy.sparse.csr_matrix instead.
    #     binary_embeddings_cpu = torch.zeros((num_sentences, num_questions), dtype=torch.float32)

    #     # 3. Process in chunks to avoid GPU OOM
    #     # Chunk size can be larger than batch_size, e.g., 5000 or 10000
    #     chunk_size = 50000 
        
    #     # Ensure question embeddings are on GPU
    #     self.question_embeddings = self.question_embeddings.to(self.device)

    #     print(f"Processing similarity in chunks of {chunk_size}...")
    #     for i in tqdm(range(0, num_sentences, chunk_size)):
    #         end_idx = min(i + chunk_size, num_sentences)
            
    #         # a. Move only this chunk to GPU
    #         batch_emb = sentences_embeddings[i:end_idx].to(self.device)
            
    #         # b. Calculate top-k for this chunk on GPU
    #         # Note: We assume self.topk_indices handles the sim calculation internally
    #         # We must override specific attributes or call logic directly if topk_indices 
    #         # expects the full matrix. Assuming it takes a tensor input:
    #         with torch.no_grad():
    #             batch_topk = self.topk_indices(batch_emb)
            
    #         # c. Create binary row for this chunk
    #         # Create a small zero tensor on GPU for just this batch
    #         batch_binary = torch.zeros((end_idx - i, num_questions), device=self.device)
    #         batch_binary.scatter_(1, batch_topk, 1.0)
            
    #         # d. Move result to CPU and store
    #         binary_embeddings_cpu[i:end_idx] = batch_binary.cpu()
            
    #         # e. Update one_indices list
    #         self.one_indices.extend(batch_topk.cpu().numpy().tolist())

    #         # Cleanup GPU cache
    #         del batch_emb, batch_topk, batch_binary
    #         torch.cuda.empty_cache()

    #     return binary_embeddings_cpu.numpy()