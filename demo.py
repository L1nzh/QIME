import torch
import json
import os
import numpy as np
import sys

# Add framework directory to Python path to import modules
script_dir = os.path.dirname(__file__)
framework_path = os.path.join(script_dir, "framework")
sys.path.append(framework_path)

from imec_model import IMECClassifier

# Input text
input_text = "The patient, with a history of lung cancer, presented with chest pain; initial concern for heart attack was raised, but a contrast-enhanced CT scan revealed no coronary occlusion. It showed metastatic disease involving the mediastinum."

# Paths to model checkpoint and questions file
model_checkpoint_path = os.path.join(script_dir, "checkpoints", "QIME", "qime_model_base.pt")
questions_path = os.path.join(script_dir, "checkpoints", "QIME", "questions.json")

# Load questions (which define the embedding dimensions)
try:
    with open(questions_path, "r") as f:
        questions = json.load(f)
except FileNotFoundError:
    print(f"Error: Questions file not found at {questions_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {questions_path}")
    sys.exit(1)

# Initialize IMECClassifier model
# The number of labels (heads) must match the number of questions
# The backbone model name should be consistent with how the model was trained
try:
    model = IMECClassifier(num_labels=len(questions), backbone="abhinand/MedEmbed-large-v0.1")
except Exception as e:
    print(f"Error initializing IMECClassifier: {e}")
    sys.exit(1)

# Determine device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model state dictionary
try:
    # Use map_location to load model trained on GPU to CPU if needed, or vice-versa
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {model_checkpoint_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model state dict: {e}")
    sys.exit(1)


print("Encoding input text...")
# Encode the input text
# The IMECClassifier's forward method expects a list of texts
try:
    with torch.no_grad():
        embeddings = model([input_text])
except Exception as e:
    print(f"Error during model encoding: {e}")
    sys.exit(1)

# Convert embeddings to a 1D numpy array for easier processing
embeddings_np = embeddings.cpu().numpy().flatten()

print("\n--- Embedding Result ---")
print(f"Total dimensions: {len(embeddings_np)}")
print(f"First 20 embedding values: {embeddings_np[:20]}")


# Get indices of top 10 highest-scoring heads
# argsort returns indices that would sort the array; [-10:] gets the indices of the top 10,
# [::-1] reverses them to show highest score first.
top_10_indices = np.argsort(embeddings_np)[-10:][::-1]

print("\n--- Top 10 Highest-Scoring Heads and Their Questions ---")
for rank, idx in enumerate(top_10_indices):
    score = embeddings_np[idx]
    question = questions[idx]
    print(f"Rank {rank+1}: Head {idx} (Score: {score:.4f}) - Question: {question}")

print("\nInference complete. To run this script, execute: python run_inference.py")
