import os
import numpy as np
import torch
import faiss
import json
from sentence_transformers import SentenceTransformer
import torch.nn as nn

# Text embedding model - SentenceTransformers
# Use FP16 if a compatible GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('intfloat/e5-large-v2', device=device)


class UnifiedEmbeddingAdapter(nn.Module):
    def __init__(self, base_model, target_dim=1024):
        super().__init__()
        self.base_model = base_model
        
        # Dynamically determine the feature extraction dimension
        with torch.no_grad():
            # Create a dummy input to determine the feature dimension
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.base_model.forward_features(dummy_input)
            input_dim = torch.flatten(features, start_dim=1).shape[1]
        
        # Create linear layer with correct input dimension
        self.adapter = nn.Linear(input_dim, target_dim)
    
    def forward_text(self, texts):
        # Use sentence transformer for text
        passages = [f"passage: {text}" for text in texts]
        embeddings = embedding_model.encode(passages, normalize_embeddings=True)
        return embeddings
    
    def forward_image(self, images):
        # Use EfficientNet for images with adapter
        with torch.no_grad():
            features = self.base_model.forward_features(images)
            features = torch.flatten(features, start_dim=1)
            adapted_features = self.adapter(features)
            return adapted_features.detach().cpu().numpy().astype(np.float32)

class VectorDatabase:
    def __init__(self, dim=None):
        # Use sentence transformer dimension explicitly
        dim = embedding_model.get_sentence_embedding_dimension()
        self.dimension = dim
        
        # Use IP (Inner Product) index with the correct dimension
        self.index = faiss.IndexFlatIP(dim)
        
        self.texts = []
        self.metadata = []
        self.loaded = False
    def add(self, embeddings, texts=None, metadata_list=None):
        """
        Flexible method to add embeddings with optional texts and metadata.
        
        Args:
            embeddings (np.ndarray): Embeddings to add
            texts (list, optional): Corresponding texts
            metadata_list (list, optional): Metadata for each entry
        """
        # Ensure embeddings is a numpy array of float32
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Ensure embeddings is 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Handle metadata
        if metadata_list is None:
            metadata_list = [{}] * embeddings.shape[0]
        elif not isinstance(metadata_list, list):
            metadata_list = [metadata_list]
        
        # Handle texts
        if texts is None:
            texts = [''] * embeddings.shape[0]
        elif isinstance(texts, str):
            texts = [texts]
        
        # Verify embedding dimension matches the index
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Extend texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding, k=5):
        # Ensure query embedding is properly formatted
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.numpy()
            
        # Reshape if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query vector
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):  # Safety check
                results.append({
                    'text': self.texts[idx],
                    'score': float(distances[0][i]),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def save(self, path):
        # Ensure path exists
        os.makedirs(path, exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save texts and metadata
        with open(os.path.join(path, "data.json"), 'w') as f:
            json.dump({'texts': self.texts, 'metadata': self.metadata}, f)
    
    def load(self, path):
        # Load the index
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        # Load texts and metadata
        with open(os.path.join(path, "data.json"), 'r') as f:
            data = json.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
        
        self.loaded = True
        return True

# Function to get embeddings for a list of texts
def get_embeddings(texts):
    """Generate embeddings for a list of texts using the sentence transformer model."""
    passages = [f"passage: {text}" for text in texts]
    return embedding_model.encode(passages, normalize_embeddings=True)

# Function to get embedding for a query
def get_query_embedding(query):
    """Generate embedding for a query string."""
    formatted_query = f"query: {query}"
    return embedding_model.encode(formatted_query, normalize_embeddings=True)