import torch
from typing import List
from ..config import config
from .scripts.qwen3_vl_embedding import Qwen3VLEmbedder

class EmbeddingService:
    def __init__(self):
        # Initialize the Qwen3VLEmbedder model
        # For Mac (Metal), we use torch.float16 if possible, otherwise float32
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        
        print(f"🚀 Initializing Qwen3-VL-Embedding-2B on {device}...")
        self.model = Qwen3VLEmbedder(
            model_name_or_path=config.EMBEDDING_MODEL,
            torch_dtype=dtype,
            device_map=device
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Qwen3-VL-Embedding expects a list of dictionaries with "text" or "image" keys
        inputs = [{"text": text} for text in texts]
        
        # Process the inputs to get embeddings
        # The model returns a numpy array
        embeddings = self.model.process(inputs)
        
        # MRL Support: if we want to truncate to 1024 (optional, but follows our config)
        # Assuming the base dimension is larger (e.g., 1024 or 4096)
        # For now, we return as is or slice if dimension in config is smaller than model default
        if embeddings.shape[1] > config.VECTOR_DIMENSION:
            embeddings = embeddings[:, :config.VECTOR_DIMENSION]
            
        return embeddings.tolist()

    def get_multimodal_embeddings(self, items: List[dict]) -> List[List[float]]:
        """
        Supports items like {'text': '...'} or {'image': 'url/path'}
        """
        embeddings = self.model.process(items)
        if embeddings.shape[1] > config.VECTOR_DIMENSION:
            embeddings = embeddings[:, :config.VECTOR_DIMENSION]
        return embeddings.tolist()
