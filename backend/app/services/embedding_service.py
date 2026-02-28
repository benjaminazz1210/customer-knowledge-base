import torch
from typing import List
from ..config import config
from .scripts.qwen3_vl_embedding import Qwen3VLEmbedder

class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._init_once(*args, **kwargs)
        return cls._instance

    def _init_once(self):
        # Determine optimal device
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float32 # Default to float32 for CUDA unless specified otherwise
        else:
            self.device = "cpu"
            self.dtype = torch.float32
        
        print(f"🚀 Initializing Qwen3-VL-Embedding-2B on {self.device}...")
        self.model = Qwen3VLEmbedder(
            model_name_or_path=config.EMBEDDING_MODEL,
            dtype=self.dtype,
            device_map=self.device
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
