import hashlib
import math
import os
from typing import List
from ..config import config

class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._init_once(*args, **kwargs)
        return cls._instance

    def _init_once(self):
        self.backend = os.getenv("NEXUSAI_EMBEDDING_BACKEND", "").strip().lower()
        if self.backend == "mock":
            self.device = "mock"
            self.dtype = None
            self.model = None
            print("🧪 EmbeddingService running in MOCK mode")
            return

        import torch
        from .scripts.qwen3_vl_embedding import Qwen3VLEmbedder

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

    @staticmethod
    def _mock_embed_text(text: str) -> List[float]:
        dim = config.VECTOR_DIMENSION
        vec = [0.0] * dim
        tokens = (text or "").split()

        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 4):
                chunk = int.from_bytes(digest[i:i + 4], "little", signed=False)
                idx = chunk % dim
                vec[idx] += 1.0 if (chunk & 1) == 0 else -1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "mock":
            return [self._mock_embed_text(text) for text in texts]

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
        if self.backend == "mock":
            vectors = []
            for item in items:
                if "text" in item:
                    seed_text = item.get("text", "")
                else:
                    seed_text = f"[image]{item.get('image', '')}"
                vectors.append(self._mock_embed_text(seed_text))
            return vectors

        embeddings = self.model.process(items)
        if embeddings.shape[1] > config.VECTOR_DIMENSION:
            embeddings = embeddings[:, :config.VECTOR_DIMENSION]
        return embeddings.tolist()
