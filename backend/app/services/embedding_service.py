import hashlib
import math
import os
from typing import Any, Dict, List, Optional
from ..config import config

class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._init_once(*args, **kwargs)
        return cls._instance

    def _init_once(self):
        cfg_backend = os.getenv("NEXUSAI_EMBEDDING_BACKEND") or config.EMBEDDING_BACKEND
        self.backend = (cfg_backend or "local").strip().lower()
        if self.backend == "mock":
            self.device = "mock"
            self.dtype = None
            self.model = None
            print("🧪 EmbeddingService running in MOCK mode")
            return
        if self.backend in ("dashscope", "aliyun"):
            self.device = "cloud"
            self.dtype = None
            self.model = None
            self._init_dashscope()
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

    def _init_dashscope(self):
        try:
            import dashscope  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "dashscope package is required for dashscope/aliyun embedding backend"
            ) from exc

        api_key = (os.getenv("DASHSCOPE_API_KEY") or config.DASHSCOPE_API_KEY or "").strip()
        if not api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY for dashscope/aliyun embedding backend")

        dashscope.api_key = api_key
        self._dashscope = dashscope
        self._dashscope_model = (
            os.getenv("DASHSCOPE_EMBEDDING_MODEL")
            or config.DASHSCOPE_EMBEDDING_MODEL
            or config.EMBEDDING_MODEL
        )
        print(f"☁️ EmbeddingService using DashScope model: {self._dashscope_model}")

    @staticmethod
    def _extract_item_embedding(item: Any) -> Optional[List[float]]:
        if isinstance(item, dict):
            emb = item.get("embedding")
            return emb if isinstance(emb, list) else None
        emb = getattr(item, "embedding", None)
        return emb if isinstance(emb, list) else None

    @staticmethod
    def _extract_item_index(item: Any, fallback: int) -> int:
        idx = None
        if isinstance(item, dict):
            idx = item.get("text_index", item.get("index"))
        else:
            idx = getattr(item, "text_index", None)
            if idx is None:
                idx = getattr(item, "index", None)
        try:
            return int(idx)
        except Exception:
            return fallback

    def _parse_dashscope_embeddings(self, response: Any) -> List[List[float]]:
        status_code = getattr(response, "status_code", None)
        if status_code not in (None, 200):
            code = getattr(response, "code", "") or ""
            msg = getattr(response, "message", "") or ""
            raise RuntimeError(f"DashScope embedding request failed ({status_code}): {code} {msg}".strip())

        output = None
        if isinstance(response, dict):
            output = response.get("output")
        if output is None:
            output = getattr(response, "output", None)

        embeddings_raw = None
        if isinstance(output, dict):
            embeddings_raw = output.get("embeddings")
        if embeddings_raw is None:
            embeddings_raw = getattr(output, "embeddings", None)
        if not isinstance(embeddings_raw, list):
            raise RuntimeError("DashScope response missing embeddings array")

        indexed: List[tuple] = []
        for pos, item in enumerate(embeddings_raw):
            emb = self._extract_item_embedding(item)
            if not emb:
                continue
            idx = self._extract_item_index(item, pos)
            indexed.append((idx, emb))

        if not indexed:
            raise RuntimeError("DashScope response contains no valid embedding vectors")

        indexed.sort(key=lambda x: x[0])
        return [v for _, v in indexed]

    @staticmethod
    def _truncate_vectors(vectors: List[List[float]]) -> List[List[float]]:
        if not vectors:
            return vectors
        dim = config.VECTOR_DIMENSION
        if len(vectors[0]) <= dim:
            return vectors
        return [vec[:dim] for vec in vectors]

    def _dashscope_embed(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        response = self._dashscope.MultiModalEmbedding.call(
            model=self._dashscope_model,
            input=inputs
        )
        vectors = self._parse_dashscope_embeddings(response)
        if len(vectors) != len(inputs):
            raise RuntimeError(
                f"DashScope returned {len(vectors)} embeddings, expected {len(inputs)}"
            )
        return self._truncate_vectors(vectors)

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
        if self.backend in ("dashscope", "aliyun"):
            dashscope_inputs = [{"text": text} for text in texts]
            return self._dashscope_embed(dashscope_inputs)

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
        if self.backend in ("dashscope", "aliyun"):
            return self._dashscope_embed(items)

        embeddings = self.model.process(items)
        if embeddings.shape[1] > config.VECTOR_DIMENSION:
            embeddings = embeddings[:, :config.VECTOR_DIMENSION]
        return embeddings.tolist()
