import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nexusai_knowledge_base")
    
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-reasoner")
    EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"  # Multimodal 32k context
    
    VECTOR_DIMENSION = 1024  # Standardized for our collection
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

config = Config()
