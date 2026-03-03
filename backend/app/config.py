from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignore unknown keys in .env
    )

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # LlamaParse
    llama_cloud_api_key: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"

    # LLM Provider: 'openai' or 'heiyucode' or 'deepseek' or 'ollama'
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:14b"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "nexusai_knowledge_base"

    # Embedding
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    vector_dimension: int = 1024

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Phase 2: Structured parsing + vision
    document_parser_backend: str = "auto"  # auto|builtin|unstructured|llamaparse
    vision_enabled: bool = True
    vision_model: str = "gpt-4o-mini"
    vision_max_images: int = 20

    # Workflow (Phase 3/4)
    workflow_output_dir: str = "generated"
    workflow_template_dir: str = "assets/templates"
    workflow_hybrid_alpha: float = 0.7
    workflow_max_context_chunks: int = 8

    # Server
    port: int = 8001
    debug: bool = True

    # --- Backward-compatible uppercase aliases ---
    @property
    def DEEPSEEK_API_KEY(self): return self.deepseek_api_key
    @property
    def DEEPSEEK_BASE_URL(self): return self.deepseek_base_url
    @property
    def OPENAI_API_KEY(self): return self.openai_api_key
    @property
    def OPENAI_BASE_URL(self): return self.openai_base_url
    @property
    def LLAMA_CLOUD_API_KEY(self): return self.llama_cloud_api_key
    @property
    def OLLAMA_BASE_URL(self): return self.ollama_base_url
    @property
    def LLM_PROVIDER(self): return self.llm_provider
    @property
    def LLM_MODEL(self): return self.llm_model
    @property
    def QDRANT_HOST(self): return self.qdrant_host
    @property
    def QDRANT_PORT(self): return self.qdrant_port
    @property
    def COLLECTION_NAME(self): return self.collection_name
    @property
    def EMBEDDING_MODEL(self): return self.embedding_model
    @property
    def VECTOR_DIMENSION(self): return self.vector_dimension
    @property
    def CHUNK_SIZE(self): return self.chunk_size
    @property
    def CHUNK_OVERLAP(self): return self.chunk_overlap
    @property
    def DOCUMENT_PARSER_BACKEND(self): return self.document_parser_backend
    @property
    def VISION_ENABLED(self): return self.vision_enabled
    @property
    def VISION_MODEL(self): return self.vision_model
    @property
    def VISION_MAX_IMAGES(self): return self.vision_max_images
    @property
    def WORKFLOW_OUTPUT_DIR(self): return self.workflow_output_dir
    @property
    def WORKFLOW_TEMPLATE_DIR(self): return self.workflow_template_dir
    @property
    def WORKFLOW_HYBRID_ALPHA(self): return self.workflow_hybrid_alpha
    @property
    def WORKFLOW_MAX_CONTEXT_CHUNKS(self): return self.workflow_max_context_chunks

config = Config()
