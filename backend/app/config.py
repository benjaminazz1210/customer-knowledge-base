from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM providers
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llama_cloud_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:14b"

    # Vector stores and state
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "nexusai_knowledge_base"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Embeddings
    embedding_backend: str = "local"  # local | dashscope | aliyun | mock
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    dashscope_api_key: str = ""
    dashscope_embedding_model: str = "qwen3-vl-embedding"
    vector_dimension: int = 1024

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "fixed"  # fixed | semantic | parent_child
    semantic_chunk_min_size: int = 200
    semantic_chunk_max_size: int = 2000
    parent_chunk_size: int = 2000
    child_chunk_size: int = 400

    # Parsing / vision
    document_parser_backend: str = "auto"  # auto | builtin | unstructured | llamaparse
    vision_enabled: bool = True
    vision_model: str = "gpt-4o-mini"
    vision_max_images: int = 20

    # Retrieval quality
    reranker_enabled: bool = False
    reranker_backend: str = "local"  # local | api | mock
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_api_key: Optional[str] = None
    reranker_api_url: Optional[str] = None
    reranker_top_n: int = 20
    reranker_top_k: int = 5

    query_transform_enabled: bool = False
    query_transform_strategy: str = "rewrite"  # rewrite | decompose | hyde | multi_query | none
    query_transform_model: Optional[str] = None
    multi_query_count: int = 3

    eval_faithfulness_threshold: float = 0.7
    eval_relevancy_threshold: float = 0.7
    eval_context_precision_threshold: float = 0.6
    eval_context_recall_threshold: float = 0.6

    # Production hardening
    observability_enabled: bool = False
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    guardrails_enabled: bool = False
    guardrails_backend: str = "rule_based"  # rule_based | llm
    guardrails_block_message: str = (
        "I'm unable to respond to that request. Please rephrase your question about our products and services."
    )
    guardrails_check_pii: bool = True
    guardrails_check_injection: bool = True
    guardrails_check_toxicity: bool = True
    guardrails_stream_holdback_chars: int = 96
    guardrails_stream_min_release_chars: int = 48
    guardrails_presidio_enabled: bool = False

    low_confidence_enabled: bool = False
    low_confidence_strategy: str = "score_threshold"  # score_threshold | llm_judge
    low_confidence_threshold: float = 0.35
    low_confidence_message: str = (
        "I don't have enough information in the knowledge base to answer that question accurately."
    )
    low_confidence_log_path: str = "generated/low_confidence_events.jsonl"
    low_confidence_ttl_days: int = 90

    # Advanced capabilities
    self_rag_enabled: bool = False
    self_rag_max_retries: int = 2
    self_rag_critique_model: Optional[str] = None

    graph_rag_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None
    graph_entity_extraction_backend: str = "auto"  # auto | regex | spacy | llm
    graph_entity_extraction_model: Optional[str] = None
    graph_spacy_model: str = "en_core_web_sm"
    graph_max_hops: int = 2

    experiments_config_path: str = "backend/app/experiments.yml"
    admin_api_key: str = ""
    feedback_ttl_days: int = 90

    # Workflow
    workflow_output_dir: str = "generated"
    workflow_template_dir: str = "assets/templates"
    workflow_hybrid_alpha: float = 0.7
    workflow_max_context_chunks: int = 8

    # Server
    port: int = 8001
    debug: bool = True

    def __getattr__(self, name: str):
        if name.isupper():
            lowered = name.lower()
            if lowered in self.__class__.model_fields:
                return getattr(self, lowered)
        raise AttributeError(name)


config = Config()
