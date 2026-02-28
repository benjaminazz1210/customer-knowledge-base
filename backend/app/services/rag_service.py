import logging
from ..config import config
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from openai import OpenAI
from typing import List, Dict, Any

logger = logging.getLogger("nexusai.rag")

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

        if config.LLM_PROVIDER == "ollama":
            self.llm_client = OpenAI(
                api_key="ollama",  # Ollama doesn't need a real key
                base_url=config.OLLAMA_BASE_URL,
            )
            logger.info(f"🦙 LLM provider: Ollama ({config.OLLAMA_BASE_URL}) — model: {config.LLM_MODEL}")
        else:
            self.llm_client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
            )
            logger.info(f"🤖 LLM provider: DeepSeek — model: {config.LLM_MODEL}")

    def generate_response(self, query: str):
        # 1. Embed query
        query_vector = self.embedding_service.get_embeddings([query])[0]

        # 2. Search relevant chunks
        hits = self.vector_store.search(query_vector, limit=5)

        # 3. Build context and track sources
        context_parts = []
        sources = set()
        for hit in hits:
            context_parts.append(hit["chunk_text"])
            sources.add(hit["source_file"])

        context_text = "\n\n---\n\n".join(context_parts)

        # 4. Prompt Engineering
        system_prompt = (
            "你是一个知识库智能助手。请根据下方检索到的文档内容回答用户的问题。"
            "如果文档中没有相关信息，请如实说明，不要编造答案。回答请简洁准确。\n\n"
            f"参考文档：\n{context_text}"
        )

        # 5. Call LLM
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True
        )

        return response, list(sources)
