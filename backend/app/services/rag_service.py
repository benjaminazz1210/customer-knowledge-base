from ..config import config
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from openai import OpenAI
from typing import List, Dict, Any

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm_client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL
        )

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
            "You are a helpful assistant for NexusAI. Use the following pieces of retrieved context "
            "to answer the user's question. If you don't know the answer, just say that you don't know, "
            "don't try to make up an answer. Keep the answer concise.\n\n"
            f"Context:\n{context_text}"
        )
        
        # 5. Call LLM (DeepSeek)
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True
        )
        
        return response, list(sources)
