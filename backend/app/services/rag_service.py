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

    @staticmethod
    def _progress_bar() -> str:
        return "[████████████████████] 100%"

    def generate_response(self, query: str, history: List[Dict[str, Any]] = None):
        if history is None:
            history = []
        
        # Step 1: Embed query
        logger.info(f"[1/4] 🔍 向量化查询中...")
        query_vector = self.embedding_service.get_embeddings([query])[0]
        logger.info(f"      {self._progress_bar()}  查询向量化完成")

        # Step 2: Search relevant chunks
        logger.info(f"[2/4] 📚 检索相关文档...")
        hits = self.vector_store.search(query_vector, limit=5)
        logger.info(f"      {self._progress_bar()}  检索到 {len(hits)} 条相关片段")

        # Step 3: Build context and track sources
        logger.info(f"[3/4] 📝 构建提示词...")
        context_parts = []
        source_details = []
        
        for hit in hits:
            payload = hit["payload"]
            score = hit["score"]
            context_parts.append(payload["chunk_text"])
            source_details.append({
                "source_file": payload["source_file"],
                "content": payload["chunk_text"],
                "score": float(score)
            })

        context_text = "\n\n---\n\n".join(context_parts)
        system_prompt = (
            "你是一个知识库智能助手。请根据下方检索到的文档内容回答用户的问题。"
            "如果文档中没有相关信息，请如实说明，不要编造答案。回答请简洁准确。\n\n"
            f"参考文档：\n{context_text}"
        )
        
        log_sources = list(set([s["source_file"] for s in source_details]))
        logger.info(f"      {self._progress_bar()}  来源文件: {log_sources}")

        # Step 4: Call LLM (streaming)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Embed chat history
        for msg in history:
            role = "assistant" if msg.get("isAi") else "user"
            messages.append({"role": role, "content": msg.get("text", "")})
            
        messages.append({"role": "user", "content": query})

        logger.info(f"[4/4] 🤖 调用 LLM 生成回答 (model={config.LLM_MODEL})...")
        response = self.llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            stream=True
        )
        logger.info(f"      {self._progress_bar()}  LLM 开始流式输出...")

        return response, source_details
