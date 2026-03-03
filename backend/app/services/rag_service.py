import logging
import os
from urllib.parse import urlparse
from ..config import config
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from openai import OpenAI
from typing import List, Dict, Any

logger = logging.getLogger("nexusai.rag")

class _MockDelta:
    def __init__(self, content: str):
        self.content = content

class _MockChoice:
    def __init__(self, content: str):
        self.delta = _MockDelta(content)

class _MockChunk:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]

class RAGService:
    @staticmethod
    def _normalize_compatible_base_url(base_url: str, provider: str) -> str:
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            return normalized

        parsed = urlparse(normalized)
        if parsed.path in ("", "/"):
            fixed = f"{normalized}/v1"
            logger.warning(
                "⚠️  %s base_url missing '/v1', auto-normalized to %s",
                provider,
                fixed,
            )
            return fixed
        return normalized

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.mock_llm = os.getenv("NEXUSAI_LLM_BACKEND", "").strip().lower() == "mock"
        self.provider = config.LLM_PROVIDER.strip().lower()

        if self.mock_llm:
            self.llm_client = None
            logger.info("🧪 LLM backend: MOCK mode")
            return

        if self.provider == "ollama":
            base_url = self._normalize_compatible_base_url(config.OLLAMA_BASE_URL, "ollama")
            self.llm_client = OpenAI(
                api_key="ollama",  # Ollama doesn't need a real key
                base_url=base_url,
            )
            logger.info(f"🦙 LLM provider: Ollama ({base_url}) — model: {config.LLM_MODEL}")
        elif self.provider == "deepseek":
            if not config.DEEPSEEK_API_KEY:
                raise ValueError("Missing DEEPSEEK_API_KEY for deepseek provider")
            base_url = self._normalize_compatible_base_url(config.DEEPSEEK_BASE_URL, "deepseek")
            self.llm_client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=base_url,
            )
            logger.info(f"🤖 LLM provider: DeepSeek ({base_url}) — model: {config.LLM_MODEL}")
        elif self.provider in ("openai", "heiyucode"):
            if not config.OPENAI_API_KEY:
                raise ValueError("Missing OPENAI_API_KEY for openai/heiyucode provider")
            base_url = self._normalize_compatible_base_url(config.OPENAI_BASE_URL, self.provider)
            self.llm_client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                base_url=base_url,
            )
            provider_name = "HeiyuCode" if self.provider == "heiyucode" else "OpenAI"
            logger.info(f"🤖 LLM provider: {provider_name} ({base_url}) — model: {config.LLM_MODEL}")
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER={config.LLM_PROVIDER!r}. "
                "Use one of: openai, heiyucode, deepseek, ollama"
            )

    @staticmethod
    def _progress_bar() -> str:
        return "[████████████████████] 100%"

    @staticmethod
    def _mock_stream(answer: str):
        # Keep token-like chunks so SSE contract remains unchanged for tests/UI.
        for token in answer.split(" "):
            if not token:
                continue
            yield _MockChunk(token + " ")

    def generate_response(self, query: str, history: List[Dict[str, Any]] = None):
        if history is None:
            history = []
        
        # Step 1: Embed query
        logger.info(f"[1/4] 🔍 向量化查询中...")
        query_vector = self.embedding_service.get_embeddings([query])[0]
        logger.info(f"      {self._progress_bar()}  查询向量化完成")

        # Step 2: Search relevant chunks
        logger.info(f"[2/4] 📚 检索相关文档...")
        hits = self.vector_store.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            limit=5,
            alpha=0.7,
        )
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

        if self.mock_llm:
            if source_details:
                source_files = ", ".join(sorted(set(s["source_file"] for s in source_details)))
                answer = f"Mock answer based on sources {source_files}. Query: {query}"
            else:
                answer = f"Mock answer: no relevant context found. Query: {query}"
            logger.info("      [MOCK] LLM stream generated")
            return self._mock_stream(answer), source_details

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
