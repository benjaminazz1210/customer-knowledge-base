import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI

from ..config import config

logger = logging.getLogger("nexusai.llm")


def normalize_compatible_base_url(base_url: str, provider: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return normalized

    parsed = urlparse(normalized)
    if parsed.path in ("", "/"):
        fixed = f"{normalized}/v1"
        logger.warning("⚠️ %s base_url missing '/v1', auto-normalized to %s", provider, fixed)
        return fixed
    return normalized


def is_mock_backend() -> bool:
    return os.getenv("NEXUSAI_LLM_BACKEND", "").strip().lower() == "mock"


def create_llm_client() -> Optional[OpenAI]:
    if is_mock_backend():
        return None

    provider = config.llm_provider.strip().lower()
    if provider == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=normalize_compatible_base_url(config.ollama_base_url, "ollama"),
        )
    if provider == "deepseek":
        if not config.deepseek_api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY for deepseek provider")
        return OpenAI(
            api_key=config.deepseek_api_key,
            base_url=normalize_compatible_base_url(config.deepseek_base_url, "deepseek"),
        )
    if provider in ("openai", "heiyucode"):
        if not config.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY for openai/heiyucode provider")
        return OpenAI(
            api_key=config.openai_api_key,
            base_url=normalize_compatible_base_url(config.openai_base_url, provider),
        )
    raise ValueError(
        "Unsupported LLM_PROVIDER={!r}. Use one of: openai, heiyucode, deepseek, ollama".format(
            config.llm_provider
        )
    )


def complete_text(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    try:
        client = create_llm_client()
        if client is None:
            return messages[-1]["content"] if messages else ""

        response = client.chat.completions.create(
            model=model or config.llm_model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        choice = response.choices[0]
        content = getattr(choice.message, "content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    text = getattr(item, "text", "")
                    if text:
                        parts.append(str(text))
            return "".join(parts).strip()
        return str(content or "").strip()
    except Exception as exc:
        logger.warning("⚠️ LLM completion fallback triggered: %s", exc)
        return messages[-1]["content"] if messages else ""
