import base64
import io
import logging
import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI
from PIL import Image, ImageStat

from ..config import config
from .document_parser import ExtractedImage

logger = logging.getLogger("nexusai.vision")


class VisionService:
    def __init__(self):
        self.enabled = bool(config.VISION_ENABLED)
        override = os.getenv("NEXUSAI_VISION_ENABLED")
        if override is not None:
            self.enabled = override.strip().lower() in ("1", "true", "yes", "on")
        self.client = self._build_client() if self.enabled else None

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized:
            return normalized
        parsed = urlparse(normalized)
        if parsed.path in ("", "/"):
            return f"{normalized}/v1"
        return normalized

    def _build_client(self) -> Optional[OpenAI]:
        provider = config.LLM_PROVIDER.strip().lower()
        try:
            if provider == "ollama":
                return OpenAI(api_key="ollama", base_url=self._normalize_base_url(config.OLLAMA_BASE_URL))
            if provider == "deepseek" and config.DEEPSEEK_API_KEY:
                return OpenAI(
                    api_key=config.DEEPSEEK_API_KEY,
                    base_url=self._normalize_base_url(config.DEEPSEEK_BASE_URL),
                )
            if provider in ("openai", "heiyucode") and config.OPENAI_API_KEY:
                return OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    base_url=self._normalize_base_url(config.OPENAI_BASE_URL),
                )
        except Exception as exc:
            logger.warning("⚠️ VisionService LLM client init failed: %s", exc)
            return None
        return None

    @staticmethod
    def _data_url(image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _describe_with_api(self, image: ExtractedImage) -> Optional[str]:
        if not self.client:
            return None

        data_url = self._data_url(image.image_bytes, image.mime_type or "image/png")
        prompt = (
            "你是文档视觉理解助手。请用中文客观描述这张文档内图片/架构图/图表的关键信息，"
            "输出1-2句话，避免臆测。"
        )
        if image.context:
            prompt += f"\n上下文：{image.context}"

        try:
            resp = self.client.chat.completions.create(
                model=config.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0.2,
            )
            if not resp or not getattr(resp, "choices", None):
                return None
            return (resp.choices[0].message.content or "").strip() or None
        except Exception as exc:
            logger.warning("⚠️ Vision API describe failed (%s): %s", image.image_id, exc)
            return None

    @staticmethod
    def _describe_fallback(image: ExtractedImage) -> str:
        try:
            with Image.open(io.BytesIO(image.image_bytes)) as img:
                rgb = img.convert("RGB")
                stat = ImageStat.Stat(rgb)
                mean = stat.mean or [0, 0, 0]
                width, height = rgb.size
                desc = (
                    f"检测到文档图片（约 {width}x{height} 像素），整体色调接近 "
                    f"RGB({int(mean[0])},{int(mean[1])},{int(mean[2])})。"
                )
        except Exception:
            desc = "检测到文档图片，建议结合上下文进行人工复核。"

        if image.context:
            desc += f" 所在上下文：{image.context[:160]}。"
        return desc

    def describe_images(
        self,
        images: List[ExtractedImage],
        source_file: str,
    ) -> List[Dict]:
        if not images:
            return []

        max_images = max(0, int(config.VISION_MAX_IMAGES))
        selected = images[:max_images] if max_images > 0 else []

        chunks: List[Dict] = []
        for idx, image in enumerate(selected):
            description = self._describe_with_api(image) if self.enabled else None
            if not description:
                description = self._describe_fallback(image)

            location = f"page={image.page}" if image.page else f"slide={image.slide}" if image.slide else "doc"
            chunk_text = f"[图片描述] {description}"
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "metadata": {
                        "section_type": "image_description",
                        "image_id": image.image_id,
                        "image_index": idx,
                        "image_mime_type": image.mime_type,
                        "image_location": location,
                        "source_hint": image.source_hint,
                        "heading_path": [f"{source_file}:{location}"],
                        "heading_level": 1,
                    },
                }
            )
        return chunks
