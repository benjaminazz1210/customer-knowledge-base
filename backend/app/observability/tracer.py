import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from ..config import config

logger = logging.getLogger("nexusai.tracer")

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency
    Langfuse = None


class Tracer:
    def __init__(self, trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, client=None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.spans: List[Dict[str, Any]] = []
        self.client = client
        self.trace_handle = None
        if self.client is None and config.observability_enabled and Langfuse is not None:
            try:
                self.client = Langfuse(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    host=config.langfuse_host,
                )
            except Exception as exc:
                logger.warning("⚠️ LangFuse unavailable, tracer running in local mode: %s", exc)
                self.client = None
        if self.client is not None:
            try:
                self.trace_handle = self.client.trace(id=self.trace_id, metadata=self.metadata)
            except Exception:
                self.trace_handle = None

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        metadata = dict(metadata or {})
        metadata.update(kwargs)
        start = time.perf_counter()
        span: Dict[str, Any] = {
            "name": name,
            "metadata": dict(metadata),
            "start_time": time.time(),
        }
        langfuse_span = None
        if self.trace_handle is not None:
            try:
                langfuse_span = self.trace_handle.span(name=name, metadata=metadata)
            except Exception:
                langfuse_span = None
        try:
            yield span
        finally:
            span["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
            span["end_time"] = time.time()
            self.spans.append(span)
            if langfuse_span is not None:
                try:
                    langfuse_span.update(metadata=span["metadata"])
                    langfuse_span.end()
                except Exception:
                    pass

    start_span = span
    step = span

    def export(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "metadata": self.metadata, "spans": self.spans}

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                pass


class NoopTracer:
    def __init__(self, trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, client=None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.spans: List[Dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        meta = dict(metadata or {})
        meta.update(kwargs)
        yield {"name": name, "metadata": meta}

    start_span = span
    step = span

    def export(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "metadata": self.metadata, "spans": []}

    def flush(self) -> None:
        return None


def create_tracer(enabled: bool = True, trace_id: Optional[str] = None, client=None, metadata=None):
    if enabled:
        return Tracer(trace_id=trace_id, metadata=metadata, client=client)
    return NoopTracer(trace_id=trace_id, metadata=metadata, client=client)


PipelineTracer = Tracer
