import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import config

logger = logging.getLogger("nexusai.abtest")


class ABTestManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ABTestManager, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self._path = Path(config.experiments_config_path).expanduser()
        if not self._path.is_absolute():
            self._path = Path(__file__).resolve().parents[2] / self._path
        self._last_mtime = None
        self._experiments = self._load_experiments()
        self._results = {}

    def __init__(self):
        return None

    def _load_experiments(self) -> List[Dict[str, Any]]:
        path = self._path
        if not path.exists():
            return []
        try:
            import yaml  # type: ignore
        except Exception as exc:
            logger.warning("PyYAML unavailable, experiments disabled: %s", exc)
            return []
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        try:
            self._last_mtime = path.stat().st_mtime
        except Exception:
            self._last_mtime = None
        return data.get("experiments", []) if isinstance(data, dict) else []

    def _refresh_if_needed(self) -> None:
        try:
            current_mtime = self._path.stat().st_mtime
        except Exception:
            current_mtime = None
        if current_mtime != self._last_mtime:
            self._experiments = self._load_experiments()

    def assign_variant(self, experiment_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        self._refresh_if_needed()
        experiment = None
        for item in self._experiments:
            if item.get("id") == experiment_id and item.get("enabled", False):
                experiment = item
                break
        if not experiment:
            return None

        variants = experiment.get("variants", [])
        if not variants:
            return None

        normalized_variants = self._normalize_variants(variants)
        bucket = int(hashlib.sha256((session_id or "default").encode("utf-8")).hexdigest()[:8], 16) % 10_000
        running = 0
        for variant, weight in normalized_variants:
            running += weight
            if bucket < running:
                return variant
        return variants[-1]

    @staticmethod
    def _normalize_variants(variants: List[Dict[str, Any]]) -> List[Any]:
        weights: List[float] = []
        for variant in variants:
            raw = variant.get("ratio", variant.get("weight", 0))
            try:
                value = float(raw)
            except Exception:
                value = 0.0
            weights.append(max(value, 0.0))
        total = sum(weights)
        if total <= 0:
            even_weight = max(10_000 // max(len(variants), 1), 1)
            return [(variant, even_weight) for variant in variants]
        normalized: List[Any] = []
        allocated = 0
        for idx, (variant, weight) in enumerate(zip(variants, weights)):
            if idx == len(variants) - 1:
                portion = max(10_000 - allocated, 1)
            else:
                portion = max(int(round((weight / total) * 10_000)), 1)
                allocated += portion
            normalized.append((variant, portion))
        return normalized

    def assign_active_variant(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._refresh_if_needed()
        for experiment in self._experiments:
            if not experiment.get("enabled", False):
                continue
            variant = self.assign_variant(str(experiment.get("id", "")), session_id)
            if variant:
                return {
                    "experiment_id": experiment.get("id"),
                    "variant_id": variant.get("id"),
                    "overrides": variant.get("overrides", {}) or {},
                }
        return None

    def record_result(self, experiment_id: str, variant_id: str, metrics: Dict[str, Any]) -> None:
        key = "%s:%s" % (experiment_id, variant_id)
        self._results.setdefault(key, []).append(metrics)

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        payload = {}
        prefix = "%s:" % experiment_id
        for key, values in self._results.items():
            if not key.startswith(prefix):
                continue
            avg_confidence = 0.0
            if values:
                avg_confidence = round(
                    sum(float(item.get("confidence_score", 0.0)) for item in values) / len(values),
                    4,
                )
            payload[key[len(prefix) :]] = {
                "count": len(values),
                "average_confidence_score": avg_confidence,
                "items": values,
            }
        return payload
