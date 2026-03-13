import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import config

logger = logging.getLogger("nexusai.abtest")


class ABTestManager:
    def __init__(self):
        self._experiments = self._load_experiments()
        self._results = {}

    def _load_experiments(self) -> List[Dict[str, Any]]:
        path = Path(__file__).resolve().parents[1] / "experiments.yml"
        if not path.exists():
            return []
        try:
            import yaml  # type: ignore
        except Exception as exc:
            logger.warning("PyYAML unavailable, experiments disabled: %s", exc)
            return []
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("experiments", []) if isinstance(data, dict) else []

    def assign_variant(self, experiment_id: str, session_id: str) -> Optional[Dict[str, Any]]:
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

        bucket = int(hashlib.sha256((session_id or "default").encode("utf-8")).hexdigest()[:8], 16) % 100
        running = 0
        for variant in variants:
            running += int(variant.get("ratio", 0))
            if bucket < running:
                return variant
        return variants[-1]

    def record_result(self, experiment_id: str, variant_id: str, metrics: Dict[str, Any]) -> None:
        key = "%s:%s" % (experiment_id, variant_id)
        self._results.setdefault(key, []).append(metrics)

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        payload = {}
        prefix = "%s:" % experiment_id
        for key, values in self._results.items():
            if not key.startswith(prefix):
                continue
            payload[key[len(prefix) :]] = values
        return payload
