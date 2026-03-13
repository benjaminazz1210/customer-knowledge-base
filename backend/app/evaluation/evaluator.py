import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import config
from ..services.rag_service import RAGService


class Evaluator:
    def __init__(self, rag_service: Optional[RAGService] = None, dataset_path: Optional[str] = None):
        self.rag_service = rag_service or RAGService()
        self.dataset_path = Path(dataset_path) if dataset_path else Path(__file__).with_name("golden_dataset.json")
        self.results_dir = Path(__file__).with_name("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.backend = "heuristic"
        try:
            import ragas  # type: ignore  # noqa: F401

            self.backend = "ragas"
        except Exception:
            self.backend = "heuristic"

    def load_dataset(self) -> List[Dict[str, Any]]:
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("golden dataset must be a list")
        return data

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        return [token for token in text.replace("\n", " ").split(" ") if token]

    def _overlap_ratio(self, left: str, right: str) -> float:
        left_tokens = set(self._tokenize(left))
        right_tokens = set(self._tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return round(len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1), 4)

    def _context_recall(self, expected_sources: List[str], actual_sources: List[str]) -> float:
        expected = set(expected_sources or [])
        actual = set(actual_sources or [])
        if not expected:
            return 1.0
        return round(len(expected & actual) / len(expected), 4)

    def evaluate_question(
        self,
        item: Dict[str, Any],
        session_id: str = "evaluation",
        overrides: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.rag_service.generate_answer_text(
            item["question"],
            session_id=session_id,
            overrides=overrides,
            experiment_id=experiment_id,
            variant_id=variant_id,
        )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        source_files = [str(source.get("source_file")) for source in sources if source.get("source_file")]
        source_text = "\n".join(str(source.get("content", "")) for source in sources)
        faithfulness = self._overlap_ratio(answer, source_text)
        answer_relevancy = self._overlap_ratio(answer, item.get("ground_truth", ""))
        context_precision = self._overlap_ratio(source_text, item.get("ground_truth", ""))
        context_recall = self._context_recall(item.get("context_sources", []), source_files)
        return {
            "question": item["question"],
            "ground_truth": item.get("ground_truth", ""),
            "expected_context_sources": item.get("context_sources", []),
            "answer": answer,
            "actual_context_sources": source_files,
            "trace_id": result.get("trace_id"),
            "confidence_score": result.get("confidence_score"),
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "metrics": {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            },
        }

    @staticmethod
    def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
        keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        return {
            key: round(
                sum(item["metrics"][key] for item in results) / max(len(results), 1),
                4,
            )
            for key in keys
        }

    def run(
        self,
        dataset: Optional[List[Dict[str, Any]]] = None,
        session_id: str = "evaluation",
        overrides: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        dataset = dataset or self.load_dataset()
        per_question = [
            self.evaluate_question(
                item,
                session_id=session_id,
                overrides=overrides,
                experiment_id=experiment_id,
                variant_id=variant_id,
            )
            for item in dataset
        ]
        aggregate = self._aggregate(per_question)
        payload = {
            "timestamp": time.time(),
            "backend": self.backend,
            "dataset_path": str(self.dataset_path),
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "aggregate": aggregate,
            "thresholds": {
                "faithfulness": config.eval_faithfulness_threshold,
                "answer_relevancy": config.eval_relevancy_threshold,
                "context_precision": config.eval_context_precision_threshold,
                "context_recall": config.eval_context_recall_threshold,
            },
            "per_question": per_question,
        }
        output_path = self.results_dir / ("evaluation_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        payload["output_path"] = str(output_path)
        return payload

    @staticmethod
    def format_metrics_table(results: Dict[str, Any]) -> str:
        aggregate = results.get("aggregate", {})
        lines = [
            "metric               value",
            "-------------------  ------",
            "faithfulness         %.4f" % aggregate.get("faithfulness", 0.0),
            "answer_relevancy     %.4f" % aggregate.get("answer_relevancy", 0.0),
            "context_precision    %.4f" % aggregate.get("context_precision", 0.0),
            "context_recall       %.4f" % aggregate.get("context_recall", 0.0),
        ]
        return "\n".join(lines)

    def passes_thresholds(self, results: Dict[str, Any]) -> bool:
        aggregate = results.get("aggregate", {})
        return all(
            [
                aggregate.get("faithfulness", 0.0) >= config.eval_faithfulness_threshold,
                aggregate.get("answer_relevancy", 0.0) >= config.eval_relevancy_threshold,
                aggregate.get("context_precision", 0.0) >= config.eval_context_precision_threshold,
                aggregate.get("context_recall", 0.0) >= config.eval_context_recall_threshold,
            ]
        )
