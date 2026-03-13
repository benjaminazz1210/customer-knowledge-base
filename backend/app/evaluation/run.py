import argparse
import sys

from .evaluator import Evaluator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run NexusAI RAG evaluation.")
    parser.add_argument("--dataset", default="", help="Path to golden dataset JSON.")
    parser.add_argument("--session-id", default="evaluation", help="Session id used for evaluation requests.")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit non-zero if aggregate metrics are below configured thresholds.",
    )
    args = parser.parse_args(argv)

    evaluator = Evaluator(dataset_path=args.dataset or None)
    results = evaluator.run(session_id=args.session_id)
    print(Evaluator.format_metrics_table(results))
    print("\nresults_json=%s" % results.get("output_path"))

    if args.fail_on_threshold and not evaluator.passes_thresholds(results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
