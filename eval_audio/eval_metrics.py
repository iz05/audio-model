import argparse
import json
from difflib import SequenceMatcher


DEFAULT_RESULTS_FILE = "eval_audio/results/eval_results.json"


def normalize_text(s: str) -> str:
    """Normalization: lowercase + strip extra whitespace"""
    return " ".join(s.lower().split())


def normalize_list(gt: list) -> list:
    """Normalize list of strings"""
    return [normalize_text(g) for g in gt]


def normalize_gt(gt) -> list:
    """Normalize gt and return as list"""
    if isinstance(gt, list):
        return normalize_list(gt)
    if isinstance(gt, str):
        return [normalize_text(gt)]
    return []


def is_exact_match(gt: str, response: str) -> bool:
    """Exact string match after normalization"""
    return normalize_text(response) in gt


def is_near_match(gt: str, response: str) -> bool:
    """
    Near match: ground truth appears as substring in response
    after normalization
    """
    norm_response = normalize_text(response)

    for gt_str in gt:
        if gt_str in norm_response and gt_str:
            return True
    return False


def is_semantic_match(gt: str, response: str, threshold: float = 0.8) -> bool:
    """
    Semantic match via string similarity heuristic.
    Uses difflib.SequenceMatcher ratio on normalized text.
    This is NOT true semantic embedding, but a lightweight approximation.
    """
    norm_response = normalize_text(response)
    if not gt or not norm_response:
        return False
    
    for gt_str in gt:
        if SequenceMatcher(None, gt_str, norm_response).ratio() > threshold:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy metrics from AQA eval results."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to JSON results file (default: {DEFAULT_RESULTS_FILE})",
    )
    args = parser.parse_args()

    results_file = args.results_file
    exact_correct = near_correct = loose_correct = 0
    total_lines = 0

    with open(results_file, "r") as f:
        results = json.load(f)
        total_lines = len(results)
    for line in results:
        gt = line.get("gt", "") or ""
        response = line.get("response", "") or ""
        norm_gt = normalize_gt(gt)

        # Accuracy checks
        if is_exact_match(norm_gt, response):
            exact_correct += 1
        
        if is_near_match(norm_gt, response):
            near_correct += 1
            loose_correct += 1
        elif is_semantic_match(norm_gt, response):
            loose_correct += 1

    print(f"Total examples: {total_lines}")
    print(f"Exact accuracy:   {exact_correct/total_lines:.4f}")
    print(f"Loose accuracy:   {near_correct/total_lines:.4f}")
    print(f"Looser accuracy:  {loose_correct/total_lines:.4f}")

if __name__ == "__main__":
    main()