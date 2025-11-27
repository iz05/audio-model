import json

results_file = "eval_audio/results/eval_results.json"

correct = 0
total_lines = 0

with open(results_file, "r") as f:
    results = json.load(f)
    total_lines = len(results)
    for line in results:
        audio = line['audio']
        question = line['question']
        gt = line['gt']
        response = line['response']
        if response in gt:
            correct += 1

accuracy = correct / total_lines if total_lines > 0 else 0.0
print(f"Accuracy: {accuracy:.4f} ({correct}/{total_lines})")