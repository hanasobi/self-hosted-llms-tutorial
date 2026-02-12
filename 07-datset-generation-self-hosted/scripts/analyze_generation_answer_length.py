# Quick check auf Full-Run Dataset
import json

lengths = []
with open('../output/qa_pairs_positive.jsonl') as f:
    for line in f:
        pair = json.loads(line)
        answer = pair.get('answer', '')
        word_count = len(answer.split())
        lengths.append(word_count)

avg_length = sum(lengths) / len(lengths)
over_60 = sum(1 for l in lengths if l > 60)
pct_over_60 = (over_60 / len(lengths)) * 100

print(f"Average: {avg_length:.1f} words")
print(f"Over 60w: {over_60} ({pct_over_60:.1f}%)")