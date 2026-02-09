import json
from collections import Counter, defaultdict

# Load the training data
with open('phase2_finetuning/data/processed/train.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

print(f"Total samples: {len(samples)}")
print()

# Count consecutive same-chunk pairs
consecutive_same_chunk = 0
for i in range(len(samples) - 1):
    current_chunk = samples[i]['metadata']['chunk_id']
    next_chunk = samples[i+1]['metadata']['chunk_id']
    if current_chunk == next_chunk:
        consecutive_same_chunk += 1

print(f"Consecutive same-chunk pairs: {consecutive_same_chunk}")
print(f"Out of {len(samples) - 1} total pairs")
print(f"Percentage: {100 * consecutive_same_chunk / (len(samples) - 1):.1f}%")
print()

# If data is perfectly grouped by chunks with 3 QA pairs per chunk,
# we'd expect about 2/3 of pairs to be consecutive (because pairs 1-2 and 2-3 
# share the same chunk, but 3-4 don't)
expected_if_grouped = (len(samples) - 1) * 2 / 3
print(f"Expected if perfectly grouped (3 per chunk): {expected_if_grouped:.0f}")
print()

# Show chunk repetition pattern (first 30 samples)
print("First 30 samples - chunk_id pattern:")
for i in range(min(30, len(samples))):
    chunk_id = samples[i]['metadata']['chunk_id']
    question_type = samples[i]['metadata']['question_type']
    print(f"  {i+1:3d}: {chunk_id:30s} ({question_type})")