import json
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load first sample from train.jsonl
with open('phase2_finetuning/data/processed/train.jsonl', 'r') as f:
    first_line = f.readline()
    sample = json.loads(first_line)

# Get the prompt_training text
prompt_text = sample['prompt_training']
print("Prompt text (last 100 chars):")
print(prompt_text[-100:])
print()

# Tokenize it the same way as in create_dataset
tokens = tokenizer(
    prompt_text,
    truncation=True,
    max_length=1024 - 1,
    padding=False,
    return_tensors=None
)

# Add EOS token (simulating what create_dataset does)
tokens['input_ids'].append(tokenizer.eos_token_id)

print(f"Total tokens: {len(tokens['input_ids'])}")
print(f"Last 5 token IDs: {tokens['input_ids'][-5:]}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"Last token is EOS: {tokens['input_ids'][-1] == tokenizer.eos_token_id}")
print()

# Decode to see what it looks like
decoded = tokenizer.decode(tokens['input_ids'][-10:])
print(f"Last 10 tokens decoded: {decoded}")