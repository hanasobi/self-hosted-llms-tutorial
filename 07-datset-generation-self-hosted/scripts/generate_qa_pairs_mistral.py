import random
import requests
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict


script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

# Configuration
INPUT_FILE = project_root / "data" / "processed" / "chunks_token_based.jsonl"
OUTPUT_FILE = script_dir.parent / "output" / "qa_pairs_positive.jsonl"
ERROR_FILE = script_dir.parent / "output" / "generation_errors.jsonl"
NUM_SAMPLES = 0  # Number of chunks to sample for QA generation

# Model settings
MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
TEMPERATURE = 0.7
MAX_TOKENS = 1500

# System Prompt
SYSTEM_PROMPT = """You are an expert in AWS documentation. Your task is to create three high-quality question-answer pairs based on a given text passage.

Rules for questions:
- Create three different question types: one factual question, one conceptual question, and one comparison or relationship question
- Questions should be realistic - how actual AWS users would ask
- All answers must be completely answerable from the given context
- Questions should be in English

Rules for answers:
- Extract and provide ALL relevant information from the context
- NEVER add information not explicitly stated in the context
- NEVER use external knowledge or your training data - only use what's in the given context
- Be as detailed as the context allows - short context = short answer, detailed context = detailed answer
- Write in complete, helpful sentences as if answering a colleague
- If comparing items, ONLY compare aspects explicitly mentioned in the context
- If the context doesn't provide enough information for a comparison, create a different question type instead
- Answers should be in English

Generate exactly three question-answer pairs from the given text.

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY a JSON array
- NO markdown code blocks
- NO explanatory text before or after
- NO extra whitespace or newlines outside the JSON

Required JSON structure:
[
  {"question": "...", "answer": "...", "type": "factual"},
  {"question": "...", "answer": "...", "type": "conceptual"},
  {"question": "...", "answer": "...", "type": "comparison"}
]

Question Type Rules:
- factual: Direct information extraction ("What is X?", "When does Y happen?")
- conceptual: Understanding or process questions ("How does X work?", "Why would you use Y?")
- comparison: Relationships or contrasts ("What's the difference between X and Y?", "How does X compare to Y?")
"""

def load_chunks(filepath: str) -> List[Dict]:
    """Load all chunks from JSONL file."""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def save_qa_pair(output_file: str, qa_pair: Dict, chunk: Dict):
    """Append a QA pair to the output file."""
    chunk_metadata = chunk.get('metadata', {})
    record = {
        'question': qa_pair['question'],
        'answer': qa_pair['answer'],
        'question_type': qa_pair['type'],
        'metadata': {
            'service': chunk_metadata.get('service'),
            'doc_type': chunk_metadata.get('doc_type'),
            'source_file': chunk_metadata.get('source_file'),
            'chunk_id': chunk_metadata.get('chunk_id'),
            'source_tokens': chunk_metadata.get('chunk_end', 0) - chunk_metadata.get('chunk_start', 0),
            'generated_at': datetime.now().isoformat()
        }
    }
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# Load and sample chunks
all_chunks = load_chunks(INPUT_FILE)
#chunks = random.sample(all_chunks, NUM_SAMPLES)
chunks = all_chunks

# Stats
stats = {
    "total_chunks": 0,
    "successful": 0,
    "json_errors": 0,
    "api_errors": 0
}

start_time = time.time()

# Generate via vLLM
for idx, chunk in enumerate(chunks):
    
    stats["total_chunks"] += 1

    try:

      # API CALL
      full_prompt = f"{SYSTEM_PROMPT}\n\nText passage:\n{chunk['content']}"
      response = requests.post(
          "http://localhost:8000/v1/chat/completions",
          headers={"Content-Type": "application/json"},
          json={
              "model": MODEL,
              "messages": [
                  {"role": "user", "content": full_prompt}
              ],
              "temperature": TEMPERATURE,
              "max_tokens": MAX_TOKENS
          }
      )

      # Check for API errors
      if response.status_code != 200:
        stats["api_errors"] += 1
        error_entry = {
            "chunk_id": chunk.get('chunk_id', f'unknown_{idx}'),
            "error_type": "api_error",
            "status_code": response.status_code,
            "response": response.text[:500],
            "timestamp": datetime.now().isoformat()
        }
        with open(ERROR_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_entry) + '\n')
        continue

      # Extract response
      content = json.loads(response.text)["choices"][0]["message"]["content"].strip()

      # Clean markdown if present
      if content.startswith("```json"):
          content = content[7:]
      elif content.startswith("```"):
          content = content[3:]
      if content.endswith("```"):
          content = content[:-3]
      content = content.strip()
    
      # Try to parse JSON
      try:
        qa_pairs = json.loads(content)
        # Validate structure
        if not isinstance(qa_pairs, list) or len(qa_pairs) != 3:
            raise ValueError(f"Expected list of 3 items, got {type(qa_pairs)} with {len(qa_pairs)} items")
        
        # Validate each pair
        for pair in qa_pairs:
            required_keys = {'question', 'answer', 'type'}
            if not all(k in pair for k in required_keys):
                raise ValueError(f"Missing required keys. Got: {pair.keys()}")
        
        # SUCCESS - Write to output
        stats["successful"] += 1
        for i, pair in enumerate(qa_pairs):
            save_qa_pair(OUTPUT_FILE, pair, chunk)
        
        print(f"✓ Chunk {idx+1}/{len(chunks)}: Generated 3 QA pairs")

      except (json.JSONDecodeError, ValueError) as e:
          stats["json_errors"] += 1
          error_entry = {
              "chunk_id": chunk.get('chunk_id', f'unknown_{idx}'),
              "error_type": "json_parse_error",
              "error_message": str(e),
              "raw_content": content[:1000],  # First 1000 chars for analysis
              "timestamp": datetime.now().isoformat()
          }
          with open(ERROR_FILE, 'a', encoding='utf-8') as f:
              f.write(json.dumps(error_entry) + '\n')
          print(f"✗ Chunk {idx+1}/{len(chunks)}: JSON parsing failed - {str(e)[:50]}...")
        

    except Exception as e:
        stats["api_errors"] += 1
        print(f"✗ Chunk {idx+1}/{len(chunks)}: Unexpected error - {str(e)}")  

end_time = time.time()

print(f"Laufzeit: {end_time - start_time:.4f} Sekunden")    

# Print summary
print("\n" + "="*50)
print("GENERATION SUMMARY")
print("="*50)
print(f"Total chunks:    {stats['total_chunks']}")
print(f"Successful:      {stats['successful']} ({stats['successful']/stats['total_chunks']*100:.1f}%)")
print(f"JSON errors:     {stats['json_errors']}")
print(f"API errors:      {stats['api_errors']}")
print(f"\nOutput file:     {OUTPUT_FILE}")
print(f"Error log:       {ERROR_FILE}")