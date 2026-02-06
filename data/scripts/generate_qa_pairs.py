#!/usr/bin/env python3
"""
Simple QA-pair generation script.

No checkpoints, no resume - just generates QA pairs from chunks.
If interrupted, you can manually skip already processed chunks by editing SKIP_FIRST.

Requirements:
- pip install openai
- export OPENAI_API_KEY="your-key-here"
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI library not installed. Run: pip install openai")
    exit(1)

# Configuration
INPUT_FILE = project_root / "data" / "processed" / "chunks_token_based.jsonl"
OUTPUT_FILE = project_root / "data" / "processed" / "qa_pairs_generated.jsonl"
SKIP_FIRST = 0  # Set to N if you want to skip the first N chunks (e.g., after interruption)

# Model settings
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 1500
REQUEST_DELAY = 0.0  # Seconds between requests

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

Generate the three question-answer pairs in the following JSON format (only the JSON array, no additional explanations):
[
  {
    "question": "...",
    "answer": "...",
    "type": "factual|conceptual|comparison"
  }
]"""


def load_chunks(filepath: str) -> List[Dict]:
    """Load all chunks from JSONL file."""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def generate_qa_pairs(client: OpenAI, chunk_content: str) -> Optional[List[Dict]]:
    """Generate QA pairs for a chunk. Returns None on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{chunk_content}\n\nNow generate the three question-answer pairs in JSON format."}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Extract response
        content = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        qa_pairs = json.loads(content)
        
        # Basic validation
        if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
            return None
        
        for pair in qa_pairs:
            if not all(k in pair for k in ['question', 'answer', 'type']):
                return None
        
        return qa_pairs
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def save_qa_pair(output_file: str, qa_pair: Dict, chunk_metadata: Dict):
    """Append a QA pair to the output file."""
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


def main():
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        exit(1)
    
    client = OpenAI()
    
    # Load chunks
    print(f"Loading chunks from: {INPUT_FILE}")
    chunks = load_chunks(INPUT_FILE)
    print(f"✓ Loaded {len(chunks):,} chunks")
    
    if SKIP_FIRST > 0:
        print(f"⏭️  Skipping first {SKIP_FIRST} chunks")
        chunks = chunks[SKIP_FIRST:]
    
    # Stats
    total_pairs = 0
    failed = 0
    start_time = time.time()
    
    # Process chunks
    for i, chunk in enumerate(chunks, 1):
        actual_index = i + SKIP_FIRST
        service = chunk.get('metadata', {}).get('service', 'unknown')
        source = chunk.get('metadata', {}).get('source_file', 'unknown')
        chunk_id = chunk.get('metadata', {}).get('chunk_id', '?')
        tokens = chunk.get('token_count', 0)
        
        print(f"[{actual_index}/{len(chunks)+SKIP_FIRST}] {service} ({source}:{chunk_id}, {tokens} tokens)...", end=' ')
        
        # Generate
        qa_pairs = generate_qa_pairs(client, chunk['content'])
        
        if qa_pairs is None:
            print("❌ FAILED")
            failed += 1
        else:
            # Save each pair
            for pair in qa_pairs:
                save_qa_pair(OUTPUT_FILE, pair, chunk['metadata'])
            
            total_pairs += len(qa_pairs)
            print(f"✓ {len(qa_pairs)} pairs")
        
        # Progress every 50
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = len(chunks) - i
            eta = remaining / rate if rate > 0 else 0
            print(f"\n  Progress: {i}/{len(chunks)} | Pairs: {total_pairs} | Failed: {failed} | ETA: {eta/60:.1f}min\n")
        
        # Rate limit
        if i < len(chunks):
            time.sleep(REQUEST_DELAY)
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Processed: {len(chunks)} chunks")
    print(f"Generated: {total_pairs} QA pairs")
    print(f"Failed: {failed} chunks")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()