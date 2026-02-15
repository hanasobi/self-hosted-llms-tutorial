#!/usr/bin/env python3
"""
Extract the same 60 samples (20 chunks × 3 QA-pairs) from Post 7.2
Uses seed=42 for reproducibility (same sampling as Post 7.2)
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_chunks(chunks_path: str) -> Dict[str, str]:
    """Load chunks from chunks_all.json and create chunk_id -> content mapping."""
    chunks = {}
    
    with open(chunks_path, 'r') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunk_id = chunk.get('metadata', {}).get('chunk_id')
                content = chunk.get('content', '')
                
                if chunk_id:
                    chunks[chunk_id] = content
    
    return chunks


def load_qa_pairs(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load QA pairs from JSONL file."""
    pairs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def group_by_chunk(qa_pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group QA pairs by chunk_id. Handles both direct and nested chunk_id."""
    chunks = {}
    for pair in qa_pairs:
        # Try direct chunk_id first
        chunk_id = pair.get('chunk_id')
        
        # If not found, try metadata.chunk_id
        if not chunk_id and 'metadata' in pair:
            chunk_id = pair.get('metadata', {}).get('chunk_id')
        
        # Fallback
        if not chunk_id:
            chunk_id = 'unknown'
        
        if chunk_id not in chunks:
            chunks[chunk_id] = []
        chunks[chunk_id].append(pair)
    return chunks


def sample_chunks(chunk_dict: Dict[str, List[Dict[str, Any]]], n_chunks: int = 20, seed: int = 42) -> List[str]:
    """Sample n_chunks chunk_ids with given seed."""
    random.seed(seed)
    chunk_ids = list(chunk_dict.keys())
    sampled_ids = random.sample(chunk_ids, n_chunks)
    return sorted(sampled_ids)  # Sort for reproducibility


def extract_samples_from_model(
    jsonl_path: str,
    sampled_chunk_ids: List[str],
    model_name: str,
    chunks_dict: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Extract QA pairs for the sampled chunks.
    
    Returns list of QA pairs with added metadata for judge evaluation.
    """
    print(f"\nProcessing {model_name}...")
    
    # Load all pairs
    all_pairs = load_qa_pairs(jsonl_path)
    print(f"  Loaded {len(all_pairs)} total QA pairs")
    
    # Group by chunk
    chunks = group_by_chunk(all_pairs)
    print(f"  Found {len(chunks)} unique chunks")
    
    # Extract samples
    samples = []
    missing_chunks_count = 0
    
    for chunk_id in sampled_chunk_ids:
        if chunk_id not in chunks:
            print(f"  ⚠️  WARNING: Chunk {chunk_id} not found in {model_name}!")
            missing_chunks_count += 1
            continue
        
        chunk_pairs = chunks[chunk_id]
        
        # Get chunk content
        chunk_content = chunks_dict.get(chunk_id, "")
        if not chunk_content:
            print(f"  ⚠️  WARNING: No content for chunk {chunk_id} in chunks file!")
        
        # Take first 3 pairs (or all if less than 3)
        for pair in chunk_pairs[:3]:
            # Extract chunk_id (handles both direct and nested)
            extracted_chunk_id = pair.get('chunk_id')
            if not extracted_chunk_id and 'metadata' in pair:
                extracted_chunk_id = pair.get('metadata', {}).get('chunk_id')
            if not extracted_chunk_id:
                extracted_chunk_id = chunk_id
            
            # Add metadata for judge evaluation
            sample = {
                "chunk_id": extracted_chunk_id,
                "model": model_name,
                "chunk": chunk_content,  # NOW FILLED!
                "question": pair.get("question", ""),
                "answer": pair.get("answer", ""),
                # Preserve original metadata
                "question_type": pair.get("question_type", ""),
                "original_pair_id": pair.get("pair_id", "")
            }
            samples.append(sample)
    
    print(f"  Extracted {len(samples)} QA pairs from {len(sampled_chunk_ids) - missing_chunks_count} chunks")
    return samples


def save_samples(samples: List[Dict[str, Any]], output_path: str):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\n✅ Saved {len(samples)} samples to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract 60 samples for Post 8 evaluation")
    
    # Input files
    parser.add_argument("--mistral", required=True, help="Path to Mistral JSONL file")
    parser.add_argument("--llama", required=True, help="Path to Llama JSONL file")
    parser.add_argument("--gpt4o", required=True, help="Path to GPT-4o-mini JSONL file")
    parser.add_argument("--chunks", required=True, help="Path to chunks_all.json (JSONL format)")
    
    # Output
    parser.add_argument("--output-dir", default="./post8_samples", help="Output directory")
    
    # Sampling
    parser.add_argument("--chunk-ids", help="Path to file with chunk IDs (one per line). If provided, uses these instead of random sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42, ignored if --chunk-ids provided)")
    parser.add_argument("--n-chunks", type=int, default=20, help="Number of chunks to sample (default: 20, ignored if --chunk-ids provided)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("EXTRACTING POST 8 SAMPLES")
    print("="*60)
    
    print("\n" + "="*60)
    print("STEP 1: Loading chunks")
    print("="*60)
    
    print(f"Loading chunks from: {args.chunks}")
    chunks_dict = load_chunks(args.chunks)
    print(f"✅ Loaded {len(chunks_dict)} chunks")
    
    # Step 2: Determine sampled chunks
    print("\n" + "="*60)
    print("STEP 2: Determining sampled chunks")
    print("="*60)
    
    if args.chunk_ids:
        # Load chunk IDs from file
        print(f"Loading chunk IDs from: {args.chunk_ids}")
        with open(args.chunk_ids, 'r') as f:
            sampled_chunk_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(sampled_chunk_ids)} chunk IDs from file")
    else:
        # Random sampling
        print(f"Random sampling with seed={args.seed}, n={args.n_chunks}")
        # Use Mistral as reference for chunk selection
        mistral_pairs = load_qa_pairs(args.mistral)
        mistral_chunks = group_by_chunk(mistral_pairs)
        sampled_chunk_ids = sample_chunks(mistral_chunks, n_chunks=args.n_chunks, seed=args.seed)
        print(f"Sampled {len(sampled_chunk_ids)} chunk IDs")
    
    print(f"\nChunk IDs to extract:")
    for i, chunk_id in enumerate(sampled_chunk_ids[:5], 1):
        print(f"  {i}. {chunk_id}")
    if len(sampled_chunk_ids) > 5:
        print(f"  ... ({len(sampled_chunk_ids) - 5} more)")
    
    # Step 3: Extract samples from each model
    print("\n" + "="*60)
    print("STEP 3: Extracting samples from each model")
    print("="*60)
    
    mistral_samples = extract_samples_from_model(args.mistral, sampled_chunk_ids, "mistral-7b", chunks_dict)
    llama_samples = extract_samples_from_model(args.llama, sampled_chunk_ids, "llama-3.1-8b", chunks_dict)
    gpt4o_samples = extract_samples_from_model(args.gpt4o, sampled_chunk_ids, "gpt-4o-mini", chunks_dict)
    
    # Step 4: Combine all samples
    print("\n" + "="*60)
    print("STEP 3: Combining samples")
    print("="*60)
    
    all_samples = mistral_samples + llama_samples + gpt4o_samples
    print(f"Total samples: {len(all_samples)}")
    print(f"  Mistral: {len(mistral_samples)}")
    print(f"  Llama: {len(llama_samples)}")
    print(f"  GPT-4o-mini: {len(gpt4o_samples)}")
    
    # Expected: 20 chunks × 3 pairs × 3 models = 180 pairs
    # But we want 60 pairs total (20 chunks × 3 pairs, one model at a time)
    
    # Step 5: Save outputs
    print("\n" + "="*60)
    print("STEP 5: Saving outputs")
    print("="*60)
    
    # Save combined file (all 180 pairs)
    combined_path = output_dir / "all_samples_60x3.jsonl"
    save_samples(all_samples, str(combined_path))
    
    # Save individual model files
    save_samples(mistral_samples, str(output_dir / "mistral_samples_60.jsonl"))
    save_samples(llama_samples, str(output_dir / "llama_samples_60.jsonl"))
    save_samples(gpt4o_samples, str(output_dir / "gpt4o_samples_60.jsonl"))
    
    # Save chunk IDs for reference
    chunk_ids_path = output_dir / "sampled_chunk_ids.txt"
    with open(chunk_ids_path, 'w') as f:
        if args.chunk_ids:
            f.write(f"# Chunk IDs loaded from: {args.chunk_ids}\n")
        else:
            f.write(f"# Sampled chunk IDs (seed={args.seed}, n={args.n_chunks})\n")
        for chunk_id in sampled_chunk_ids:
            f.write(f"{chunk_id}\n")
    print(f"✅ Saved chunk IDs to: {chunk_ids_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files created in {output_dir}:")
    print(f"  1. all_samples_60x3.jsonl    - All 180 samples (60 per model)")
    print(f"  2. mistral_samples_60.jsonl  - 60 Mistral samples")
    print(f"  3. llama_samples_60.jsonl    - 60 Llama samples")
    print(f"  4. gpt4o_samples_60.jsonl    - 60 GPT-4o-mini samples")
    print(f"  5. sampled_chunk_ids.txt     - List of sampled chunk IDs")
    print("\nNext step:")
    print(f"  python llm_as_judge.py --samples {output_dir}/all_samples_60x3.jsonl --output results.jsonl")


if __name__ == "__main__":
    main()