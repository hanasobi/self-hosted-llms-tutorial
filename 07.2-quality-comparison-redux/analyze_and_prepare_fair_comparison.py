#!/usr/bin/env python3
"""
Fair Comparison Setup for Mistral-7B vs Llama-3.1-8B

This script:
1. Analyzes chunk coverage across all three models
2. Finds common chunks present in ALL THREE models
3. Generates fair comparison review files with IDENTICAL samples

Usage:
    python analyze_and_prepare_fair_comparison.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION - Adjust these paths to your local setup
# =============================================================================

CHUNKS_FILE = "data/chunks_all.jsonl"
GPT4_FILE = "data/qa_pairs_gpt4o_mini.jsonl"
MISTRAL_FILE = "data/qa_pairs_mistral.jsonl"
LLAMA_FILE = "data/qa_pairs_llama.jsonl"

OUTPUT_DIR = "output"
N_SAMPLES = 20  # Number of samples for review
RANDOM_SEED = 42  # For reproducibility

# =============================================================================
# STEP 1: ANALYZE CHUNK COVERAGE
# =============================================================================

def load_chunks(filepath):
    """Load chunk IDs from chunks_all.jsonl"""
    chunks = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunk_id = chunk.get('metadata', {}).get('chunk_id')
                if chunk_id:
                    chunks[chunk_id] = chunk
    return chunks

def load_qa_pairs(filepath):
    """Load QA pairs and group by chunk_id"""
    pairs_by_chunk = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                pair = json.loads(line)
                chunk_id = pair.get('chunk_id')
                if not chunk_id:
                    # Fallback for old format
                    chunk_id = pair.get('metadata', {}).get('chunk_id')
                if chunk_id:
                    pairs_by_chunk[chunk_id].append(pair)
    return pairs_by_chunk

print("="*80)
print("STEP 1: ANALYZING CHUNK COVERAGE")
print("="*80)

# Load all data
print("\nLoading data files...")
all_chunks = load_chunks(CHUNKS_FILE)
gpt4_pairs = load_qa_pairs(GPT4_FILE)
mistral_pairs = load_qa_pairs(MISTRAL_FILE)
llama_pairs = load_qa_pairs(LLAMA_FILE)

print(f"\nChunks available: {len(all_chunks)}")
print(f"GPT-4o-mini chunks: {len(gpt4_pairs)}")
print(f"Mistral-7B chunks: {len(mistral_pairs)}")
print(f"Llama-3.1-8B chunks: {len(llama_pairs)}")

# Analyze pair counts
print("\nQA Pair Statistics:")
for name, pairs_dict in [("GPT-4o-mini", gpt4_pairs), ("Mistral-7B", mistral_pairs), ("Llama-3.1-8B", llama_pairs)]:
    total_pairs = sum(len(pairs) for pairs in pairs_dict.values())
    avg_pairs = total_pairs / len(pairs_dict) if pairs_dict else 0
    print(f"  {name}: {total_pairs} pairs, avg {avg_pairs:.2f} per chunk")
    
    # Show anomalies (chunks with !=3 pairs)
    anomalies = {cid: len(pairs) for cid, pairs in pairs_dict.items() if len(pairs) != 3}
    if anomalies:
        print(f"    Chunks with !=3 pairs: {len(anomalies)}")
        for cid, count in sorted(anomalies.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {cid}: {count} pairs")

# Find intersections
gpt4_set = set(gpt4_pairs.keys())
mistral_set = set(mistral_pairs.keys())
llama_set = set(llama_pairs.keys())

common_all_three = gpt4_set & mistral_set & llama_set

print("\n" + "="*80)
print("INTERSECTION ANALYSIS")
print("="*80)
print(f"\nCommon to ALL THREE models: {len(common_all_three)} chunks")
print(f"GPT + Mistral only: {len(gpt4_set & mistral_set - llama_set)}")
print(f"GPT + Llama only: {len(gpt4_set & llama_set - mistral_set)}")
print(f"Mistral + Llama only: {len(mistral_set & llama_set - gpt4_set)}")

print(f"\nUnique to GPT-4o-mini: {len(gpt4_set - mistral_set - llama_set)}")
print(f"Unique to Mistral-7B: {len(mistral_set - gpt4_set - llama_set)}")
print(f"Unique to Llama-3.1-8B: {len(llama_set - gpt4_set - mistral_set)}")

if len(common_all_three) < N_SAMPLES:
    print(f"\n⚠️  WARNING: Only {len(common_all_three)} common chunks, but {N_SAMPLES} samples requested!")
    print(f"   Reducing to {len(common_all_three)} samples...")
    N_SAMPLES = len(common_all_three)
elif len(common_all_three) >= N_SAMPLES:
    print(f"\n✅ Sufficient common chunks ({len(common_all_three)}) for {N_SAMPLES} samples")

# =============================================================================
# STEP 2: STRATIFIED SAMPLING
# =============================================================================

print("\n" + "="*80)
print("STEP 2: STRATIFIED SAMPLING")
print("="*80)

# Extract service from chunk_id (e.g., "amazon-faq-0" -> "amazon")
def get_service(chunk_id):
    return chunk_id.split('-')[0] if '-' in chunk_id else chunk_id

# Group common chunks by service
chunks_by_service = defaultdict(list)
for chunk_id in common_all_three:
    service = get_service(chunk_id)
    chunks_by_service[service].append(chunk_id)

print(f"\nServices represented: {len(chunks_by_service)}")
print(f"Service distribution:")
for service, chunks in sorted(chunks_by_service.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"  {service}: {len(chunks)} chunks")

# Stratified sampling: try to get one chunk per service
random.seed(RANDOM_SEED)
selected_chunks = []

# First pass: one per service (if possible)
for service, chunks in sorted(chunks_by_service.items()):
    if len(selected_chunks) < N_SAMPLES:
        chunk = random.choice(chunks)
        selected_chunks.append(chunk)

# Second pass: fill remaining slots randomly
remaining = list(common_all_three - set(selected_chunks))
if len(selected_chunks) < N_SAMPLES and remaining:
    additional = random.sample(remaining, min(N_SAMPLES - len(selected_chunks), len(remaining)))
    selected_chunks.extend(additional)

print(f"\nSelected {len(selected_chunks)} chunks for review")
print(f"Sample services: {[get_service(c) for c in selected_chunks[:10]]}...")

# =============================================================================
# STEP 3: GENERATE FAIR COMPARISON FILES
# =============================================================================

print("\n" + "="*80)
print("STEP 3: GENERATING FAIR COMPARISON FILES")
print("="*80)

def generate_review_file(chunks_to_review, model_b_name, model_b_pairs, output_file):
    """Generate review file for one model comparison"""
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Quality Comparison: gpt-4o-mini vs {model_b_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Samples:** {len(chunks_to_review)} chunks\n")
        f.write(f"**QA Pairs per chunk:** 3 (expected)\n")
        f.write(f"**Random Seed:** {RANDOM_SEED}\n")
        f.write(f"**Fair Comparison:** YES - identical samples for both models\n\n")
        f.write("---\n\n")
        
        # Instructions
        f.write("## Instructions for Review\n\n")
        f.write("For each chunk, compare the QA pairs from both datasets:\n\n")
        f.write("**Rating Criteria:**\n")
        f.write("- **A (Perfect):** Faktisch korrekt, natürliche Frage, hilfreiche Antwort, aus Chunk ableitbar\n")
        f.write("- **B (Minor Issues):** Kleine Probleme (z.B. etwas verbose, leicht ungenaue Formulierung)\n")
        f.write("- **C (Problematic):** Faktisch falsch, unverständlich, oder nicht aus Chunk ableitbar\n\n")
        f.write("**Check for:**\n")
        f.write("1. Faktische Korrektheit (aus Chunk ableitbar?)\n")
        f.write("2. Fragenqualität (natürlich, sinnvoll?)\n")
        f.write("3. Antwortqualität (hilfreich, präzise?)\n")
        f.write("4. Type-Labeling (korrekt: factual/conceptual/comparison?)\n")
        f.write("5. Halluzinationen (erfundene Informationen?)\n\n")
        f.write("---\n\n")
        
        # Sample distribution
        services = defaultdict(int)
        for chunk_id in chunks_to_review:
            service = get_service(chunk_id)
            services[service] += 1
        
        f.write("## Sample Distribution\n\n")
        f.write("| Service | Count |\n")
        f.write("|---------|-------|\n")
        for service, count in sorted(services.items()):
            f.write(f"| {service} | {count} |\n")
        f.write("\n---\n\n")
        
        # Generate samples
        for idx, chunk_id in enumerate(chunks_to_review, 1):
            service = get_service(chunk_id)
            
            f.write(f"## Sample {idx}/{len(chunks_to_review)}: {chunk_id}\n\n")
            f.write(f"**Service:** {service}\n")
            f.write(f"**Chunk ID:** {chunk_id}\n\n")
            
            # Original chunk content (FULL, no truncation)
            chunk = all_chunks.get(chunk_id)
            f.write("### Original Chunk Content\n\n")
            if chunk:
                content = chunk.get('content', 'N/A')
                f.write(f"```\n{content}\n```\n\n")
            else:
                f.write("*Chunk not found*\n\n")
            
            # Dataset A: gpt-4o-mini
            f.write("### Dataset A: gpt-4o-mini\n\n")
            gpt4_chunk_pairs = gpt4_pairs.get(chunk_id, [])
            for pair_idx, pair in enumerate(gpt4_chunk_pairs, 1):
                f.write(f"**Pair {pair_idx}:**\n")
                f.write(f"- **Q:** {pair.get('question', 'N/A')}\n")
                f.write(f"- **A:** {pair.get('answer', 'N/A')}\n")
                f.write(f"- **Type:** {pair.get('question_type', 'N/A')}\n")
                f.write(f"- **Rating:** [TODO: A/B/C]\n\n")
            
            # Dataset B: Model being compared
            f.write(f"### Dataset B: {model_b_name}\n\n")
            model_b_chunk_pairs = model_b_pairs.get(chunk_id, [])
            for pair_idx, pair in enumerate(model_b_chunk_pairs, 1):
                f.write(f"**Pair {pair_idx}:**\n")
                f.write(f"- **Q:** {pair.get('question', 'N/A')}\n")
                f.write(f"- **A:** {pair.get('answer', 'N/A')}\n")
                f.write(f"- **Type:** {pair.get('question_type', 'N/A')}\n")
                f.write(f"- **Rating:** [TODO: A/B/C]\n\n")
            
            # Comparison notes
            f.write("### Comparison Notes\n\n")
            f.write("**Observations:**\n")
            f.write("- [TODO: Comparative quality assessment]\n")
            f.write("- [TODO: Which dataset better? Why?]\n")
            f.write("- [TODO: Any hallucinations or errors?]\n\n")
            f.write("---\n\n")
        
        # Summary section
        f.write("## Summary (To be completed after review)\n\n")
        f.write("### Quality Distribution\n\n")
        f.write("**gpt-4o-mini:**\n")
        f.write("- A-Quality: X/Y (Z%)\n")
        f.write("- B-Quality: X/Y (Z%)\n")
        f.write("- C-Quality: X/Y (Z%)\n\n")
        f.write(f"**{model_b_name}:**\n")
        f.write("- A-Quality: X/Y (Z%)\n")
        f.write("- B-Quality: X/Y (Z%)\n")
        f.write("- C-Quality: X/Y (Z%)\n\n")
        f.write("### Key Findings\n\n")
        f.write("1. [TODO: Main quality differences]\n")
        f.write("2. [TODO: Strengths of each dataset]\n")
        f.write("3. [TODO: Common issues]\n")
        f.write("4. [TODO: Hallucinations observed?]\n\n")
        f.write("### Recommendation\n\n")
        f.write("[TODO: Overall assessment and recommendation]\n\n")
    
    return output_path

# Generate review files for both models
mistral_output = generate_review_file(
    selected_chunks,
    "Mistral-7B",
    mistral_pairs,
    "quality_comparison_Mistral_7B_FAIR.md"
)

llama_output = generate_review_file(
    selected_chunks,
    "Llama-3.1-8B",
    llama_pairs,
    "quality_comparison_Llama_3.1_8B_FAIR.md"
)

print(f"\n✅ Generated review files:")
print(f"   Mistral-7B: {mistral_output}")
print(f"   Llama-3.1-8B: {llama_output}")

# =============================================================================
# STEP 4: VERIFICATION
# =============================================================================

print("\n" + "="*80)
print("STEP 4: VERIFICATION")
print("="*80)

print(f"\nVerifying both files have identical samples...")
print(f"Selected chunks ({len(selected_chunks)}):")
for i, chunk_id in enumerate(selected_chunks[:10], 1):
    print(f"  {i}. {chunk_id}")
if len(selected_chunks) > 10:
    print(f"  ... and {len(selected_chunks) - 10} more")

print(f"\n✅ FAIR COMPARISON READY!")
print(f"\nNext steps:")
print(f"1. Upload both review files to Claude")
print(f"2. Claude will review using IDENTICAL samples")
print(f"3. Results will be directly comparable")
print(f"\nFiles location: {OUTPUT_DIR}/")
print("="*80)