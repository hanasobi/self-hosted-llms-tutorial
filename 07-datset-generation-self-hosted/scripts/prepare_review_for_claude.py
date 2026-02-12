#!/usr/bin/env python3
"""
Prepare Quality Comparison for Claude Review

Erstellt eine Markdown-Datei mit 30 Chunks und ihren QA-Pairs
von beiden Datasets (gpt-4o-mini und Mistral-7B) für systematisches
Quality-Review durch Claude.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_chunks(chunks_file: str) -> dict:
    """Load original chunks"""
    chunks = {}
    with open(chunks_file, 'r') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunk_id = chunk.get('id')
                if chunk_id:
                    chunks[chunk_id] = chunk
    return chunks


def load_qa_pairs(filepath: str) -> dict:
    """Load QA pairs grouped by chunk_id"""
    chunk_to_pairs = defaultdict(list)
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                pair = json.loads(line)
                chunk_id = pair.get('metadata', {}).get('chunk_id')
                if chunk_id:
                    chunk_to_pairs[chunk_id].append(pair)
    
    return chunk_to_pairs


def get_chunk_service(chunk_id: str) -> str:
    """Extract service from chunk_id"""
    parts = chunk_id.split('-')
    return parts[0] if len(parts) > 0 else 'unknown'


def stratified_sampling(chunk_ids: list, n_samples: int = 30) -> list:
    """Sample chunks stratified by service"""
    
    # Group by service
    service_chunks = defaultdict(list)
    for chunk_id in chunk_ids:
        service = get_chunk_service(chunk_id)
        service_chunks[service].append(chunk_id)
    
    # Calculate samples per service
    services = sorted(service_chunks.keys())
    samples_per_service = max(1, n_samples // len(services))
    
    # Sample from each service
    sampled = []
    for service in services:
        chunks = service_chunks[service]
        n = min(samples_per_service, len(chunks))
        sampled.extend(random.sample(chunks, n))
    
    # Fill up to n_samples if needed
    if len(sampled) < n_samples:
        remaining = set(chunk_ids) - set(sampled)
        additional = min(n_samples - len(sampled), len(remaining))
        sampled.extend(random.sample(list(remaining), additional))
    
    return sampled[:n_samples]


def write_review_file(chunks: dict, gpt4_pairs: dict, mistral_pairs: dict,
                      sampled_chunks: list, output_file: str):
    """Write comparison data to markdown file for Claude review"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Quality Comparison: gpt-4o-mini vs Mistral-7B\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Samples:** {len(sampled_chunks)} chunks\n")
        f.write(f"**QA Pairs per chunk:** 3 (expected)\n\n")
        
        # Instructions for Claude
        f.write("---\n\n")
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
        
        # Service distribution
        service_dist = defaultdict(int)
        for chunk_id in sampled_chunks:
            service_dist[get_chunk_service(chunk_id)] += 1
        
        f.write("## Sample Distribution\n\n")
        f.write("| Service | Count |\n")
        f.write("|---------|-------|\n")
        for service, count in sorted(service_dist.items()):
            f.write(f"| {service} | {count} |\n")
        f.write("\n---\n\n")
        
        # Write each chunk comparison
        for i, chunk_id in enumerate(sampled_chunks, 1):
            chunk = chunks.get(chunk_id)
            gpt4_qa = gpt4_pairs.get(chunk_id, [])
            mistral_qa = mistral_pairs.get(chunk_id, [])
            
            f.write(f"## Sample {i}/{len(sampled_chunks)}: {chunk_id}\n\n")
            
            # Chunk metadata
            if chunk:
                service = get_chunk_service(chunk_id)
                f.write(f"**Service:** {service}\n")
                f.write(f"**Chunk ID:** {chunk_id}\n\n")
            
            # Chunk content
            f.write("### Original Chunk Content\n\n")
            if chunk:
                content = chunk.get('content', 'N/A')
                # Truncate if too long
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"
                f.write(f"```\n{content}\n```\n\n")
            else:
                f.write("*Chunk not found*\n\n")
            
            # gpt-4o-mini QA pairs
            f.write("### Dataset A: gpt-4o-mini\n\n")
            if gpt4_qa:
                for j, pair in enumerate(gpt4_qa, 1):
                    f.write(f"**Pair {j}:**\n")
                    f.write(f"- **Q:** {pair.get('question', 'N/A')}\n")
                    f.write(f"- **A:** {pair.get('answer', 'N/A')}\n")
                    f.write(f"- **Type:** {pair.get('question_type', 'N/A')}\n")
                    f.write(f"- **Rating:** [TODO: A/B/C]\n\n")
            else:
                f.write("*No QA pairs found*\n\n")
            
            # Mistral-7B QA pairs
            f.write("### Dataset B: Mistral-7B\n\n")
            if mistral_qa:
                for j, pair in enumerate(mistral_qa, 1):
                    f.write(f"**Pair {j}:**\n")
                    f.write(f"- **Q:** {pair.get('question', 'N/A')}\n")
                    f.write(f"- **A:** {pair.get('answer', 'N/A')}\n")
                    f.write(f"- **Type:** {pair.get('question_type', 'N/A')}\n")
                    f.write(f"- **Rating:** [TODO: A/B/C]\n\n")
            else:
                f.write("*No QA pairs found*\n\n")
            
            # Comparison notes
            f.write("### Comparison Notes\n\n")
            f.write("**Observations:**\n")
            f.write("- [TODO: Comparative quality assessment]\n")
            f.write("- [TODO: Which dataset better? Why?]\n")
            f.write("- [TODO: Any hallucinations or errors?]\n\n")
            
            f.write("---\n\n")
        
        # Summary section (to be filled by Claude)
        f.write("## Summary (To be completed after review)\n\n")
        f.write("### Quality Distribution\n\n")
        f.write("**gpt-4o-mini:**\n")
        f.write("- A-Quality: X/Y (Z%)\n")
        f.write("- B-Quality: X/Y (Z%)\n")
        f.write("- C-Quality: X/Y (Z%)\n\n")
        
        f.write("**Mistral-7B:**\n")
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


def main(chunks_file: str, gpt4_file: str, mistral_file: str, 
         output_file: str = "../output/quality_comparison_for_review.md",
         n_samples: int = 30):
    """Main function"""
    
    print(f"\n{'='*60}")
    print("PREPARING QUALITY COMPARISON FOR CLAUDE REVIEW")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    chunks = load_chunks(chunks_file)
    gpt4_pairs = load_qa_pairs(gpt4_file)
    mistral_pairs = load_qa_pairs(mistral_file)
    
    print(f"  Chunks:        {len(chunks)}")
    print(f"  gpt-4o-mini:   {len(gpt4_pairs)} chunks")
    print(f"  Mistral-7B:    {len(mistral_pairs)} chunks")
    
    # Find common chunks
    common_chunks = set(gpt4_pairs.keys()) & set(mistral_pairs.keys())
    print(f"\n  Common chunks: {len(common_chunks)}")
    
    if len(common_chunks) < n_samples:
        print(f"\n⚠️  Warning: Only {len(common_chunks)} common chunks available")
        n_samples = len(common_chunks)
    
    # Sample chunks
    print(f"\nSampling {n_samples} chunks (stratified by service)...")
    sampled_chunks = stratified_sampling(list(common_chunks), n_samples)
    
    # Write review file
    print(f"\nWriting review file: {output_file}")
    write_review_file(chunks, gpt4_pairs, mistral_pairs, sampled_chunks, output_file)
    
    print(f"\n✅ Review file created: {output_file}")
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("\n1. Upload file to Claude Project")
    print("2. Ask Claude to review and rate all samples")
    print("3. Claude will fill in ratings and summary")
    print("4. Download reviewed file for analysis")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python prepare_review_for_claude.py <chunks_file> <gpt4_file> <mistral_file> [n_samples]")
        print("\nExample:")
        print("  python prepare_review_for_claude.py \\")
        print("    all_chunks.jsonl \\")
        print("    qa_pairs_gpt4o_mini.jsonl \\")
        print("    qa_pairs_mistral.jsonl \\")
        print("    30")
        sys.exit(1)
    
    chunks_file = sys.argv[1]
    gpt4_file = sys.argv[2]
    mistral_file = sys.argv[3]
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    
    main(chunks_file, gpt4_file, mistral_file, n_samples=n_samples)