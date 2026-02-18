#!/usr/bin/env python3
"""
Merge old comparison (Claude + Llama-8B) with new Llama-70B results
Matches samples by content (chunk + question + answer)
"""

import json
import argparse
from typing import Dict, List, Any
from collections import defaultdict


def normalize_text(text: str) -> str:
    """Normalize text for matching (remove extra whitespace, unicode chars)."""
    # Replace non-breaking space with regular space
    text = text.replace('\u00a0', ' ')
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def create_key(chunk: str, question: str, answer: str) -> str:
    """Create unique key from chunk + question + answer."""
    return f"{normalize_text(chunk)}|||{normalize_text(question)}|||{normalize_text(answer)}"


def load_old_results(path: str) -> Dict[str, Dict[str, Any]]:
    """Load old results with Claude + Llama-8B ratings."""
    results = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                key = create_key(
                    data.get('chunk', ''),
                    data.get('question', ''),
                    data.get('answer', '')
                )
                results[key] = {
                    'chunk_id': data.get('chunk_id'),
                    'model': data.get('model'),
                    'chunk': data.get('chunk'),
                    'question': data.get('question'),
                    'answer': data.get('answer'),
                    'claude_rating': data.get('claude_rating'),
                    'claude_hallucination': data.get('claude_hallucination'),
                    'claude_reasoning': data.get('claude_reasoning'),
                    'llama8b_rating': data.get('llama_rating'),
                    'llama8b_hallucination': data.get('llama_hallucination'),
                    'llama8b_reasoning': data.get('llama_reasoning')
                }
    return results


def load_new_results(path: str) -> Dict[str, Dict[str, Any]]:
    """Load new results with Llama-70B ratings."""
    results = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                key = create_key(
                    data.get('chunk', ''),
                    data.get('question', ''),
                    data.get('answer', '')
                )
                results[key] = {
                    'sample_id': data.get('sample_id'),
                    'llama70b_rating': data.get('rating'),
                    'llama70b_hallucination': data.get('hallucination'),
                    'llama70b_reasoning': data.get('reasoning')
                }
    return results


def merge_results(old_results: Dict, new_results: Dict) -> List[Dict[str, Any]]:
    """Merge old and new results."""
    merged = []
    matched = 0
    unmatched_old = 0
    unmatched_new = 0
    
    # Merge where both exist
    for key, old_data in old_results.items():
        if key in new_results:
            new_data = new_results[key]
            merged_sample = {
                **old_data,
                **new_data
            }
            merged.append(merged_sample)
            matched += 1
        else:
            unmatched_old += 1
            print(f"Warning: Old sample not found in new results: {old_data.get('chunk_id')}")
    
    # Check for new samples not in old
    for key in new_results:
        if key not in old_results:
            unmatched_new += 1
            print(f"Warning: New sample not found in old results: sample_id={new_results[key].get('sample_id')}")
    
    print(f"\nMatching Summary:")
    print(f"  Matched: {matched}")
    print(f"  Unmatched in old: {unmatched_old}")
    print(f"  Unmatched in new: {unmatched_new}")
    
    return merged


def calculate_agreement(merged: List[Dict]) -> Dict[str, Any]:
    """Calculate agreement metrics between judges."""
    
    # Claude vs Llama-8B
    claude_8b_agree = sum(1 for s in merged if s['claude_rating'] == s['llama8b_rating'])
    claude_8b_pct = claude_8b_agree / len(merged) * 100 if merged else 0
    
    # Claude vs Llama-70B
    claude_70b_agree = sum(1 for s in merged if s['claude_rating'] == s['llama70b_rating'])
    claude_70b_pct = claude_70b_agree / len(merged) * 100 if merged else 0
    
    # Llama-8B vs Llama-70B
    llama_8b_70b_agree = sum(1 for s in merged if s['llama8b_rating'] == s['llama70b_rating'])
    llama_8b_70b_pct = llama_8b_70b_agree / len(merged) * 100 if merged else 0
    
    # Rating distributions
    claude_dist = defaultdict(int)
    llama8b_dist = defaultdict(int)
    llama70b_dist = defaultdict(int)
    
    for s in merged:
        claude_dist[s['claude_rating']] += 1
        llama8b_dist[s['llama8b_rating']] += 1
        llama70b_dist[s['llama70b_rating']] += 1
    
    return {
        'total_samples': len(merged),
        'claude_vs_llama8b': {
            'agreement': claude_8b_agree,
            'agreement_pct': claude_8b_pct
        },
        'claude_vs_llama70b': {
            'agreement': claude_70b_agree,
            'agreement_pct': claude_70b_pct
        },
        'llama8b_vs_llama70b': {
            'agreement': llama_8b_70b_agree,
            'agreement_pct': llama_8b_70b_pct
        },
        'distributions': {
            'claude': dict(claude_dist),
            'llama8b': dict(llama8b_dist),
            'llama70b': dict(llama70b_dist)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Merge old and new judge results")
    parser.add_argument('--old', required=True, help='Old results (Claude + Llama-8B)')
    parser.add_argument('--new', required=True, help='New results (Llama-70B)')
    parser.add_argument('--output', required=True, help='Output merged JSONL')
    
    args = parser.parse_args()
    
    print(f"Loading old results from {args.old}...")
    old_results = load_old_results(args.old)
    print(f"  Loaded {len(old_results)} samples")
    
    print(f"\nLoading new results from {args.new}...")
    new_results = load_new_results(args.new)
    print(f"  Loaded {len(new_results)} samples")
    
    print(f"\nMerging results...")
    merged = merge_results(old_results, new_results)
    
    # Save merged
    print(f"\nSaving merged results to {args.output}...")
    with open(args.output, 'w') as f:
        for sample in merged:
            f.write(json.dumps(sample) + '\n')
    
    # Calculate agreement
    print(f"\n{'='*60}")
    print("AGREEMENT ANALYSIS")
    print(f"{'='*60}")
    
    stats = calculate_agreement(merged)
    
    print(f"\nTotal Samples: {stats['total_samples']}")
    
    print(f"\nClaude vs Llama-8B:")
    print(f"  Agreement: {stats['claude_vs_llama8b']['agreement']}/{stats['total_samples']} ({stats['claude_vs_llama8b']['agreement_pct']:.1f}%)")
    
    print(f"\nClaude vs Llama-70B:")
    print(f"  Agreement: {stats['claude_vs_llama70b']['agreement']}/{stats['total_samples']} ({stats['claude_vs_llama70b']['agreement_pct']:.1f}%)")
    
    print(f"\nLlama-8B vs Llama-70B:")
    print(f"  Agreement: {stats['llama8b_vs_llama70b']['agreement']}/{stats['total_samples']} ({stats['llama8b_vs_llama70b']['agreement_pct']:.1f}%)")
    
    print(f"\nRating Distributions:")
    print(f"\n  Claude:")
    for rating, count in sorted(stats['distributions']['claude'].items()):
        pct = count / stats['total_samples'] * 100
        print(f"    {rating}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n  Llama-8B:")
    for rating, count in sorted(stats['distributions']['llama8b'].items()):
        pct = count / stats['total_samples'] * 100
        print(f"    {rating}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n  Llama-70B:")
    for rating, count in sorted(stats['distributions']['llama70b'].items()):
        pct = count / stats['total_samples'] * 100
        print(f"    {rating}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nMerged results saved to: {args.output}")


if __name__ == "__main__":
    main()