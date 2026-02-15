#!/usr/bin/env python3
"""
Compare Llama-Judge ratings with Claude-Judge baseline
Computes Cohen's Kappa, confusion matrix, agreement statistics
"""

import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict
import math


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def compute_cohens_kappa(ratings1: List[str], ratings2: List[str]) -> float:
    """
    Compute Cohen's Kappa for inter-rater agreement.
    
    Kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating lists must have same length")
    
    n = len(ratings1)
    
    # Count agreements
    agreements = sum(1 for r1, r2 in zip(ratings1, ratings2) if r1 == r2)
    observed_agreement = agreements / n
    
    # Count marginals for expected agreement
    rating_labels = ['A', 'B', 'C']
    counts1 = {label: ratings1.count(label) for label in rating_labels}
    counts2 = {label: ratings2.count(label) for label in rating_labels}
    
    # Expected agreement by chance
    expected_agreement = sum(
        (counts1[label] / n) * (counts2[label] / n)
        for label in rating_labels
    )
    
    # Cohen's Kappa
    if expected_agreement == 1.0:
        return 1.0  # Perfect agreement
    
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa


def create_confusion_matrix(ratings1: List[str], ratings2: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Create confusion matrix.
    Returns dict[rater1][rater2] = count
    """
    matrix = defaultdict(lambda: defaultdict(int))
    
    for r1, r2 in zip(ratings1, ratings2):
        matrix[r1][r2] += 1
    
    return matrix


def print_confusion_matrix(matrix: Dict[str, Dict[str, int]], rater1_name: str, rater2_name: str):
    """Print confusion matrix in readable format."""
    labels = ['A', 'B', 'C']
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"Rows: {rater1_name}, Columns: {rater2_name}")
    print()
    
    # Header
    print(f"{'':>12}", end='')
    for label in labels:
        print(f"{label:>8}", end='')
    print(f"{'Total':>10}")
    print("-" * 60)
    
    # Rows
    for r1 in labels:
        print(f"{r1:>12}", end='')
        row_total = 0
        for r2 in labels:
            count = matrix[r1][r2]
            row_total += count
            print(f"{count:>8}", end='')
        print(f"{row_total:>10}")
    
    # Column totals
    print("-" * 60)
    print(f"{'Total':>12}", end='')
    for r2 in labels:
        col_total = sum(matrix[r1][r2] for r1 in labels)
        print(f"{col_total:>8}", end='')
    total = sum(sum(matrix[r1].values()) for r1 in labels)
    print(f"{total:>10}")


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "No agreement (worse than chance)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def analyze_disagreements(data: List[Dict[str, Any]]):
    """Analyze patterns in disagreements."""
    print(f"\n{'='*60}")
    print("DISAGREEMENT ANALYSIS")
    print(f"{'='*60}")
    
    # Group disagreements by type
    disagreements = defaultdict(list)
    
    for sample in data:
        llama = sample.get('llama_rating')
        claude = sample.get('claude_rating')
        
        if llama != claude:
            key = f"{claude}→{llama}"  # Claude says X, Llama says Y
            disagreements[key].append(sample)
    
    # Print summary
    print(f"\nDisagreement patterns:")
    for pattern, samples in sorted(disagreements.items()):
        print(f"  {pattern}: {len(samples)} cases")
    
    # Show examples of most common disagreement
    if disagreements:
        most_common = max(disagreements.items(), key=lambda x: len(x[1]))
        pattern, samples = most_common
        
        print(f"\n{pattern} Examples (showing first 3):")
        for i, sample in enumerate(samples[:3], 1):
            print(f"\n  Example {i}:")
            print(f"    Model: {sample.get('model')}")
            print(f"    Chunk: {sample.get('chunk_id')}")
            print(f"    Q: {sample.get('question')[:60]}...")
            print(f"    Claude: {sample.get('claude_rating')} - {sample.get('claude_reasoning', '')[:60]}...")
            print(f"    Llama:  {sample.get('llama_rating')} - {sample.get('llama_reasoning', '')[:60]}...")


def analyze_by_model(data: List[Dict[str, Any]]):
    """Analyze agreement per generator model."""
    print(f"\n{'='*60}")
    print("PER-MODEL ANALYSIS")
    print(f"{'='*60}")
    
    models = ['mistral-7b', 'llama-3.1-8b', 'gpt-4o-mini']
    
    for model in models:
        model_data = [s for s in data if s.get('model') == model]
        
        if not model_data:
            continue
        
        llama_ratings = [s.get('llama_rating') for s in model_data]
        claude_ratings = [s.get('claude_rating') for s in model_data]
        
        agreements = sum(1 for l, c in zip(llama_ratings, claude_ratings) if l == c)
        total = len(model_data)
        agreement_rate = 100 * agreements / total if total > 0 else 0
        
        kappa = compute_cohens_kappa(llama_ratings, claude_ratings)
        
        print(f"\n{model.upper()}:")
        print(f"  Samples: {total}")
        print(f"  Agreement: {agreements}/{total} ({agreement_rate:.1f}%)")
        print(f"  Cohen's Kappa: {kappa:.3f} ({interpret_kappa(kappa)})")


def main():
    parser = argparse.ArgumentParser(description="Compare Llama-Judge vs Claude-Judge")
    parser.add_argument("--llama", required=True, help="Path to Llama ratings JSONL")
    parser.add_argument("--claude", required=True, help="Path to Claude ratings JSONL")
    
    args = parser.parse_args()
    
    print("="*60)
    print("LLAMA-JUDGE vs CLAUDE-JUDGE COMPARISON")
    print("="*60)
    
    # Load data
    print(f"\nLoading Llama ratings: {args.llama}")
    llama_data = load_jsonl(args.llama)
    print(f"  Loaded {len(llama_data)} samples")
    
    print(f"\nLoading Claude ratings: {args.claude}")
    claude_data = load_jsonl(args.claude)
    print(f"  Loaded {len(claude_data)} samples")
    
    # Merge data by matching chunk_id, model, pair_num
    print("\nMerging datasets...")
    merged = []
    
    # Create lookup for Claude data
    claude_lookup = {}
    for sample in claude_data:
        key = (sample.get('chunk_id'), sample.get('model'), sample.get('pair_num'))
        claude_lookup[key] = sample
    
    # Merge Llama with Claude
    # Group both datasets by (chunk_id, model) and match by position
    
    # Group Claude samples
    claude_grouped = defaultdict(list)
    for sample in claude_data:
        key = (sample.get('chunk_id'), sample.get('model'))
        claude_grouped[key].append(sample)
    
    # Sort each group by pair_num for consistent ordering
    for key in claude_grouped:
        claude_grouped[key].sort(key=lambda x: x.get('pair_num', 0))
    
    # Group Llama samples
    llama_grouped = defaultdict(list)
    for sample in llama_data:
        key = (sample.get('chunk_id'), sample.get('model'))
        llama_grouped[key].append(sample)
    
    # Match by position within each group
    matched = 0
    for key in llama_grouped:
        llama_samples = llama_grouped[key]
        claude_samples = claude_grouped.get(key, [])
        
        # Debug first group
        if matched == 0:
            print(f"\nDebug first group: {key}")
            print(f"  Llama samples: {len(llama_samples)}")
            print(f"  Claude samples: {len(claude_samples)}")
        
        # Match by position
        for i, (llama_sample, claude_sample) in enumerate(zip(llama_samples, claude_samples)):
            merged_sample = {
                **llama_sample,
                'claude_rating': claude_sample.get('claude_rating'),
                'claude_reasoning': claude_sample.get('claude_reasoning'),
                'claude_pair_num': claude_sample.get('pair_num')
            }
            merged.append(merged_sample)
            matched += 1
            
            # Debug first few
            if matched <= 3:
                print(f"\nMatched sample {matched}:")
                print(f"  Llama rating: {llama_sample.get('llama_rating')}")
                print(f"  Claude rating: {claude_sample.get('claude_rating')}")
    
    print(f"\n  Matched {matched} samples by position within (chunk_id, model) groups")
    
    print(f"  Merged {len(merged)} samples")
    
    if len(merged) == 0:
        print("\n❌ ERROR: No samples could be matched!")
        print("Check that chunk_ids and models align between files.")
        return
    
    # Extract ratings
    llama_ratings = [s.get('llama_rating') for s in merged if s.get('llama_rating')]
    claude_ratings = [s.get('claude_rating') for s in merged if s.get('claude_rating')]
    
    if len(llama_ratings) != len(claude_ratings):
        print(f"\n⚠️  WARNING: Rating count mismatch!")
        print(f"  Llama: {len(llama_ratings)}, Claude: {len(claude_ratings)}")
        # Use minimum
        min_len = min(len(llama_ratings), len(claude_ratings))
        llama_ratings = llama_ratings[:min_len]
        claude_ratings = claude_ratings[:min_len]
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    total = len(llama_ratings)
    agreements = sum(1 for l, c in zip(llama_ratings, claude_ratings) if l == c)
    agreement_rate = 100 * agreements / total
    
    print(f"\nTotal samples: {total}")
    print(f"Agreement: {agreements}/{total} ({agreement_rate:.1f}%)")
    
    kappa = compute_cohens_kappa(llama_ratings, claude_ratings)
    print(f"\nCohen's Kappa: {kappa:.3f}")
    print(f"Interpretation: {interpret_kappa(kappa)}")
    
    # Confusion matrix
    matrix = create_confusion_matrix(claude_ratings, llama_ratings)
    print_confusion_matrix(matrix, "Claude", "Llama")
    
    # Rating distribution
    print(f"\n{'='*60}")
    print("RATING DISTRIBUTION")
    print(f"{'='*60}")
    
    for rater_name, ratings in [("Claude", claude_ratings), ("Llama", llama_ratings)]:
        a_count = ratings.count('A')
        b_count = ratings.count('B')
        c_count = ratings.count('C')
        print(f"\n{rater_name}:")
        print(f"  A: {a_count}/{total} ({100*a_count/total:.1f}%)")
        print(f"  B: {b_count}/{total} ({100*b_count/total:.1f}%)")
        print(f"  C: {c_count}/{total} ({100*c_count/total:.1f}%)")
    
    # Disagreement analysis
    analyze_disagreements(merged)
    
    # Per-model analysis
    analyze_by_model(merged)
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()