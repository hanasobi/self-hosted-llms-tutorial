#!/usr/bin/env python3
"""
Dataset Statistics - Detailed analysis of QA pairs before train/val split.

Shows service distribution, question types, answer lengths, and provides
recommendations for balancing/sampling.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict
import statistics

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
QA_PAIRS_FILE = PROJECT_ROOT / "data" / "processed" / "qa_pairs_generated.jsonl"


def load_qa_pairs(filepath: Path) -> List[Dict]:
    """Load all QA pairs."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def analyze_services(pairs: List[Dict]):
    """Analyze service distribution."""
    print(f"\n{'='*80}")
    print("SERVICE DISTRIBUTION")
    print(f"{'='*80}")
    
    services = Counter(p['metadata']['service'] for p in pairs)
    total = len(pairs)
    
    print(f"\nTotal QA pairs: {total:,}")
    print(f"Unique services: {len(services)}")
    
    # Top services
    print(f"\nTop 20 Services:")
    print(f"{'Service':<20} {'Count':>8} {'Percentage':>12} {'Bar'}")
    print(f"{'-'*80}")
    
    for service, count in services.most_common(20):
        percentage = (count / total) * 100
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        print(f"{service:<20} {count:>8,} {percentage:>11.2f}% {bar}")
    
    if len(services) > 20:
        others = sum(count for _, count in services.most_common()[20:])
        percentage = (others / total) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{'Others':<20} {others:>8,} {percentage:>11.2f}% {bar}")
    
    # Balance metrics
    counts = list(services.values())
    print(f"\nBalance Metrics:")
    print(f"  Max service:     {max(counts):,} pairs ({max(counts)/total*100:.1f}%)")
    print(f"  Min service:     {min(counts):,} pairs ({min(counts)/total*100:.1f}%)")
    print(f"  Average:         {statistics.mean(counts):.1f} pairs")
    print(f"  Median:          {statistics.median(counts):.1f} pairs")
    print(f"  Std Deviation:   {statistics.stdev(counts):.1f}")
    
    # Concentration check
    top5_percentage = sum(count for _, count in services.most_common(5)) / total * 100
    top10_percentage = sum(count for _, count in services.most_common(10)) / total * 100
    
    print(f"\nConcentration:")
    print(f"  Top 5 services:  {top5_percentage:.1f}% of dataset")
    print(f"  Top 10 services: {top10_percentage:.1f}% of dataset")
    
    # Balance assessment
    if max(counts) / total > 0.15:  # Any service >15%
        print(f"\n⚠️  WARNING: Dataset is imbalanced (max service has {max(counts)/total*100:.1f}%)")
        print(f"    Consider stratified sampling for train/val split")
    elif top5_percentage > 60:
        print(f"\n⚠️  WARNING: Top 5 services dominate ({top5_percentage:.1f}%)")
        print(f"    Consider stratified sampling")
    else:
        print(f"\n✓ Service distribution is reasonably balanced")


def analyze_question_types(pairs: List[Dict]):
    """Analyze question type distribution."""
    print(f"\n{'='*80}")
    print("QUESTION TYPE DISTRIBUTION")
    print(f"{'='*80}")
    
    qtypes = Counter(p['question_type'] for p in pairs)
    total = len(pairs)
    
    print(f"\n{'Type':<15} {'Count':>8} {'Percentage':>12} {'Bar'}")
    print(f"{'-'*80}")
    
    for qtype, count in qtypes.most_common():
        percentage = (count / total) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{qtype:<15} {count:>8,} {percentage:>11.2f}% {bar}")
    
    # Balance check
    counts = list(qtypes.values())
    max_count = max(counts)
    min_count = min(counts)
    
    if max_count / min_count > 1.5:  # More than 50% difference
        print(f"\n⚠️  Question types are imbalanced (ratio: {max_count/min_count:.2f}:1)")
    else:
        print(f"\n✓ Question types are well balanced")


def analyze_answer_lengths(pairs: List[Dict]):
    """Analyze answer length distribution by question type."""
    print(f"\n{'='*80}")
    print("ANSWER LENGTH ANALYSIS")
    print(f"{'='*80}")
    
    # Overall statistics
    all_lengths = [len(p['answer']) for p in pairs]
    
    print(f"\nOverall Answer Lengths (characters):")
    print(f"  Min:        {min(all_lengths):,}")
    print(f"  Max:        {max(all_lengths):,}")
    print(f"  Mean:       {statistics.mean(all_lengths):,.0f}")
    print(f"  Median:     {statistics.median(all_lengths):,.0f}")
    print(f"  Std Dev:    {statistics.stdev(all_lengths):,.0f}")
    
    # Distribution
    print(f"\nLength Distribution:")
    ranges = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500), (500, 1000), (1000, 10000)]
    
    print(f"{'Range':<20} {'Count':>8} {'Percentage':>12} {'Bar'}")
    print(f"{'-'*80}")
    
    for start, end in ranges:
        count = sum(1 for l in all_lengths if start <= l < end)
        if count == 0 and start > 500:
            continue  # Skip empty high ranges
        percentage = (count / len(all_lengths)) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        label = f"{start}-{end}" if end < 10000 else f"{start}+"
        print(f"{label:<20} {count:>8,} {percentage:>11.2f}% {bar}")
    
    # By question type
    print(f"\nAnswer Length by Question Type:")
    print(f"{'Type':<15} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*80}")
    
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p['question_type']].append(len(p['answer']))
    
    for qtype in sorted(by_type.keys()):
        lengths = by_type[qtype]
        print(f"{qtype:<15} {statistics.mean(lengths):>8.0f} "
              f"{statistics.median(lengths):>8.0f} "
              f"{min(lengths):>8,} {max(lengths):>8,}")
    
    # Expected pattern check
    factual_mean = statistics.mean(by_type.get('factual', [100]))
    comparison_mean = statistics.mean(by_type.get('comparison', [100]))
    
    print(f"\nPattern Check:")
    if factual_mean < comparison_mean:
        print(f"  ✓ Factual answers shorter than comparisons (expected)")
    else:
        print(f"  ⚠️  Factual answers longer than comparisons (unexpected)")


def analyze_service_x_question_type(pairs: List[Dict]):
    """Analyze Service x Question Type combinations."""
    print(f"\n{'='*80}")
    print("SERVICE x QUESTION TYPE MATRIX")
    print(f"{'='*80}")
    
    # Build matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        service = p['metadata']['service']
        qtype = p['question_type']
        matrix[service][qtype] += 1
    
    # Get top 15 services and all question types
    services = Counter(p['metadata']['service'] for p in pairs)
    top_services = [s for s, _ in services.most_common(15)]
    
    all_qtypes = sorted(set(p['question_type'] for p in pairs))
    
    print(f"\nTop 15 Services x Question Types:")
    
    # Header
    header = f"{'Service':<20}"
    for qtype in all_qtypes:
        header += f" {qtype[:10]:>10}"
    header += f" {'Total':>10}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for service in top_services:
        row = f"{service:<20}"
        total = 0
        for qtype in all_qtypes:
            count = matrix[service][qtype]
            row += f" {count:>10,}"
            total += count
        row += f" {total:>10,}"
        print(row)
    
    # Check for missing combinations
    print(f"\nCombination Coverage:")
    total_combinations = len(top_services) * len(all_qtypes)
    missing = sum(1 for s in top_services for q in all_qtypes if matrix[s][q] == 0)
    
    print(f"  Total combinations: {total_combinations}")
    print(f"  Missing:            {missing}")
    print(f"  Coverage:           {(total_combinations - missing) / total_combinations * 100:.1f}%")
    
    if missing > 0:
        print(f"\n⚠️  {missing} service x type combinations have no samples")
        print(f"    This may affect stratified splitting")


def recommend_strategy(pairs: List[Dict]):
    """Recommend train/val split strategy based on analysis."""
    print(f"\n{'='*80}")
    print("RECOMMENDATION FOR TRAIN/VAL SPLIT")
    print(f"{'='*80}")
    
    services = Counter(p['metadata']['service'] for p in pairs)
    qtypes = Counter(p['question_type'] for p in pairs)
    
    # Check balance
    service_counts = list(services.values())
    max_service_pct = max(service_counts) / len(pairs) * 100
    
    qtype_counts = list(qtypes.values())
    qtype_ratio = max(qtype_counts) / min(qtype_counts) if min(qtype_counts) > 0 else 1
    
    print(f"\nDataset Characteristics:")
    print(f"  Total pairs:          {len(pairs):,}")
    print(f"  Unique services:      {len(services)}")
    print(f"  Max service:          {max_service_pct:.1f}%")
    print(f"  Question type ratio:  {qtype_ratio:.2f}:1")
    
    print(f"\nRecommendation:")
    
    if max_service_pct > 15 or qtype_ratio > 1.5:
        print(f"  Strategy: STRATIFIED SPLIT")
        print(f"  Reason:   Dataset shows imbalance")
        print(f"  Action:")
        print(f"    1. Stratify by service")
        print(f"    2. Ensure all question types in train & val")
        print(f"    3. 90/10 split")
        print(f"    4. Validate val set has all services")
    else:
        print(f"  Strategy: SIMPLE RANDOM SPLIT")
        print(f"  Reason:   Dataset is reasonably balanced")
        print(f"  Action:")
        print(f"    1. Random 90/10 split")
        print(f"    2. Optional: Stratify by service for safety")
    
    # Size check for val set
    val_size = len(pairs) * 0.1
    min_per_service = val_size / len(services)
    
    print(f"\nValidation Set Size Check:")
    print(f"  Expected val size:    {val_size:.0f} pairs")
    print(f"  Services:             {len(services)}")
    print(f"  Pairs per service:    {min_per_service:.1f}")
    
    if min_per_service < 2:
        print(f"  ⚠️  WARNING: Some services may have <2 samples in val set")
        print(f"      Consider 85/15 split or filter out rare services")
    else:
        print(f"  ✓ Sufficient samples per service in val set")


def main():
    print(f"{'='*80}")
    print("QA DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Input: {QA_PAIRS_FILE}")
    
    # Load
    pairs = load_qa_pairs(QA_PAIRS_FILE)
    print(f"\n✓ Loaded {len(pairs):,} QA pairs")
    
    # Run analyses
    analyze_services(pairs)
    analyze_question_types(pairs)
    analyze_answer_lengths(pairs)
    analyze_service_x_question_type(pairs)
    recommend_strategy(pairs)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext step: Create train/val split based on recommendation above")


if __name__ == "__main__":
    main()