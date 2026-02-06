#!/usr/bin/env python3
"""
Quality Check for generated QA pairs.

Stratified sampling across services and question types for manual review.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict

# Configuration - Auto-detect project root
SCRIPT_DIR = Path(__file__).parent
# Assuming script is in data/scripts/dataset_preparation/
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "qa_pairs_generated.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "qa_pairs_flagged_for_review.txt"
MIN_ANSWER_LENGTH = 20  # Flag answers shorter than this
MAX_ANSWER_LENGTH = 500  # Flag answers longer than this


def load_qa_pairs(filepath: Path) -> List[Dict]:
    """Load all QA pairs from JSONL."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def analyze_dataset(pairs: List[Dict]):
    """Print dataset statistics."""
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total QA pairs: {len(pairs):,}")
    
    # Question Types
    question_types = Counter(p['question_type'] for p in pairs)
    print(f"\nQuestion Type Distribution:")
    for qtype, count in question_types.most_common():
        percentage = (count / len(pairs)) * 100
        print(f"  {qtype:15s}: {count:5,} ({percentage:5.1f}%)")
    
    # Services
    services = Counter(p['metadata']['service'] for p in pairs)
    print(f"\nService Distribution (Top 15):")
    for service, count in services.most_common(15):
        percentage = (count / len(pairs)) * 100
        print(f"  {service:15s}: {count:5,} ({percentage:5.1f}%)")
    
    if len(services) > 15:
        others = sum(count for _, count in services.most_common()[15:])
        percentage = (others / len(pairs)) * 100
        print(f"  {'Others':15s}: {others:5,} ({percentage:5.1f}%)")
    
    # Answer Length Statistics
    answer_lengths = [len(p['answer']) for p in pairs]
    print(f"\nAnswer Length Statistics (characters):")
    print(f"  Min:     {min(answer_lengths):,}")
    print(f"  Max:     {max(answer_lengths):,}")
    print(f"  Average: {sum(answer_lengths) / len(answer_lengths):,.0f}")
    print(f"  Median:  {sorted(answer_lengths)[len(answer_lengths)//2]:,}")
    
    # Length Distribution
    print(f"\nAnswer Length Distribution:")
    ranges = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500), (500, 1000)]
    for start, end in ranges:
        count = sum(1 for l in answer_lengths if start <= l < end)
        percentage = (count / len(answer_lengths)) * 100
        print(f"  {start:4d}-{end:4d} chars: {count:5,} ({percentage:5.1f}%)")
    
    # Question Length Statistics
    question_lengths = [len(p['question']) for p in pairs]
    print(f"\nQuestion Length Statistics (characters):")
    print(f"  Min:     {min(question_lengths):,}")
    print(f"  Max:     {max(question_lengths):,}")
    print(f"  Average: {sum(question_lengths) / len(question_lengths):,.0f}")


def flag_potential_issues(pair: Dict) -> List[str]:
    """Check for potential quality issues. Returns list of flags."""
    flags = []
    
    answer = pair['answer']
    question = pair['question']
    
    # Very short answer
    if len(answer) < MIN_ANSWER_LENGTH:
        flags.append(f"⚠️  SHORT_ANSWER ({len(answer)} chars)")
    
    # Very long answer
    if len(answer) > MAX_ANSWER_LENGTH:
        flags.append(f"⚠️  LONG_ANSWER ({len(answer)} chars)")
    
    # Answer starts with phrases that might indicate hallucination
    hallucination_indicators = [
        "based on my knowledge",
        "as an ai",
        "i don't have access",
        "according to aws documentation",  # Should say "according to the context"
        "generally speaking",
        "in general,",
    ]
    
    answer_lower = answer.lower()
    for indicator in hallucination_indicators:
        if indicator in answer_lower:
            flags.append(f"🚨 HALLUCINATION_INDICATOR: '{indicator}'")
    
    # Very generic questions
    generic_indicators = ["what is", "what are", "how do", "how does"]
    question_lower = question.lower()
    if any(question_lower.startswith(ind) for ind in generic_indicators):
        # This is OK, but flag if it's TOO generic (no specific terms)
        if len(question.split()) < 6:
            flags.append("💭 GENERIC_QUESTION")
    
    # Answer much shorter than question (suspicious)
    if len(answer) < len(question):
        flags.append("⚠️  ANSWER_SHORTER_THAN_QUESTION")
    
    # Missing required fields
    if not question.strip():
        flags.append("🚨 EMPTY_QUESTION")
    if not answer.strip():
        flags.append("🚨 EMPTY_ANSWER")
    
    return flags


def write_flagged_sample(f, idx: int, pair: Dict, flags: List[str]):
    """Write a single flagged QA pair to file."""
    f.write(f"\n{'='*80}\n")
    f.write(f"FLAGGED SAMPLE #{idx}\n")
    f.write(f"{'='*80}\n\n")
    
    # Metadata
    meta = pair['metadata']
    f.write(f"Service:       {meta['service']}\n")
    f.write(f"Doc Type:      {meta['doc_type']}\n")
    f.write(f"Source:        {meta['source_file']} ({meta['chunk_id']})\n")
    f.write(f"Question Type: {pair['question_type']}\n")
    f.write(f"Source Tokens: {meta['source_tokens']}\n")
    
    # Flags
    f.write(f"\nQuality Flags:\n")
    for flag in flags:
        f.write(f"  {flag}\n")
    
    # Question & Answer
    f.write(f"\nQUESTION:\n")
    f.write(f"  {pair['question']}\n")
    
    f.write(f"\nANSWER: ({len(pair['answer'])} chars)\n")
    # Wrap answer nicely
    answer = pair['answer']
    words = answer.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > 76:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(' '.join(current_line))
    
    for line in lines:
        f.write(f"  {line}\n")
    
    f.write("\n")


def check_all_pairs(pairs: List[Dict], output_file: Path) -> int:
    """Check all pairs and write flagged ones to file. Returns count of flagged pairs."""
    print(f"\n{'='*80}")
    print("CHECKING ALL PAIRS FOR QUALITY ISSUES")
    print(f"{'='*80}")
    
    flagged_pairs = []
    
    for pair in pairs:
        flags = flag_potential_issues(pair)
        if flags:
            flagged_pairs.append((pair, flags))
    
    print(f"Total pairs checked: {len(pairs):,}")
    print(f"Flagged pairs:       {len(flagged_pairs)}")
    
    if flagged_pairs:
        print(f"\nWriting flagged pairs to: {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("QA PAIRS FLAGGED FOR MANUAL REVIEW\n")
            f.write("="*80 + "\n")
            f.write(f"\nTotal QA pairs: {len(pairs):,}\n")
            f.write(f"Flagged pairs:  {len(flagged_pairs)}\n")
            f.write(f"\nReview these samples for quality issues:\n")
            f.write("  - Are answers accurate based on context?\n")
            f.write("  - Are questions realistic?\n")
            f.write("  - Any hallucinated information?\n")
            f.write("  - Are question types correctly labeled?\n")
            f.write("\n")
            
            for i, (pair, flags) in enumerate(flagged_pairs, 1):
                write_flagged_sample(f, i, pair, flags)
        
        print(f"✓ Wrote {len(flagged_pairs)} flagged samples to file")
    else:
        print("\n✓ No quality issues found!")
    
    return len(flagged_pairs)


def main():
    print(f"{'='*80}")
    print("QA PAIRS QUALITY CHECK")
    print(f"{'='*80}")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Load
    pairs = load_qa_pairs(INPUT_FILE)
    print(f"✓ Loaded {len(pairs):,} QA pairs")
    
    # Statistics
    analyze_dataset(pairs)
    
    # Check all pairs and write flagged ones to file
    flagged_count = check_all_pairs(pairs, OUTPUT_FILE)
    
    # Summary
    print(f"\n{'='*80}")
    print("QUALITY CHECK SUMMARY")
    print(f"{'='*80}")
    print(f"Total pairs:     {len(pairs):,}")
    print(f"Flagged pairs:   {flagged_count}")
    print(f"Clean pairs:     {len(pairs) - flagged_count:,}")
    
    if flagged_count > 0:
        percentage = (flagged_count / len(pairs)) * 100
        print(f"\nFlag rate:       {percentage:.2f}%")
        print(f"\n⚠️  {flagged_count} pairs flagged for review")
        print(f"Review file:     {OUTPUT_FILE}")
        print(f"\nNext steps:")
        print(f"  1. Open {OUTPUT_FILE.name} and review flagged samples")
        print(f"  2. Decide: Keep, filter, or regenerate these samples")
        print(f"  3. If acceptable, proceed with train/val split")
    else:
        print(f"\n✓ All pairs passed automatic quality checks!")
        print(f"\nNext step: Proceed with train/val split")


if __name__ == "__main__":
    main()