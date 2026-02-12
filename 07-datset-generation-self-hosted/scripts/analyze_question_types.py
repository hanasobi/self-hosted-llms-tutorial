#!/usr/bin/env python3
"""
Analyse Question Type Distribution

Analysiert qa_pairs_positive.jsonl und zeigt:
- Type-Distribution (factual, conceptual, comparison)
- Unexpected types (procedural, etc.)
- Recommendations für Post-Processing
"""

import json
from collections import Counter
from pathlib import Path


def analyze_question_types(qa_file: str):
    """Analysiere Question Types"""
    
    if not Path(qa_file).exists():
        print(f"❌ QA file nicht gefunden: {qa_file}")
        return
    
    pairs = []
    with open(qa_file, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    if not pairs:
        print("❌ Keine QA-Paare gefunden!")
        return
    
    print(f"\n{'='*60}")
    print(f"QUESTION TYPE ANALYSIS: {len(pairs)} QA-Paare")
    print(f"{'='*60}\n")
    
    # 1. Type Distribution
    types = Counter([p.get('question_type', 'unknown') for p in pairs])
    
    # Expected types
    expected_types = {'factual', 'conceptual', 'comparison'}
    unexpected_types = set(types.keys()) - expected_types - {'unknown'}
    
    print("TYPE DISTRIBUTION:")
    print("\n✅ Expected Types:")
    for qtype in ['factual', 'conceptual', 'comparison']:
        count = types.get(qtype, 0)
        percentage = (count / len(pairs)) * 100
        print(f"  {qtype:15s}: {count:5d} ({percentage:5.1f}%)")
    
    if unexpected_types:
        print("\n⚠️  Unexpected Types:")
        for qtype in sorted(unexpected_types):
            count = types[qtype]
            percentage = (count / len(pairs)) * 100
            print(f"  {qtype:15s}: {count:5d} ({percentage:5.1f}%) ← UNEXPECTED!")
    
    if types.get('unknown', 0) > 0:
        print(f"\n❌ Unknown Type: {types['unknown']} ({(types['unknown']/len(pairs))*100:.1f}%)")
    
    # 2. Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    
    unexpected_count = sum(types[t] for t in unexpected_types)
    unexpected_pct = (unexpected_count / len(pairs)) * 100
    
    if unexpected_pct < 5:
        print(f"\n✅ Unexpected types: {unexpected_pct:.1f}% (< 5%)")
        print("   → ACTION: FILTER OUT unexpected types")
        print("   → Expected loss: negligible")
    
    elif 5 <= unexpected_pct < 15:
        print(f"\n⚠️  Unexpected types: {unexpected_pct:.1f}% (5-15%)")
        print("   → ACTION: APPLY HEURISTIC FIX")
        print("   → Example: procedural → conceptual")
    
    else:
        print(f"\n❌ Unexpected types: {unexpected_pct:.1f}% (> 15%)")
        print("   → ACTION: RE-RUN with improved prompt")
        print("   → Too many errors for simple fix")
    
    # 3. Sample unexpected types
    if unexpected_types:
        print(f"\n{'='*60}")
        print("EXAMPLES OF UNEXPECTED TYPES:")
        
        for qtype in sorted(unexpected_types)[:3]:  # Max 3 examples
            samples = [p for p in pairs if p.get('question_type') == qtype][:2]
            print(f"\n--- {qtype.upper()} (showing 2/{types[qtype]}) ---")
            
            for i, sample in enumerate(samples, 1):
                print(f"\n  Example {i}:")
                print(f"  Question: {sample.get('question', 'N/A')[:80]}...")
                print(f"  Answer:   {sample.get('answer', 'N/A')[:80]}...")
    
    # 4. Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Total Pairs:         {len(pairs)}")
    print(f"  Expected Types:      {sum(types[t] for t in expected_types)} ({(sum(types[t] for t in expected_types)/len(pairs))*100:.1f}%)")
    print(f"  Unexpected Types:    {unexpected_count} ({unexpected_pct:.1f}%)")
    print(f"  Type-Labeling Accuracy: {(sum(types[t] for t in expected_types)/len(pairs))*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    qa_file = sys.argv[1] if len(sys.argv) > 1 else "qa_pairs_positive.jsonl"
    analyze_question_types(qa_file)
