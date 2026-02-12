#!/usr/bin/env python3
"""
Analyse Generation Errors Log

Analysiert generation_errors.jsonl und zeigt:
- Error-Pattern (welche Art von Errors)
- Betroffene Chunks
- Error-Häufigkeiten
"""

import json
from collections import Counter
from pathlib import Path


def analyze_errors(error_file: str):
    """Analysiere Error-Log"""
    
    if not Path(error_file).exists():
        print(f"❌ Error file nicht gefunden: {error_file}")
        return
    
    errors = []
    with open(error_file, 'r') as f:
        for line in f:
            if line.strip():
                errors.append(json.loads(line))
    
    if not errors:
        print("✅ Keine Errors gefunden!")
        return
    
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {len(errors)} Errors")
    print(f"{'='*60}\n")
    
    # 1. Error Types
    error_types = Counter([e.get('error_type', 'unknown') for e in errors])
    print("ERROR TYPES:")
    for error_type, count in error_types.most_common():
        percentage = (count / len(errors)) * 100
        print(f"  {error_type:30s}: {count:4d} ({percentage:5.1f}%)")
    
    # 2. Error Messages (first 50 chars)
    print(f"\n{'='*60}")
    print("ERROR MESSAGES (Top 5):")
    error_msgs = Counter([e.get('error_message', 'unknown')[:50] for e in errors])
    for msg, count in error_msgs.most_common(5):
        print(f"  [{count:3d}×] {msg}...")
    
    # 3. Affected Chunks
    print(f"\n{'='*60}")
    print(f"AFFECTED CHUNKS: {len(errors)} total")
    
    # Group by service if available
    services = Counter([e.get('chunk_id', 'unknown').split('-')[0] 
                       for e in errors if 'chunk_id' in e])
    if services:
        print("\nBy Service:")
        for service, count in services.most_common(10):
            print(f"  {service:20s}: {count:3d}")
    
    # 4. Detailed Examples
    print(f"\n{'='*60}")
    print("DETAILED EXAMPLES (First 3):")
    for i, error in enumerate(errors[:3], 1):
        print(f"\n--- Error #{i} ---")
        print(f"Chunk ID: {error.get('chunk_id', 'N/A')}")
        print(f"Error Type: {error.get('error_type', 'N/A')}")
        print(f"Error Message: {error.get('error_message', 'N/A')[:100]}...")
        if 'response' in error:
            print(f"Response (first 100 chars): {str(error['response'])[:100]}...")
    
    # 5. Summary Statistics
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Total Errors:     {len(errors)}")
    print(f"  Unique Messages:  {len(error_msgs)}")
    print(f"  Most Common:      {error_types.most_common(1)[0][0]}")
    print(f"  Loss Rate:        {(len(errors) / 1932) * 100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    error_file = sys.argv[1] if len(sys.argv) > 1 else "generation_errors.jsonl"
    analyze_errors(error_file)
