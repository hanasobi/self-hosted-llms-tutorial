#!/usr/bin/env python3
"""
Extract cases where Claude found C (Hallucination) but Llama-70B didn't
These are critical False Negatives
"""

import json
import argparse
from typing import List, Dict, Any


def extract_missed_hallucinations(merged_path: str) -> List[Dict[str, Any]]:
    """Extract cases where Claude=C but Llama-70B≠C."""
    missed = []
    
    with open(merged_path, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip():
                sample = json.loads(line)
                
                if sample.get('claude_rating') == 'C' and sample.get('llama70b_rating') != 'C':
                    missed.append({
                        'index': idx,
                        'chunk_id': sample.get('chunk_id'),
                        'model': sample.get('model'),
                        'chunk': sample.get('chunk'),
                        'question': sample.get('question'),
                        'answer': sample.get('answer'),
                        'claude_rating': sample.get('claude_rating'),
                        'claude_reasoning': sample.get('claude_reasoning'),
                        'llama70b_rating': sample.get('llama70b_rating'),
                        'llama70b_reasoning': sample.get('llama70b_reasoning')
                    })
    
    return missed


def write_markdown_review(missed: List[Dict], output_path: str):
    """Write missed hallucinations to markdown for manual review."""
    
    with open(output_path, 'w') as f:
        f.write("# Llama-70B False Negatives: Missed Hallucinations\n\n")
        f.write(f"**Total Cases:** {len(missed)}\n\n")
        f.write("These are cases where Claude detected a hallucination (Rating C) but Llama-70B did not.\n\n")
        f.write("---\n\n")
        
        for idx, case in enumerate(missed, 1):
            f.write(f"## Case {idx}/{len(missed)}\n\n")
            
            # Metadata
            f.write(f"**Chunk ID:** `{case['chunk_id']}`  \n")
            f.write(f"**Model:** `{case['model']}`  \n")
            f.write(f"**Index:** {case['index']}\n\n")
            
            # Full Chunk (nicht kürzen!)
            f.write("### Chunk\n\n")
            f.write("```\n")
            f.write(case['chunk'])
            f.write("\n```\n\n")
            
            # Question
            f.write("### Question\n\n")
            f.write(f"> {case['question']}\n\n")
            
            # Answer
            f.write("### Answer\n\n")
            f.write(f"> {case['answer']}\n\n")
            
            # Claude's Judgment
            f.write("### Claude's Judgment (C - Hallucination)\n\n")
            f.write(f"**Rating:** {case['claude_rating']}  \n")
            f.write(f"**Reasoning:**\n\n")
            f.write(f"{case['claude_reasoning']}\n\n")
            
            # Llama-70B's Judgment (MISSED!)
            f.write(f"### Llama-70B's Judgment ({case['llama70b_rating']} - MISSED!)\n\n")
            f.write(f"**Rating:** {case['llama70b_rating']}  \n")
            f.write(f"**Reasoning:**\n\n")
            f.write(f"{case['llama70b_reasoning']}\n\n")
            
            # Manual Review Section
            f.write("### Manual Review\n\n")
            f.write("**Is Claude correct (true hallucination)?**  \n")
            f.write("- [ ] Yes - Claude is right, this is a hallucination  \n")
            f.write("- [ ] No - Claude is wrong, answer is actually fine  \n")
            f.write("- [ ] Unclear - needs deeper investigation  \n\n")
            
            f.write("**Notes:**  \n")
            f.write("_[Add your manual review notes here]_\n\n")
            
            f.write("---\n\n")
        
        # Summary section
        f.write("\n## Summary\n\n")
        f.write(f"**Total False Negatives:** {len(missed)}\n\n")
        f.write("**Claude C-Detection Rate:** 10/170 (5.9%)  \n")
        f.write(f"**Llama-70B Missed:** {len(missed)}/{10 if len(missed) <= 10 else 'X'} ({len(missed)/10*100:.0f}% of Claude's C-ratings)  \n")
        f.write(f"**Llama-70B C-Detection Rate:** 6/170 (3.5%)  \n\n")
        
        f.write("### Review Instructions\n\n")
        f.write("For each case:\n")
        f.write("1. Read the full chunk carefully\n")
        f.write("2. Check if the answer adds information not in the chunk\n")
        f.write("3. Mark your judgment in the checkboxes\n")
        f.write("4. Add notes explaining your reasoning\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Claude C-ratings that Llama-70B missed"
    )
    parser.add_argument(
        '--merged',
        required=True,
        help='Merged JSONL with all three judges'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output Markdown file for manual review'
    )
    
    args = parser.parse_args()
    
    print(f"Loading merged results from {args.merged}...")
    missed = extract_missed_hallucinations(args.merged)
    
    print(f"\nFound {len(missed)} cases where Claude=C but Llama-70B≠C")
    
    if missed:
        print(f"\nBreakdown:")
        llama_ratings = {}
        for case in missed:
            rating = case['llama70b_rating']
            llama_ratings[rating] = llama_ratings.get(rating, 0) + 1
        
        for rating, count in sorted(llama_ratings.items()):
            print(f"  Claude C → Llama-70B {rating}: {count}")
    
    print(f"\nWriting manual review file to {args.output}...")
    write_markdown_review(missed, args.output)
    
    print(f"\n{'='*60}")
    print("FALSE NEGATIVES EXTRACTED")
    print(f"{'='*60}")
    print(f"Total cases: {len(missed)}")
    print(f"Output: {args.output}")
    print(f"\nReview each case manually to verify if Claude or Llama-70B is correct.")


if __name__ == "__main__":
    main()
