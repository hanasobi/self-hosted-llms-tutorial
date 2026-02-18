#!/usr/bin/env python3
"""
Multi-LoRA A/B Testing Script

Systematischer Vergleich von zwei LoRA-Adaptern:
- v1: Ohne negative samples trainiert
- v2: Mit negative samples trainiert

Testet auf:
1. Positive Samples (1158): Sollte beide Adapter gleich gut performen
2. Negative Samples (42): v2 sollte deutlich besser "cannot answer" erkennen

Output:
- Detaillierte Metriken pro Adapter
- Side-by-side Vergleich
- Saved results für weitere Analyse
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import time

# Configuration
VLLM_API_URL = "http://localhost:8000/v1/completions"
EVAL_DATASET_PATH = "data/processed/eval.jsonl"
OUTPUT_DIR = "evaluation/multi_lora_ab_test"

# Adapter Names (from vLLM logs)
ADAPTER_V1 = "aws-rag-qa-v1"  # Ohne negative samples
ADAPTER_V2 = "aws-rag-qa-v2"  # Mit negative samples (beachte den Punkt!)

# System Prompt (wie im Training)
SYSTEM_PROMPT = """You are an expert assistant for AWS (Amazon Web Services) certification preparation.

CRITICAL: You must answer ONLY based on the provided context below. Follow these rules strictly:

Rules for answers:
- Extract and provide ALL relevant information from the context
- NEVER add information not explicitly stated in the context
- NEVER use external knowledge or your training data - only use what's in the given context
- Be as detailed as the context allows - short context = short answer, detailed context = detailed answer
- Write in complete, helpful sentences as if answering a colleague
- If comparing items, ONLY compare aspects explicitly mentioned in the context
- If the context doesn't provide enough information to answer the question, respond with: "I cannot answer this question with the given context."
- Answers should be in English
- Do not reference or mention "the context" in your answer - answer naturally as if you had this knowledge
"""


def load_eval_dataset(path: str) -> List[Dict]:
    """Load eval dataset from JSONL."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def create_prompt(context: str, question: str) -> str:
    """Create properly formatted prompt for inference."""
    return f"[INST] {SYSTEM_PROMPT}\n\n{context}\n\nQuestion: {question} [/INST]"


def call_vllm(adapter_name: str, prompt: str, max_retries: int = 3) -> str:
    """Call vLLM API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                VLLM_API_URL,
                json={
                    "model": adapter_name,
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.0,  # Deterministic for evaluation
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return f"ERROR: {str(e)}"


def is_refusal(answer: str) -> bool:
    """
    Check if answer is a refusal (cannot answer).
    
    Looks for various refusal patterns:
    - "I cannot answer"
    - "cannot be answered"
    - "does not contain sufficient information"
    - etc.
    """
    answer_lower = answer.lower()
    
    refusal_patterns = [
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not able to answer",
        "does not contain",
        "doesn't contain",
        "no information",
        "insufficient information",
        "not enough information",
        "not provided",
        "not mentioned",
    ]
    
    return any(pattern in answer_lower for pattern in refusal_patterns)


def evaluate_adapter(
    adapter_name: str,
    samples: List[Dict],
    sample_type: str = "all"
) -> Dict:
    """
    Evaluate a single adapter on the dataset.
    
    Args:
        adapter_name: vLLM adapter name
        samples: List of eval samples
        sample_type: "all", "positive", or "negative"
    
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {adapter_name}")
    print(f"Sample type: {sample_type}")
    print(f"{'='*80}\n")
    
    # Filter samples if needed
    if sample_type == "positive":
        samples = [s for s in samples if s['metadata']['question_type'] != 'negative']
    elif sample_type == "negative":
        samples = [s for s in samples if s['metadata']['question_type'] == 'negative']
    
    print(f"Testing on {len(samples)} samples...")
    
    results = []
    
    for sample in tqdm(samples, desc=f"Testing {adapter_name}"):
        # Extract fields
        context = sample['context']
        question = sample['question']
        reference_answer = sample['reference_answer']
        question_type = sample['metadata']['question_type']
        
        # Create prompt
        prompt = create_prompt(context, question)
        
        # Get model response
        model_answer = call_vllm(adapter_name, prompt)
        
        # Analyze response
        is_negative_sample = (question_type == 'negative')
        model_refused = is_refusal(model_answer)
        
        # Determine correctness
        if is_negative_sample:
            # For negative samples: should refuse
            correct = model_refused
            error_type = None if correct else "hallucination"
        else:
            # For positive samples: should NOT refuse
            correct = not model_refused
            error_type = None if correct else "false_negative"
        
        result = {
            'question': question,
            'context': context[:200] + "...",  # Truncate for logging
            'reference_answer': reference_answer,
            'model_answer': model_answer,
            'question_type': question_type,
            'is_negative_sample': is_negative_sample,
            'model_refused': model_refused,
            'correct': correct,
            'error_type': error_type
        }
        
        results.append(result)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    return {
        'adapter': adapter_name,
        'results': results,
        'metrics': metrics
    }


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics from results."""
    total = len(results)
    
    # Overall
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total if total > 0 else 0
    
    # Split by sample type
    positive_results = [r for r in results if not r['is_negative_sample']]
    negative_results = [r for r in results if r['is_negative_sample']]
    
    # Positive samples (should answer)
    positive_correct = sum(1 for r in positive_results if r['correct'])
    positive_accuracy = positive_correct / len(positive_results) if positive_results else 0
    false_negatives = sum(1 for r in positive_results if r['error_type'] == 'false_negative')
    false_negative_rate = false_negatives / len(positive_results) if positive_results else 0
    
    # Negative samples (should refuse)
    negative_correct = sum(1 for r in negative_results if r['correct'])
    negative_accuracy = negative_correct / len(negative_results) if negative_results else 0
    hallucinations = sum(1 for r in negative_results if r['error_type'] == 'hallucination')
    hallucination_rate = hallucinations / len(negative_results) if negative_results else 0
    
    return {
        'total_samples': total,
        'overall_accuracy': accuracy,
        
        'positive_samples': len(positive_results),
        'positive_accuracy': positive_accuracy,
        'false_negatives': false_negatives,
        'false_negative_rate': false_negative_rate,
        
        'negative_samples': len(negative_results),
        'negative_accuracy': negative_accuracy,
        'hallucinations': hallucinations,
        'hallucination_rate': hallucination_rate,
    }


def print_comparison(eval_v1: Dict, eval_v2: Dict):
    """Print side-by-side comparison of both adapters."""
    print("\n" + "="*80)
    print("A/B TEST RESULTS: ADAPTER COMPARISON")
    print("="*80)
    
    m1 = eval_v1['metrics']
    m2 = eval_v2['metrics']
    
    print(f"\n{'Metric':<40} {'v1 (baseline)':<20} {'v2 (+ negatives)':<20}")
    print("-"*80)
    
    # Overall
    print(f"{'Overall Accuracy':<40} {m1['overall_accuracy']:>18.2%} {m2['overall_accuracy']:>20.2%}")
    
    # Positive Samples
    print(f"\n{'POSITIVE SAMPLES (should answer):':<40}")
    print(f"{'  Samples':<40} {m1['positive_samples']:>18} {m2['positive_samples']:>20}")
    print(f"{'  Accuracy':<40} {m1['positive_accuracy']:>18.2%} {m2['positive_accuracy']:>20.2%}")
    print(f"{'  False Negatives':<40} {m1['false_negatives']:>18} {m2['false_negatives']:>20}")
    print(f"{'  False Negative Rate':<40} {m1['false_negative_rate']:>18.2%} {m2['false_negative_rate']:>20.2%}")
    
    # Negative Samples
    print(f"\n{'NEGATIVE SAMPLES (should refuse):':<40}")
    print(f"{'  Samples':<40} {m1['negative_samples']:>18} {m2['negative_samples']:>20}")
    print(f"{'  Accuracy (refusal rate)':<40} {m1['negative_accuracy']:>18.2%} {m2['negative_accuracy']:>20.2%}")
    print(f"{'  Hallucinations':<40} {m1['hallucinations']:>18} {m2['hallucinations']:>20}")
    print(f"{'  Hallucination Rate':<40} {m1['hallucination_rate']:>18.2%} {m2['hallucination_rate']:>20.2%}")
    
    # Key Improvements
    print(f"\n{'KEY IMPROVEMENTS (v2 vs v1):':<40}")
    
    halluc_improvement = m1['hallucination_rate'] - m2['hallucination_rate']
    print(f"{'  Hallucination Reduction':<40} {halluc_improvement:>38.2%}")
    
    false_neg_change = m2['false_negative_rate'] - m1['false_negative_rate']
    if false_neg_change > 0:
        print(f"{'  False Negative Increase (trade-off)':<40} {false_neg_change:>38.2%}")
    else:
        print(f"{'  False Negative Reduction (bonus!)':<40} {-false_neg_change:>38.2%}")
    
    print("="*80)


def save_results(eval_v1: Dict, eval_v2: Dict, output_dir: str):
    """Save detailed results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(output_path / 'eval_v1_results.json', 'w') as f:
        json.dump(eval_v1, f, indent=2)
    
    with open(output_path / 'eval_v2_results.json', 'w') as f:
        json.dump(eval_v2, f, indent=2)
    
    # Save metrics summary
    summary = {
        'v1': eval_v1['metrics'],
        'v2': eval_v2['metrics'],
        'improvements': {
            'hallucination_reduction': eval_v1['metrics']['hallucination_rate'] - eval_v2['metrics']['hallucination_rate'],
            'false_negative_change': eval_v2['metrics']['false_negative_rate'] - eval_v1['metrics']['false_negative_rate'],
        }
    }
    
    with open(output_path / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save error cases for analysis
    v1_errors = [r for r in eval_v1['results'] if not r['correct']]
    v2_errors = [r for r in eval_v2['results'] if not r['correct']]
    
    with open(output_path / 'v1_errors.json', 'w') as f:
        json.dump(v1_errors, f, indent=2)
    
    with open(output_path / 'v2_errors.json', 'w') as f:
        json.dump(v2_errors, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}/")


def main():
    """Run A/B test evaluation."""
    print("="*80)
    print("MULTI-LORA A/B TESTING")
    print("="*80)
    print(f"Eval Dataset: {EVAL_DATASET_PATH}")
    print(f"Adapter v1: {ADAPTER_V1} (baseline, no negative samples)")
    print(f"Adapter v2: {ADAPTER_V2} (with negative samples)")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    
    # Load dataset
    print("\nLoading eval dataset...")
    samples = load_eval_dataset(EVAL_DATASET_PATH)
    
    # Split by type
    positive_samples = [s for s in samples if s['metadata']['question_type'] != 'negative']
    negative_samples = [s for s in samples if s['metadata']['question_type'] == 'negative']
    
    print(f"Total samples: {len(samples)}")
    print(f"  Positive (answerable): {len(positive_samples)}")
    print(f"  Negative (unanswerable): {len(negative_samples)}")
    
    # Evaluate both adapters
    eval_v1 = evaluate_adapter(ADAPTER_V1, samples)
    eval_v2 = evaluate_adapter(ADAPTER_V2, samples)
    
    # Print comparison
    print_comparison(eval_v1, eval_v2)
    
    # Save results
    save_results(eval_v1, eval_v2, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review detailed results in {OUTPUT_DIR}/")
    print(f"2. Analyze error cases:")
    print(f"   - v1_errors.json: Where v1 failed")
    print(f"   - v2_errors.json: Where v2 failed")
    print(f"3. Check if v2 improvements are statistically significant")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()