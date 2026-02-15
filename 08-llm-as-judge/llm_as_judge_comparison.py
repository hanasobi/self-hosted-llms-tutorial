#!/usr/bin/env python3
"""
LLM-as-Judge Comparison: Test multiple judges on the same samples
Supports: Llama (vLLM), GPT-4o (OpenAI), Claude Sonnet (Anthropic)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI
from anthropic import Anthropic


def load_judge_prompt(prompt_path: str = "judge_prompt.txt") -> str:
    """Load the system prompt for the judge."""
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def load_qa_samples(samples_path: str) -> List[Dict[str, Any]]:
    """Load QA samples from JSONL file."""
    samples = []
    with open(samples_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def create_evaluation_prompt(chunk: str, question: str, answer: str) -> str:
    """Create the user prompt for evaluation."""
    return f"""CHUNK:
{chunk}

QUESTION:
{question}

ANSWER:
{answer}

Evaluate this QA pair and respond with JSON only."""


def judge_with_mistral(
    client: OpenAI,
    system_prompt: str,
    chunk: str,
    question: str,
    answer: str,
    model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    temperature: float = 0.0,
    max_tokens: int = 300
) -> Dict[str, Any]:
    """Use Mistral via OpenAI-compatible API as judge. Combines system+user prompt."""
    user_prompt = create_evaluation_prompt(chunk, question, answer)
    
    # Mistral doesn't support system prompt - combine into user message
    combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": combined_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Handle potential markdown code blocks
    if response_text.startswith("```json"):
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif response_text.startswith("```"):
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {response_text}")
        return {
            "rating": "ERROR",
            "hallucination": None,
            "reasoning": f"Failed to parse: {response_text[:100]}"
        }


def judge_with_openai(
    client: OpenAI,
    system_prompt: str,
    chunk: str,
    question: str,
    answer: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 300
) -> Dict[str, Any]:
    """Use OpenAI API (GPT-4o or vLLM) as judge."""
    user_prompt = create_evaluation_prompt(chunk, question, answer)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Handle potential markdown code blocks
    if response_text.startswith("```json"):
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif response_text.startswith("```"):
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {response_text}")
        return {
            "rating": "ERROR",
            "hallucination": None,
            "reasoning": f"Failed to parse: {response_text[:100]}"
        }


def judge_with_claude(
    client: Anthropic,
    system_prompt: str,
    chunk: str,
    question: str,
    answer: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    max_tokens: int = 300
) -> Dict[str, Any]:
    """Use Anthropic Claude API as judge."""
    user_prompt = create_evaluation_prompt(chunk, question, answer)
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    response_text = response.content[0].text.strip()
    
    # Handle potential markdown code blocks
    if response_text.startswith("```json"):
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif response_text.startswith("```"):
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {response_text}")
        return {
            "rating": "ERROR",
            "hallucination": None,
            "reasoning": f"Failed to parse: {response_text[:100]}"
        }


def evaluate_with_judge(
    samples: List[Dict[str, Any]],
    judge_name: str,
    system_prompt: str,
    judge_config: Dict[str, Any],
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Evaluate samples with a specific judge.
    
    Args:
        judge_name: Name prefix for results (e.g., "llama", "gpt4o", "claude")
        judge_config: Dict with "type", "client", "model"
    """
    results = []
    
    for i, sample in enumerate(samples, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{judge_name.upper()}] Evaluating Sample {i}/{len(samples)}")
            print(f"{'='*60}")
        
        chunk = sample.get("chunk", sample.get("source_chunk", ""))
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        
        # Judge based on type
        if judge_config["type"] == "openai":
            evaluation = judge_with_openai(
                client=judge_config["client"],
                system_prompt=system_prompt,
                chunk=chunk,
                question=question,
                answer=answer,
                model=judge_config["model"]
            )
        elif judge_config["type"] == "mistral":
            evaluation = judge_with_mistral(
                client=judge_config["client"],
                system_prompt=system_prompt,
                chunk=chunk,
                question=question,
                answer=answer,
                model=judge_config["model"]
            )
        elif judge_config["type"] == "claude":
            evaluation = judge_with_claude(
                client=judge_config["client"],
                system_prompt=system_prompt,
                chunk=chunk,
                question=question,
                answer=answer,
                model=judge_config["model"]
            )
        else:
            raise ValueError(f"Unknown judge type: {judge_config['type']}")
        
        # Combine original sample with evaluation
        result = {
            **sample,
            f"{judge_name}_rating": evaluation.get("rating"),
            f"{judge_name}_hallucination": evaluation.get("hallucination"),
            f"{judge_name}_reasoning": evaluation.get("reasoning")
        }
        
        results.append(result)
        
        if verbose:
            print(f"Rating: {evaluation.get('rating')}")
            print(f"Hallucination: {evaluation.get('hallucination')}")
            print(f"Reasoning: {evaluation.get('reasoning')}")
    
    return results


def merge_results(
    base_results: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge results from multiple judges into single records."""
    merged = []
    for base, new in zip(base_results, new_results):
        merged_record = {**base, **new}
        merged.append(merged_record)
    return merged


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save evaluation results to JSONL file."""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\n✅ Results saved to: {output_path}")


def compute_statistics(results: List[Dict[str, Any]], judge_name: str) -> Dict[str, Any]:
    """Compute statistics for a specific judge."""
    total = len(results)
    rating_key = f"{judge_name}_rating"
    hallucination_key = f"{judge_name}_hallucination"
    
    rating_counts = {"A": 0, "B": 0, "C": 0, "ERROR": 0}
    for r in results:
        rating = r.get(rating_key, "ERROR")
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    hallucination_count = sum(1 for r in results if r.get(hallucination_key) == True)
    
    return {
        "judge": judge_name,
        "total": total,
        "ratings": rating_counts,
        "a_percent": round(100 * rating_counts["A"] / total, 1),
        "b_percent": round(100 * rating_counts["B"] / total, 1),
        "c_percent": round(100 * rating_counts["C"] / total, 1),
        "hallucinations": hallucination_count
    }


def print_comparison(all_stats: List[Dict[str, Any]]):
    """Print comparison table of all judges."""
    print(f"\n{'='*80}")
    print("JUDGE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Judge':<15} {'A-Rating':<12} {'B-Rating':<12} {'C-Rating':<12} {'Halluc.':<10}")
    print(f"{'-'*80}")
    
    for stats in all_stats:
        print(f"{stats['judge'].upper():<15} "
              f"{stats['ratings']['A']:>3} ({stats['a_percent']:>5.1f}%)  "
              f"{stats['ratings']['B']:>3} ({stats['b_percent']:>5.1f}%)  "
              f"{stats['ratings']['C']:>3} ({stats['c_percent']:>5.1f}%)  "
              f"{stats['hallucinations']:>3}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple LLM judges on the same samples")
    parser.add_argument("--samples", required=True, help="Path to JSONL file with QA samples")
    parser.add_argument("--output", required=True, help="Path to save comparison results (JSONL)")
    parser.add_argument("--prompt", default="judge_prompt.txt", help="Path to judge prompt")
    
    # Judge configurations
    parser.add_argument("--llama-url", help="vLLM endpoint URL for Llama")
    parser.add_argument("--llama-model", default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    
    parser.add_argument("--mistral-url", help="vLLM endpoint URL for Mistral")
    parser.add_argument("--mistral-model", default="solidrust/Mistral-7B-Instruct-v0.3-AWQ")
    
    parser.add_argument("--gpt4o", action="store_true", help="Include GPT-4o as judge")
    parser.add_argument("--gpt4o-key", help="OpenAI API key")
    
    parser.add_argument("--claude", action="store_true", help="Include Claude Sonnet as judge")
    parser.add_argument("--claude-key", help="Anthropic API key")
    
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Load resources
    print(f"Loading judge prompt from: {args.prompt}")
    system_prompt = load_judge_prompt(args.prompt)
    
    print(f"Loading samples from: {args.samples}")
    samples = load_qa_samples(args.samples)
    print(f"Loaded {len(samples)} samples")
    
    # Setup judges
    judges = {}
    all_results = None
    all_stats = []
    
    # Llama judge
    if args.llama_url:
        print(f"\n{'='*60}")
        print("Setting up Llama judge...")
        print(f"{'='*60}")
        judges["llama"] = {
            "type": "openai",
            "client": OpenAI(base_url=args.llama_url, api_key="token-abc123"),
            "model": args.llama_model
        }
    
    # Mistral judge
    if args.mistral_url:
        print(f"\n{'='*60}")
        print("Setting up Mistral judge...")
        print(f"{'='*60}")
        judges["mistral"] = {
            "type": "mistral",
            "client": OpenAI(base_url=args.mistral_url, api_key="token-abc123"),
            "model": args.mistral_model
        }
    
    # GPT-4o judge
    if args.gpt4o:
        print(f"\n{'='*60}")
        print("Setting up GPT-4o judge...")
        print(f"{'='*60}")
        if not args.gpt4o_key:
            raise ValueError("--gpt4o-key required when --gpt4o is enabled")
        judges["gpt4o"] = {
            "type": "openai",
            "client": OpenAI(api_key=args.gpt4o_key),
            "model": "gpt-4o"
        }
    
    # Claude judge
    if args.claude:
        print(f"\n{'='*60}")
        print("Setting up Claude judge...")
        print(f"{'='*60}")
        if not args.claude_key:
            raise ValueError("--claude-key required when --claude is enabled")
        judges["claude"] = {
            "type": "claude",
            "client": Anthropic(api_key=args.claude_key),
            "model": "claude-sonnet-4-20250514"
        }
    
    if not judges:
        raise ValueError("At least one judge must be specified (--llama-url, --mistral-url, --gpt4o, or --claude)")
    
    # Run evaluations
    for judge_name, judge_config in judges.items():
        print(f"\n{'='*60}")
        print(f"Running {judge_name.upper()} evaluation...")
        print(f"{'='*60}")
        
        results = evaluate_with_judge(
            samples=samples,
            judge_name=judge_name,
            system_prompt=system_prompt,
            judge_config=judge_config,
            verbose=not args.quiet
        )
        
        # Merge results
        if all_results is None:
            all_results = results
        else:
            all_results = merge_results(all_results, results)
        
        # Compute stats
        stats = compute_statistics(all_results, judge_name)
        all_stats.append(stats)
    
    # Save results
    save_results(all_results, args.output)
    
    # Print comparison
    print_comparison(all_stats)


if __name__ == "__main__":
    main()