#!/usr/bin/env python3
"""
LLM-as-Judge with Ollama: Evaluate QA pairs using local Llama-70B
Uses Ollama API for local model inference
"""

import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


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


def judge_with_ollama(
    system_prompt: str,
    chunk: str,
    question: str,
    answer: str,
    model: str = "llama3.1-70b-4k",
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Use Ollama-hosted model as judge."""
    user_prompt = create_evaluation_prompt(chunk, question, answer)
    
    # Combine system and user prompt (Ollama format)
    combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    # Ollama API call
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": combined_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 300  # max_tokens equivalent
            }
        },
        timeout=120  # 2 minutes timeout
    )
    
    if response.status_code != 200:
        print(f"Ollama API error: {response.status_code} - {response.text}")
        return {
            "rating": "ERROR",
            "hallucination": None,
            "reasoning": f"API error: {response.status_code}"
        }
    
    response_data = response.json()
    response_text = response_data.get("response", "").strip()
    
    # Handle potential markdown code blocks
    if response_text.startswith("```json"):
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif response_text.startswith("```"):
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {response_text[:200]}")
        return {
            "rating": "ERROR",
            "hallucination": None,
            "reasoning": f"Failed to parse: {response_text[:100]}"
        }


def run_evaluation(
    samples_path: str,
    output_path: str,
    judge_prompt_path: str,
    model: str,
    ollama_url: str,
    temperature: float,
    limit: int = None
) -> None:
    """Run evaluation on all samples."""
    
    # Load
    print(f"Loading judge prompt from {judge_prompt_path}...")
    system_prompt = load_judge_prompt(judge_prompt_path)
    
    print(f"Loading samples from {samples_path}...")
    samples = load_qa_samples(samples_path)
    
    if limit:
        samples = samples[:limit]
        print(f"Limited to first {limit} samples")
    
    print(f"\nEvaluating {len(samples)} samples with Ollama model: {model}")
    print(f"Ollama URL: {ollama_url}")
    print(f"Temperature: {temperature}\n")
    
    # Evaluate
    results = []
    errors = 0
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        chunk = sample.get("chunk", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        model_name = sample.get("model", "unknown")
        
        # Judge
        judgment = judge_with_ollama(
            system_prompt=system_prompt,
            chunk=chunk,
            question=question,
            answer=answer,
            model=model,
            ollama_url=ollama_url,
            temperature=temperature
        )
        
        if judgment.get("rating") == "ERROR":
            errors += 1
        
        # Store result
        result = {
            "sample_id": idx,
            "model": model_name,
            "chunk": chunk,
            "question": question,
            "answer": answer,
            "judge_model": model,
            "rating": judgment.get("rating"),
            "hallucination": judgment.get("hallucination"),
            "reasoning": judgment.get("reasoning", "")
        }
        results.append(result)
        
        # Save incrementally (in case of crash)
        if (idx + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
    
    # Final save
    print(f"\n\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Errors: {errors}")
    
    # Rating distribution
    rating_counts = {}
    for r in results:
        rating = r.get("rating", "UNKNOWN")
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    print(f"\nRating Distribution:")
    for rating, count in sorted(rating_counts.items()):
        pct = count / len(results) * 100
        print(f"  {rating}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA pairs using Ollama-hosted LLM as judge"
    )
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Path to JSONL file with QA samples"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file for results"
    )
    parser.add_argument(
        "--judge-prompt",
        type=str,
        default="judge_prompt.txt",
        help="Path to judge system prompt (default: judge_prompt.txt)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1-70b-4k",
        help="Ollama model name (default: llama3.1-70b-4k)"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        samples_path=args.samples,
        output_path=args.output,
        judge_prompt_path=args.judge_prompt,
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        limit=args.limit
    )


if __name__ == "__main__":
    main()