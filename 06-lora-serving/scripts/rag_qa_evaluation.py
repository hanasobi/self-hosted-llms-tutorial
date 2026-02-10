#!/usr/bin/env python3
"""
RAG-QA Evaluation Script
Tests Base Model vs LoRA Adapter on AWS certification Q&A tasks
"""

import json
import requests
import time
from datetime import datetime

# Configuration
VLLM_URL = "http://vllm-service:8000/v1/completions"
BASE_MODEL = "TheBloke/Mistral-7B-v0.1-AWQ"
LORA_MODEL = "aws-rag-qa"
MAX_TOKENS = 200
TEMPERATURE = 0.7

def send_request(model, prompt):
    """Send a completion request to vLLM"""
    try:
        start = time.time()
        response = requests.post(
            VLLM_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            },
            timeout=30
        )
        latency = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            text = data['choices'][0]['text']
            return text, latency, None
        else:
            return None, latency, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return None, 0, str(e)

def run_evaluation(test_file):
    """Run evaluation on all test cases"""
    
    # Load test cases
    print(f"Loading test cases from {test_file}...")
    test_cases = []
    with open(test_file, 'r') as f:
        for line in f:
            test_cases.append(json.loads(line))
    
    print(f"Loaded {len(test_cases)} test cases\n")
    print("=" * 80)
    
    results = []
    
    for idx, test_case in enumerate(test_cases, 1):
        service = test_case['metadata']['service']
        q_type = test_case['metadata']['question_type']
        question = test_case['question']
        reference = test_case['reference_answer']
        prompt = test_case['prompt_inference']
        
        print(f"\n[Test {idx}/15] Service: {service} | Type: {q_type}")
        print(f"Question: {question[:80]}...")
        print("-" * 80)
        
        # Test Base Model
        print("Testing Base Model...", end=" ", flush=True)
        base_answer, base_latency, base_error = send_request(BASE_MODEL, prompt)
        if base_error:
            print(f"ERROR: {base_error}")
        else:
            print(f"Done ({base_latency:.2f}s)")
        
        # Wait a bit between requests
        time.sleep(1)
        
        # Test LoRA Adapter
        print("Testing LoRA Adapter...", end=" ", flush=True)
        lora_answer, lora_latency, lora_error = send_request(LORA_MODEL, prompt)
        if lora_error:
            print(f"ERROR: {lora_error}")
        else:
            print(f"Done ({lora_latency:.2f}s)")
        
        # Store results
        result = {
            "test_id": idx,
            "metadata": test_case['metadata'],
            "question": question,
            "reference_answer": reference,
            "base_model": {
                "answer": base_answer,
                "latency": base_latency,
                "error": base_error
            },
            "lora_adapter": {
                "answer": lora_answer,
                "latency": lora_latency,
                "error": lora_error
            }
        }
        results.append(result)
        
        # Show preview
        if base_answer and lora_answer:
            print(f"\nBase:  {base_answer[:100]}...")
            print(f"LoRA:  {lora_answer[:100]}...")
            print(f"Ref:   {reference[:100]}...")
        
        print("=" * 80)
    
    return results

def save_results(results, output_file):
    """Save results to JSON file"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": BASE_MODEL,
            "lora_model": LORA_MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        },
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")

def print_summary(results):
    """Print summary statistics"""
    base_latencies = [r['base_model']['latency'] for r in results if r['base_model']['answer']]
    lora_latencies = [r['lora_adapter']['latency'] for r in results if r['lora_adapter']['answer']]
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(results)}")
    print(f"Base Model - Avg latency: {sum(base_latencies)/len(base_latencies):.2f}s")
    print(f"LoRA Adapter - Avg latency: {sum(lora_latencies)/len(lora_latencies):.2f}s")
    print("\nBy question type:")
    
    for q_type in ['factual', 'comparison', 'conceptual']:
        type_results = [r for r in results if r['metadata']['question_type'] == q_type]
        print(f"  {q_type.capitalize()}: {len(type_results)} tests")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 rag_qa_test.py <test_file.jsonl> [output_file.json]")
        sys.exit(1)
    
    test_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "evaluation_results.json"
    
    print("RAG-QA Evaluation")
    print("=" * 80)
    print(f"Base Model: {BASE_MODEL}")
    print(f"LoRA Model: {LORA_MODEL}")
    print(f"Test File: {test_file}")
    print(f"Output File: {output_file}")
    
    # Run evaluation
    results = run_evaluation(test_file)
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print_summary(results)
    
    print("\n✓ Evaluation complete!")