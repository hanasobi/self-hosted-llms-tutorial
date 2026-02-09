"""
Generate and Inspect Sample Responses from Fine-tuned Model

This script loads the fine-tuned model, generates answers for a stratified sample
of questions from eval.jsonl, and outputs them in a readable format for manual
inspection. This helps us:
1. Understand if the model actually generates good answers
2. Diagnose if low loss values are real or due to bugs
3. Identify typical error patterns before building automated evaluation

The script can be run from either:
- Project root: 05-lora-training/scripts/inspect_model_response.py
- Scripts dir: python inspect_model_response.py

Paths are automatically resolved relative to project root.

Usage:
    # With defaults (uses standard_r8_qkvo model)
    python inspect_model_response.py
    
    # Or specify different model
    python inspect_model_response.py \
        --model_path 05-lora-training/models/standard_r8_qkvo \
        --num_samples 20
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Works whether script is run from:
    - project_root/
    - project_root/05.2-model-evaluation/scripts/
    
    Returns:
        Path to project root (llm-fundamentals/)
    """
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Check if we're in scripts/ directory
    if script_dir.name == 'scripts':
        # Navigate up: scripts -> 05.2-model-evaluation -> project_root
        project_root = script_dir.parent.parent
    else:
        # Assume we're already at project root
        project_root = Path.cwd()
    
    return project_root


def resolve_path(path: str, project_root: Path) -> Path:
    """
    Resolve a path relative to project root.
    
    Args:
        path: Path string (can be relative or absolute)
        project_root: Project root directory
    
    Returns:
        Absolute path
    """
    path_obj = Path(path)
    
    # If already absolute, return as is
    if path_obj.is_absolute():
        return path_obj
    
    # Otherwise, make it relative to project_root
    # First try with 05-lora-training prefix
    full_path = project_root / '05-lora-training' / path
    if full_path.exists():
        return full_path
    
    # Then try without prefix (for outputs)
    full_path = project_root / path
    return full_path


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def stratified_sample(data: List[Dict], num_samples: int, stratify_key: str = 'question_type') -> List[Dict]:
    """
    Sample data stratified by a given key.
    
    For example, if we have 3 question types (factual, conceptual, comparison)
    and want 15 samples, we'll get 5 of each type.
    
    Args:
        data: List of data records
        num_samples: Total number of samples to return
        stratify_key: Key in metadata to stratify by
    
    Returns:
        Stratified sample of records
    """
    # Group by stratify key
    groups = defaultdict(list)
    for record in data:
        key = record['metadata'].get(stratify_key, 'unknown')
        groups[key].append(record)
    
    # Calculate samples per group
    num_groups = len(groups)
    samples_per_group = num_samples // num_groups
    
    # Sample from each group
    sampled = []
    for group_name, group_records in sorted(groups.items()):
        # Take random sample from this group
        n = min(samples_per_group, len(group_records))
        group_sample = random.sample(group_records, n)
        sampled.extend(group_sample)
    
    # If we need a few more to reach num_samples (due to integer division)
    remaining = num_samples - len(sampled)
    if remaining > 0:
        # Sample randomly from all data not yet sampled
        not_sampled = [r for r in data if r not in sampled]
        if not_sampled:
            sampled.extend(random.sample(not_sampled, min(remaining, len(not_sampled))))
    
    return sampled[:num_samples]


def load_model(model_path: str, base_model_name: str = "mistralai/Mistral-7B-v0.1"):
    """
    Load fine-tuned model with LoRA adapter.
    
    Uses 4-bit quantization to fit the model on GPU without offloading.
    Merges LoRA adapter for faster inference.
    
    Args:
        model_path: Path to directory containing LoRA adapter
        base_model_name: Name or path of base model
    
    Returns:
        model, tokenizer
    """
    from transformers import BitsAndBytesConfig
    
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    # Configure 4-bit quantization to fit on GPU without offloading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapter if path exists
    adapter_path = Path(model_path)
    if adapter_path.exists() and (adapter_path / 'adapter_config.json').exists():
        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Merge LoRA weights for faster inference
        # This takes ~30 seconds but makes generation 3-5x faster
        print("Merging LoRA weights (this takes ~30 seconds)...")
        model = model.merge_and_unload()
        print("LoRA weights merged - inference will be faster now")
    else:
        print(f"Warning: No LoRA adapter found at {model_path}, using base model")
        model = base_model
    
    model.eval()
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    verbose: bool = True
) -> str:
    """
    Generate answer for a given prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt (should be prompt_inference from eval.jsonl)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample or use greedy decoding
        verbose: Print timing and token statistics
    
    Returns:
        Generated answer text
    """
    import time
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.unk_token_id
        )
    
    generation_time = time.time() - start_time
    
    # Decode only the newly generated tokens
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Calculate stats
    num_tokens = len(generated_tokens)
    tokens_per_sec = num_tokens / generation_time if generation_time > 0 else 0
    
    if verbose:
        print(f"    Generated {num_tokens} tokens in {generation_time:.2f}s "
              f"({tokens_per_sec:.1f} tok/s)")
    
    return answer.strip()



def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def format_output_markdown(samples_with_answers: List[Dict], output_path: str):
    """
    Format samples and their generated answers as Markdown for easy reading.
    
    Args:
        samples_with_answers: List of dicts with sample data and generated answer
        output_path: Where to save the markdown file
    """
    md_lines = []
    
    # Header
    md_lines.append("# Model Response Inspection")
    md_lines.append("")
    md_lines.append(f"Generated {len(samples_with_answers)} sample responses for manual inspection.")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # Each sample
    for i, sample in enumerate(samples_with_answers, 1):
        metadata = sample['metadata']
        
        md_lines.append(f"## Sample {i}")
        md_lines.append("")
        md_lines.append(f"**Service:** {metadata['service']}  ")
        md_lines.append(f"**Question Type:** {metadata['question_type']}  ")
        md_lines.append(f"**Chunk ID:** {metadata['chunk_id']}")
        md_lines.append("")
        
        # Context (truncated)
        md_lines.append("### Context")
        md_lines.append("```")
        #md_lines.append(truncate_text(sample['context'], max_chars=1024))
        md_lines.append(sample['context'])
        md_lines.append("```")
        md_lines.append("")
        
        # Question
        md_lines.append("### Question")
        md_lines.append(f"> {sample['question']}")
        md_lines.append("")
        
        # Reference Answer
        md_lines.append("### Reference Answer")
        md_lines.append(sample['reference_answer'])
        md_lines.append("")
        
        # Generated Answer
        md_lines.append("### Generated Answer")
        md_lines.append(sample['generated_answer'])
        md_lines.append("")
        
        # Space for manual notes
        md_lines.append("### Manual Assessment")
        md_lines.append("- [ ] Factually correct")
        md_lines.append("- [ ] No hallucinations")
        md_lines.append("- [ ] Complete answer")
        md_lines.append("- [ ] Appropriate length")
        md_lines.append("")
        md_lines.append("**Notes:**")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Summary section
    md_lines.append("## Summary Notes")
    md_lines.append("")
    md_lines.append("### Patterns Observed")
    md_lines.append("- ")
    md_lines.append("")
    md_lines.append("### Typical Errors")
    md_lines.append("- ")
    md_lines.append("")
    md_lines.append("### Overall Assessment")
    md_lines.append("")
    
    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\nMarkdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample responses for manual inspection"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='05-lora-training/models/standard_r8_qkvo',
        help='Path to fine-tuned model relative to 05-lora-training/ (e.g., block2_lora_finetuning/models/standard_r8_qkvo)'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='Base model name or path'
    )
    parser.add_argument(
        '--eval_data',
        type=str,
        default='data/processed/eval.jsonl',
        help='Path to eval.jsonl relative to 05-lora-training/'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=15,
        help='Number of samples to generate (default: 15)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='05.2-model-evaluation/output/responses/sample_responses.md',
        help='Output markdown file path relative to project root'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate per answer'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.0 = greedy, higher = more random)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling'
    )
    
    args = parser.parse_args()
    
    # Get project root and resolve all paths
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    model_path = resolve_path(args.model_path, project_root)
    eval_data_path = resolve_path(args.eval_data, project_root)
    output_path = resolve_path(args.output, project_root)
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("MODEL RESPONSE INSPECTION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_path}")
    print("="*80 + "\n")
    
    # Load eval data
    print("Loading evaluation data...")
    eval_data = load_jsonl(str(eval_data_path))
    print(f"Loaded {len(eval_data)} evaluation samples")
    
    # Stratified sampling
    print(f"\nSampling {args.num_samples} examples (stratified by question_type)...")
    samples = stratified_sample(eval_data, args.num_samples)
    
    # Show distribution
    type_counts = defaultdict(int)
    for s in samples:
        type_counts[s['metadata']['question_type']] += 1
    print("Sample distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(str(model_path), args.base_model)
    print("Model loaded successfully")
    
    # Generate answers
    print(f"\nGenerating answers for {len(samples)} samples...")
    print("(Timing info will show tokens/sec - expect 15-25 tok/s on T4 after LoRA merge)")
    samples_with_answers = []
    
    for i, sample in enumerate(samples, 1):
        print(f"\n  [{i}/{len(samples)}] {sample['metadata']['chunk_id']} "
              f"({sample['metadata']['question_type']})")
        
        # Use pre-formatted inference prompt
        prompt = sample['prompt_inference']
        
        # Generate (verbose=True shows timing)
        answer = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            verbose=True
        )
        
        # Store
        sample_with_answer = {
            'context': sample['context'],
            'question': sample['question'],
            'reference_answer': sample['reference_answer'],
            'generated_answer': answer,
            'metadata': sample['metadata']
        }
        samples_with_answers.append(sample_with_answer)
    
    # Format and save output
    print("\nFormatting output...")
    format_output_markdown(samples_with_answers, str(output_path))
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(samples_with_answers)} responses")
    print(f"Output saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Open the markdown file and manually review each response")
    print("  2. Check the boxes for quality criteria")
    print("  3. Note patterns in the Summary section")
    print("  4. Use insights to design LLM-as-Judge prompts")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()