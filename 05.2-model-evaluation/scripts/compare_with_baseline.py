"""
Compare Fine-tuned Model vs. Mistral Instruct Baseline

This script generates responses from both models on the same evaluation samples
and produces a side-by-side comparison report.

Usage:
    python compare_with_baseline.py \
        --finetuned_path ../../05-lora-training/models/standard_r8_qkvo \
        --eval_data ../../data/processed/eval.jsonl \
        --num_samples 15 \
        --output_dir ../output/baseline_comparison
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


def load_finetuned_model(model_path: str, base_model_name: str = "mistralai/Mistral-7B-v0.1"):
    """Load fine-tuned model with LoRA adapter."""
    print(f"\n{'='*80}")
    print("LOADING FINE-TUNED MODEL")
    print(f"{'='*80}")
    print(f"Base: {base_model_name}")
    print(f"LoRA adapter: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
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
    
    adapter_path = Path(model_path)
    if adapter_path.exists() and (adapter_path / 'adapter_config.json').exists():
        print(f"Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        print(f"Warning: No LoRA adapter found, using base model")
        model = base_model
    
    model.eval()
    print("Fine-tuned model loaded ✓")
    return model, tokenizer


def load_instruct_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Load Mistral Instruct baseline model."""
    print(f"\n{'='*80}")
    print("LOADING MISTRAL INSTRUCT BASELINE")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    model.eval()
    print("Instruct baseline loaded ✓")
    return model, tokenizer


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def stratified_sample(data: List[Dict], num_samples: int) -> List[Dict]:
    """Sample data stratified by question type."""
    groups = defaultdict(list)
    for record in data:
        qtype = record['metadata'].get('question_type', 'unknown')
        groups[qtype].append(record)
    
    samples_per_group = num_samples // len(groups)
    sampled = []
    
    for group_records in groups.values():
        n = min(samples_per_group, len(group_records))
        sampled.extend(random.sample(group_records, n))
    
    return sampled[:num_samples]


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 128, 
                    temperature: float = 0.3) -> Dict:
    """
    Generate answer and return both text and metadata.
    
    Returns dict with:
        - answer: Generated text
        - num_tokens: Number of tokens generated
        - stopped_naturally: Whether model generated EOS (vs hitting max_tokens)
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Check if model stopped naturally (generated EOS) or hit max_tokens
    stopped_naturally = generated_tokens[-1].item() == tokenizer.eos_token_id
    
    # Post-process: Cut off if model starts generating new questions
    cutoff_markers = ["\n\nQuestion:", "\n[INST]", "\nQuestion:"]
    min_cutoff = len(answer)
    for marker in cutoff_markers:
        pos = answer.find(marker)
        if pos != -1 and pos < min_cutoff:
            min_cutoff = pos
    
    was_truncated = min_cutoff < len(answer)
    if was_truncated:
        answer = answer[:min_cutoff]
    
    return {
        'answer': answer.strip(),
        'num_tokens': len(generated_tokens),
        'stopped_naturally': stopped_naturally,
        'was_truncated': was_truncated
    }


def compare_models_sequential(finetuned_model, finetuned_tokenizer,
                             instruct_model_name: str,
                             eval_samples: List[Dict],
                             max_new_tokens: int = 128,
                             temperature: float = 0.3) -> List[Dict]:
    """
    Generate responses from both models SEQUENTIALLY to avoid OOM.
    
    This loads one model, generates all responses, unloads it,
    then loads the second model. Slower but safer on 16GB GPU.
    """
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL GENERATION (Memory-Safe Mode)")
    print(f"{'='*80}")
    print(f"Samples: {len(eval_samples)}")
    print(f"This will take longer but avoids OOM on 16GB GPU")
    print(f"{'='*80}\n")
    
    # Step 1: Generate from fine-tuned model
    print("Step 1/2: Generating from fine-tuned model...")
    finetuned_results = []
    for sample in tqdm(eval_samples, desc="Fine-tuned model"):
        prompt = sample['prompt_inference']
        result = generate_answer(
            finetuned_model,
            finetuned_tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        finetuned_results.append(result)
    
    # Unload fine-tuned model to free memory
    print("\nUnloading fine-tuned model...")
    del finetuned_model
    del finetuned_tokenizer
    torch.cuda.empty_cache()
    
    # Step 2: Load instruct model and generate
    print("\nStep 2/2: Loading instruct model and generating...")
    instruct_model, instruct_tokenizer = load_instruct_model(instruct_model_name)
    
    instruct_results = []
    for sample in tqdm(eval_samples, desc="Instruct baseline"):
        prompt = sample['prompt_inference']
        result = generate_answer(
            instruct_model,
            instruct_tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        instruct_results.append(result)
    
    # Combine results
    results = []
    for i, sample in enumerate(eval_samples):
        results.append({
            'sample_id': i + 1,
            'metadata': sample['metadata'],
            'context': sample['context'],
            'question': sample['question'],
            'reference_answer': sample['reference_answer'],
            'finetuned': finetuned_results[i],
            'instruct': instruct_results[i],
        })
    
    return results


def compare_models_parallel(finetuned_model, finetuned_tokenizer,
                            instruct_model, instruct_tokenizer,
                            eval_samples: List[Dict],
                            max_new_tokens: int = 128,
                            temperature: float = 0.3) -> List[Dict]:
    """
    Generate responses from both models in PARALLEL.
    
    Faster but requires ~16GB GPU memory. May cause OOM on T4.
    Use sequential mode if this fails.
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"PARALLEL GENERATION (Faster but needs more memory)")
    print(f"{'='*80}")
    print(f"Samples: {len(eval_samples)}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"temperature: {temperature}")
    print(f"{'='*80}\n")
    
    for i, sample in enumerate(tqdm(eval_samples, desc="Comparing models"), 1):
        # Use the pre-formatted inference prompt
        prompt = sample['prompt_inference']
        
        # Generate from fine-tuned model
        finetuned_result = generate_answer(
            finetuned_model, 
            finetuned_tokenizer, 
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Generate from instruct baseline
        instruct_result = generate_answer(
            instruct_model,
            instruct_tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Store comparison
        results.append({
            'sample_id': i,
            'metadata': sample['metadata'],
            'context': sample['context'],
            'question': sample['question'],
            'reference_answer': sample['reference_answer'],
            'finetuned': finetuned_result,
            'instruct': instruct_result,
        })
    
    return results


def format_comparison_markdown(results: List[Dict], output_path: str):
    """Format comparison results as markdown."""
    md_lines = []
    
    # Header
    md_lines.append("# Model Comparison: Fine-tuned vs. Mistral Instruct Baseline")
    md_lines.append("")
    md_lines.append(f"Compared {len(results)} samples")
    md_lines.append("")
    
    # Summary statistics
    md_lines.append("## Summary Statistics")
    md_lines.append("")
    
    finetuned_stopped = sum(1 for r in results if r['finetuned']['stopped_naturally'])
    instruct_stopped = sum(1 for r in results if r['instruct']['stopped_naturally'])
    
    finetuned_truncated = sum(1 for r in results if r['finetuned']['was_truncated'])
    instruct_truncated = sum(1 for r in results if r['instruct']['was_truncated'])
    
    finetuned_avg_tokens = sum(r['finetuned']['num_tokens'] for r in results) / len(results)
    instruct_avg_tokens = sum(r['instruct']['num_tokens'] for r in results) / len(results)
    
    md_lines.append("| Metric | Fine-tuned | Instruct Baseline |")
    md_lines.append("|--------|------------|-------------------|")
    md_lines.append(f"| Stopped naturally (EOS) | {finetuned_stopped}/{len(results)} ({100*finetuned_stopped/len(results):.0f}%) | {instruct_stopped}/{len(results)} ({100*instruct_stopped/len(results):.0f}%) |")
    md_lines.append(f"| Truncated (continuation) | {finetuned_truncated}/{len(results)} ({100*finetuned_truncated/len(results):.0f}%) | {instruct_truncated}/{len(results)} ({100*instruct_truncated/len(results):.0f}%) |")
    md_lines.append(f"| Avg tokens generated | {finetuned_avg_tokens:.1f} | {instruct_avg_tokens:.1f} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # Individual samples
    for result in results:
        md_lines.append(f"## Sample {result['sample_id']}")
        md_lines.append("")
        
        metadata = result['metadata']
        md_lines.append(f"**Service:** {metadata['service']}  ")
        md_lines.append(f"**Question Type:** {metadata['question_type']}  ")
        md_lines.append(f"**Chunk ID:** {metadata['chunk_id']}")
        md_lines.append("")
        
        # Context (truncated)
        md_lines.append("### Context")
        md_lines.append("```")
        context = result['context']
        if len(context) > 1500:
            md_lines.append(context[:1500] + "\n...[truncated]")
        else:
            md_lines.append(context)
        md_lines.append("```")
        md_lines.append("")
        
        # Question
        md_lines.append("### Question")
        md_lines.append(f"> {result['question']}")
        md_lines.append("")
        
        # Reference Answer
        md_lines.append("### Reference Answer")
        md_lines.append(result['reference_answer'])
        md_lines.append("")
        
        # Fine-tuned Answer
        ft = result['finetuned']
        md_lines.append("### Fine-tuned Model Answer")
        md_lines.append(ft['answer'])
        md_lines.append("")
        md_lines.append(f"*Tokens: {ft['num_tokens']}, "
                       f"Stopped naturally: {'✓' if ft['stopped_naturally'] else '✗'}, "
                       f"Truncated: {'✓' if ft['was_truncated'] else '✗'}*")
        md_lines.append("")
        
        # Instruct Baseline Answer
        inst = result['instruct']
        md_lines.append("### Instruct Baseline Answer")
        md_lines.append(inst['answer'])
        md_lines.append("")
        md_lines.append(f"*Tokens: {inst['num_tokens']}, "
                       f"Stopped naturally: {'✓' if inst['stopped_naturally'] else '✗'}, "
                       f"Truncated: {'✓' if inst['was_truncated'] else '✗'}*")
        md_lines.append("")
        
        # Manual assessment
        md_lines.append("### Manual Assessment")
        md_lines.append("")
        md_lines.append("**Fine-tuned:**")
        md_lines.append("- [ ] Factually correct")
        md_lines.append("- [ ] No hallucinations")
        md_lines.append("- [ ] Complete answer")
        md_lines.append("- [ ] Follows instruction (answer only from context)")
        md_lines.append("")
        md_lines.append("**Instruct Baseline:**")
        md_lines.append("- [ ] Factually correct")
        md_lines.append("- [ ] No hallucinations")
        md_lines.append("- [ ] Complete answer")
        md_lines.append("- [ ] Follows instruction (answer only from context)")
        md_lines.append("")
        md_lines.append("**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie")
        md_lines.append("")
        md_lines.append("**Notes:**")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Summary section
    md_lines.append("## Overall Assessment")
    md_lines.append("")
    md_lines.append("### Patterns Observed")
    md_lines.append("")
    md_lines.append("**Fine-tuned model strengths:**")
    md_lines.append("- ")
    md_lines.append("")
    md_lines.append("**Fine-tuned model weaknesses:**")
    md_lines.append("- ")
    md_lines.append("")
    md_lines.append("**Instruct baseline strengths:**")
    md_lines.append("- ")
    md_lines.append("")
    md_lines.append("**Instruct baseline weaknesses:**")
    md_lines.append("- ")
    md_lines.append("")
    md_lines.append("### Win Rate")
    md_lines.append("")
    md_lines.append("Count manually after filling in assessments:")
    md_lines.append("- Fine-tuned wins: ___")
    md_lines.append("- Instruct wins: ___")
    md_lines.append("- Ties: ___")
    md_lines.append("")
    md_lines.append("### Recommendation")
    md_lines.append("")
    md_lines.append("Based on this comparison:")
    md_lines.append("- [ ] Fine-tuning provides significant value → Deploy fine-tuned model")
    md_lines.append("- [ ] Marginal improvement → Consider if fine-tuning effort is worth it")
    md_lines.append("- [ ] No improvement or worse → Use Instruct baseline, investigate why fine-tuning didn't help")
    md_lines.append("")
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n✅ Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare fine-tuned model against Mistral Instruct baseline"
    )
    parser.add_argument(
        '--finetuned_path',
        type=str,
        required=True,
        help='Path to fine-tuned LoRA adapter'
    )
    parser.add_argument(
        '--instruct_model',
        type=str,
        default='mistralai/Mistral-7B-Instruct-v0.2',
        help='Mistral Instruct model to use as baseline'
    )
    parser.add_argument(
        '--eval_data',
        type=str,
        required=True,
        help='Path to eval.jsonl'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/baseline_comparison',
        help='Output directory for comparison report'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=128,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Load both models in parallel (faster but needs ~16GB GPU memory). Default is sequential (safer).'
    )
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("MODEL COMPARISON: FINE-TUNED VS. INSTRUCT BASELINE")
    print("="*80)
    print(f"Fine-tuned model: {args.finetuned_path}")
    print(f"Instruct baseline: {args.instruct_model}")
    print(f"Evaluation data: {args.eval_data}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Loading mode: {'Parallel (risky on 16GB)' if args.parallel else 'Sequential (safe)'}")
    print("="*80)
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    eval_data = load_jsonl(args.eval_data)
    print(f"Loaded {len(eval_data)} evaluation samples")
    
    # Sample
    samples = stratified_sample(eval_data, args.num_samples)
    print(f"Selected {len(samples)} samples (stratified by question type)")
    
    # Load fine-tuned model
    finetuned_model, finetuned_tokenizer = load_finetuned_model(args.finetuned_path)
    
    # Run comparison (sequential or parallel)
    if args.parallel:
        print("\n⚠️  WARNING: Parallel mode may cause OOM on 16GB GPU!")
        print("If you get OOM error, run again without --parallel flag")
        
        # Load both models
        instruct_model, instruct_tokenizer = load_instruct_model(args.instruct_model)
        
        try:
            results = compare_models_parallel(
                finetuned_model,
                finetuned_tokenizer,
                instruct_model,
                instruct_tokenizer,
                samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n❌ OUT OF MEMORY ERROR!")
                print("Your GPU doesn't have enough memory for parallel loading.")
                print("Please run again without --parallel flag for sequential loading.")
                return
            else:
                raise
    else:
        # Sequential mode (safe)
        results = compare_models_sequential(
            finetuned_model,
            finetuned_tokenizer,
            args.instruct_model,
            samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
    
    # Generate report
    output_path = Path(args.output_dir) / 'comparison_report.md'
    format_comparison_markdown(results, str(output_path))
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Report saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review the comparison report")
    print("  2. Fill in manual assessments")
    print("  3. Count wins/ties to determine overall performance")
    print("  4. Decide: Is fine-tuning providing value?")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()