"""
Generate Train/Val/Eval Datasets from Chunks and QA Pairs

This script creates properly formatted datasets for LLM fine-tuning and evaluation
by joining chunks with QA pairs, formatting context with heading hierarchy, and
performing stratified splits by question type.

Key features:
- Joins chunks with QA pairs on chunk_id
- Prepends heading hierarchy to context for better understanding
- Generates TWO prompt formats:
  * prompt_training: Full prompt with answer (for training/validation)
  * prompt_inference: Prompt without answer (for evaluation/inference)
- Performs stratified split by question_type (factual/conceptual/comparison)
- Ensures exactly 1/3 of each question type in all splits

Output format per record:
{
    "system": "System prompt...",
    "context": "Section: X > Y\n\nContent...",
    "question": "Question text",
    "reference_answer": "Answer text",
    "prompt_training": "[INST] context\n\nQuestion: X [/INST] answer",
    "prompt_inference": "[INST] context\n\nQuestion: X [/INST]",
    "metadata": {...}
}

Usage:
    python generate_datasets.py \
        --chunks chunks.jsonl \
        --qa_pairs qa_pairs_generated.jsonl \
        --output_dir data/processed \
        --train_ratio 0.60 \
        --val_ratio 0.20 \
        --eval_ratio 0.20
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import random

# For stratified splitting
from sklearn.model_selection import train_test_split


SYSTEM_PROMPT = """You are an expert assistant for AWS (Amazon Web Services) certification preparation.

CRITICAL: You must answer ONLY based on the provided context below. Follow these rules strictly:

Rules for answers:
- Extract and provide ALL relevant information from the context
- NEVER add information not explicitly stated in the context
- NEVER use external knowledge or your training data - only use what's in the given context
- Be as detailed as the context allows - short context = short answer, detailed context = detailed answer
- Write in complete, helpful sentences as if answering a colleague
- If comparing items, ONLY compare aspects explicitly mentioned in the context
- If the context doesn't provide enough information to answer the question, respond with: "The provided context does not contain sufficient information to answer this question."
- Answers should be in English
- Do not reference or mention "the context" in your answer - answer naturally as if you had this knowledge
"""


def format_mistral_training_prompt(context: str, question: str, answer: str) -> str:
    """
    Format a complete training prompt in Mistral's format.
    
    Includes both the input (context + question) and the expected output (answer).
    This is used for training and validation where the model learns from complete
    input-output pairs.
    
    Args:
        context: Context text with heading hierarchy
        question: The question to answer
        answer: The reference answer
    
    Returns:
        Full prompt string in format: [INST] ... [/INST] answer
    """
    # Mistral format: [INST] instruction [/INST] response
    prompt = f"[INST] {context}\n\nQuestion: {question} [/INST] {answer}"
    return prompt


def format_mistral_inference_prompt(context: str, question: str) -> str:
    """
    Format an inference-only prompt in Mistral's format.
    
    Includes only the input (context + question) without the answer.
    This is used during evaluation/inference where the model generates the answer.
    
    Args:
        context: Context text with heading hierarchy
        question: The question to answer
    
    Returns:
        Inference prompt string in format: [INST] ... [/INST]
    """
    # Same format but without the answer - model will generate it
    prompt = f"[INST] {context}\n\nQuestion: {question} [/INST]"
    return prompt


# For future extensibility: could add format_llama2_training_prompt(), etc.


# System prompt (constant for all samples)
SYSTEM_PROMPT = """You are an expert assistant for AWS (Amazon Web Services) certification preparation.

CRITICAL: You must answer ONLY based on the provided context below. Follow these rules strictly:

Rules for answers:
- Extract and provide ALL relevant information from the context
- NEVER add information not explicitly stated in the context
- NEVER use external knowledge or your training data - only use what's in the given context
- Be as detailed as the context allows - short context = short answer, detailed context = detailed answer
- Write in complete, helpful sentences as if answering a colleague
- If comparing items, ONLY compare aspects explicitly mentioned in the context
- If the context doesn't provide enough information to answer the question, respond with: "The provided context does not contain sufficient information to answer this question."
- Answers should be in English
- Do not reference or mention "the context" in your answer - answer naturally as if you had this knowledge
"""


def load_jsonl(filepath: str) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    return data


def format_heading_hierarchy(hierarchy: List[str]) -> str:
    """
    Format heading hierarchy as breadcrumb-style section indicator.
    
    Args:
        hierarchy: List of heading levels, e.g. ["AWS X-Ray FAQs", "Using AWS X-Ray"]
    
    Returns:
        Formatted string like "Section: AWS X-Ray FAQs > Using AWS X-Ray"
    """
    if not hierarchy:
        return ""
    
    # Join with " > " separator
    breadcrumb = " > ".join(hierarchy)
    return f"Section: {breadcrumb}"


def create_context_with_heading(chunk: Dict) -> str:
    """
    Create context string with heading hierarchy prepended.
    
    Args:
        chunk: Chunk dictionary with 'content' and 'metadata.heading_hierarchy'
    
    Returns:
        Formatted context string
    """
    content = chunk.get('content', '')
    metadata = chunk.get('metadata', {})
    hierarchy = metadata.get('heading_hierarchy', [])
    
    # Format heading
    heading_str = format_heading_hierarchy(hierarchy)
    
    # Combine: heading + blank line + content
    if heading_str:
        return f"{heading_str}\n\n{content}"
    else:
        return content


def join_chunks_and_qa_pairs(chunks: List[Dict], qa_pairs: List[Dict]) -> List[Dict]:
    """
    Join chunks with QA pairs on chunk_id and create formatted records.
    
    Creates records with structured data AND pre-formatted prompts for both
    training and inference. This makes debugging easier and ensures transparency
    about what actually goes into the model.
    
    Output record format:
    {
        'system': System prompt (constant),
        'context': Context with heading hierarchy,
        'question': The question,
        'reference_answer': The expected answer,
        'prompt_training': Full prompt with answer (for training/validation),
        'prompt_inference': Prompt without answer (for evaluation/inference),
        'metadata': {
            'service': AWS service name,
            'doc_type': Document type,
            'question_type': 'factual', 'conceptual', or 'comparison',
            'chunk_id': Unique chunk identifier
        }
    }
    
    Args:
        chunks: List of chunk dictionaries
        qa_pairs: List of QA pair dictionaries
    
    Returns:
        List of joined records in evaluation format
    """
    # Create chunk lookup by chunk_id
    chunk_lookup = {}
    for chunk in chunks:
        chunk_id = chunk.get('metadata', {}).get('chunk_id')
        if chunk_id:
            chunk_lookup[chunk_id] = chunk
        else:
            print(f"Warning: Chunk missing chunk_id: {chunk.get('metadata', {})}")
    
    print(f"Loaded {len(chunk_lookup)} chunks")
    
    # Join QA pairs with chunks
    joined_records = []
    missing_chunks = set()
    
    for qa_pair in qa_pairs:
        chunk_id = qa_pair.get('metadata', {}).get('chunk_id')
        
        if not chunk_id:
            print(f"Warning: QA pair missing chunk_id: {qa_pair.get('metadata', {})}")
            continue
        
        if chunk_id not in chunk_lookup:
            missing_chunks.add(chunk_id)
            continue
        
        chunk = chunk_lookup[chunk_id]
        
        # Extract fields
        context = create_context_with_heading(chunk)
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # Create formatted record
        record = {
            'system': SYSTEM_PROMPT,
            'context': context,
            'question': question,
            'reference_answer': answer,
            'prompt_training': format_mistral_training_prompt(context, question, answer),
            'prompt_inference': format_mistral_inference_prompt(context, question),
            'metadata': {
                'service': chunk.get('metadata', {}).get('service', 'UNKNOWN'),
                'doc_type': chunk.get('metadata', {}).get('doc_type', 'UNKNOWN'),
                'question_type': qa_pair.get('question_type', 'UNKNOWN'),
                'chunk_id': chunk_id
            }
        }
        
        joined_records.append(record)
    
    if missing_chunks:
        print(f"\nWarning: {len(missing_chunks)} QA pairs reference missing chunks:")
        for chunk_id in sorted(missing_chunks)[:10]:  # Show first 10
            print(f"  - {chunk_id}")
        if len(missing_chunks) > 10:
            print(f"  ... and {len(missing_chunks) - 10} more")
    
    print(f"\nSuccessfully joined {len(joined_records)} records")
    
    # Show example of generated prompts for verification
    if joined_records:
        print("\n" + "="*80)
        print("EXAMPLE GENERATED PROMPTS (first record)")
        print("="*80)
        example = joined_records[0]
        print("\nPrompt for Training (with answer):")
        print("-" * 80)
        print(example['prompt_training'][:500] + "..." if len(example['prompt_training']) > 500 else example['prompt_training'])
        print("\nPrompt for Inference (without answer):")
        print("-" * 80)
        print(example['prompt_inference'][:500] + "..." if len(example['prompt_inference']) > 500 else example['prompt_inference'])
        print("="*80)
    
    return joined_records


def create_stratify_key(record: Dict) -> str:
    """
    Create stratification key using question type only.
    
    We stratify by question_type (factual, conceptual, comparison) to ensure
    all splits have exactly 1/3 of each type. Service distribution will be
    approximately proportional due to random sampling, which is sufficient
    given the large dataset size.
    
    Note: We don't stratify by service because some services have very few
    chunks (only 1-2), which causes sklearn to fail with "too few members".
    
    Args:
        record: Data record with metadata
    
    Returns:
        Question type string: "factual", "conceptual", or "comparison"
    """
    return record['metadata']['question_type']


def stratified_split(
    records: List[Dict],
    train_ratio: float,
    val_ratio: float,
    eval_ratio: float,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform stratified split maintaining question type distribution.
    
    Stratifies by question_type to ensure all splits have exactly 1/3 of each
    type (factual, conceptual, comparison). Service distribution will be
    approximately proportional due to random sampling.
    
    Args:
        records: All data records
        train_ratio: Fraction for training (e.g., 0.20)
        val_ratio: Fraction for validation (e.g., 0.20)
        eval_ratio: Fraction for evaluation (e.g., 0.60)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_records, val_records, eval_records)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + eval_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Create stratification keys
    stratify_keys = [create_stratify_key(r) for r in records]
    
    # Check if all strata have enough samples
    strata_counts = Counter(stratify_keys)
    min_samples_per_stratum = min(strata_counts.values())
    
    print(f"\nStratification by question type:")
    for qtype in sorted(strata_counts.keys()):
        count = strata_counts[qtype]
        pct = 100 * count / len(records)
        print(f"  {qtype:15s}: {count:5d} ({pct:5.1f}%)")
    
    if min_samples_per_stratum < 3:
        raise ValueError(
            f"Not enough samples for stratification. "
            f"Minimum samples per question type: {min_samples_per_stratum}"
        )
    
    # First split: separate out eval set
    train_val_records, eval_records, train_val_keys, _ = train_test_split(
        records,
        stratify_keys,
        test_size=eval_ratio,
        random_state=random_seed,
        stratify=stratify_keys
    )
    
    # Second split: separate train and val from remaining
    # Adjust ratio: if we had 20/20/60, and we removed 60, we now split 20/20 from remaining 40
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=train_val_keys
    )
    
    return train_records, val_records, eval_records


def print_split_statistics(
    train: List[Dict],
    val: List[Dict],
    eval: List[Dict]
):
    """
    Print detailed statistics about the splits.
    """
    print("\n" + "="*80)
    print("SPLIT STATISTICS")
    print("="*80)
    
    def get_stats(records: List[Dict], name: str):
        print(f"\n{name} Set: {len(records)} samples")
        
        # Question type distribution
        question_types = [r['metadata']['question_type'] for r in records]
        type_counts = Counter(question_types)
        print(f"  Question Types:")
        for qtype, count in sorted(type_counts.items()):
            pct = 100 * count / len(records)
            print(f"    {qtype:15s}: {count:5d} ({pct:5.1f}%)")
        
        # Service distribution
        services = [r['metadata']['service'] for r in records]
        service_counts = Counter(services)
        print(f"  Services ({len(service_counts)} unique):")
        for service, count in service_counts.most_common(10):
            pct = 100 * count / len(records)
            print(f"    {service:15s}: {count:5d} ({pct:5.1f}%)")
        if len(service_counts) > 10:
            print(f"    ... and {len(service_counts) - 10} more")
    
    get_stats(train, "TRAIN")
    get_stats(val, "VALIDATION")
    get_stats(eval, "EVALUATION")
    
    print("\n" + "="*80)


def save_jsonl(records: List[Dict], filepath: str, prompt_type: str = 'training'):
    """
    Save records as JSONL file, including only the relevant prompt field.
    
    Args:
        records: List of records
        filepath: Output path
        prompt_type: 'training' (keeps prompt_training, for train/val files) or
                     'inference' (keeps prompt_inference, for eval file)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            # Make a copy to avoid mutating the original
            record_copy = record.copy()
            
            # Remove the prompt field that's not needed for this use case
            if prompt_type == 'training':
                # Training/validation files don't need inference prompts
                record_copy.pop('prompt_inference', None)
            elif prompt_type == 'inference':
                # Evaluation files don't need training prompts
                record_copy.pop('prompt_training', None)
            else:
                raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'training' or 'inference'")
            
            f.write(json.dumps(record_copy, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(records)} records to {filepath} (prompt_type={prompt_type})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/eval datasets from chunks and QA pairs"
    )
    parser.add_argument(
        '--chunks',
        type=str,
        required=True,
        help='Path to chunks.jsonl file'
    )
    parser.add_argument(
        '--qa_pairs',
        type=str,
        required=True,
        help='Path to qa_pairs_generated.jsonl file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for generated datasets'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.60,
        help='Fraction of data for training (default: 0.20)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.20,
        help='Fraction of data for validation (default: 0.20)'
    )
    parser.add_argument(
        '--eval_ratio',
        type=float,
        default=0.20,
        help='Fraction of data for evaluation (default: 0.60)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("DATASET GENERATION")
    print("="*80)
    print(f"Chunks file: {args.chunks}")
    print(f"QA pairs file: {args.qa_pairs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: Train={args.train_ratio:.0%}, Val={args.val_ratio:.0%}, Eval={args.eval_ratio:.0%}")
    print(f"Random seed: {args.seed}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    chunks = load_jsonl(args.chunks)
    qa_pairs = load_jsonl(args.qa_pairs)
    
    print(f"Loaded {len(chunks)} chunks")
    print(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Join chunks and QA pairs
    print("\nJoining chunks with QA pairs...")
    records = join_chunks_and_qa_pairs(chunks, qa_pairs)
    
    if len(records) == 0:
        print("\nError: No records were successfully joined!")
        print("Check that chunk_ids in qa_pairs match chunk_ids in chunks")
        return
    
    # Perform stratified split
    print(f"\nPerforming stratified split...")
    train, val, eval_data = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        eval_ratio=args.eval_ratio,
        random_seed=args.seed
    )
    
    # Print statistics
    print_split_statistics(train, val, eval_data)
    
    # Save datasets
    print("\nSaving datasets...")
    output_dir = Path(args.output_dir)
    
    save_jsonl(train, output_dir / 'train.jsonl', prompt_type='training')
    save_jsonl(val, output_dir / 'val.jsonl', prompt_type='training')
    save_jsonl(eval_data, output_dir / 'eval.jsonl', prompt_type='inference')
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files in {output_dir}:")
    print(f"  - train.jsonl ({len(train)} samples) - contains prompt_training")
    print(f"  - val.jsonl ({len(val)} samples) - contains prompt_training")
    print(f"  - eval.jsonl ({len(eval_data)} samples) - contains prompt_inference")
    print(f"\nTotal: {len(records)} samples")
    print("\nPrompt fields:")
    print("  - prompt_training: Full prompt with answer (for self-supervised learning)")
    print("  - prompt_inference: Prompt without answer (for evaluation/generation)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()