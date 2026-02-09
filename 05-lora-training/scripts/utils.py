"""
Utility functions for LoRA fine-tuning.

This module handles dataset loading, preprocessing, and helper functions.
Similar to your CV project's data loading utilities, but adapted for text/instruction data.
"""

import json
import os
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import torch


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load a JSONL file where each line is a JSON object.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def format_chatml_prompt(example: Dict) -> str:
    """
    Format a single example into ChatML format for instruction tuning.
    
    NOTE: This function is primarily for inference/evaluation with the new dataset
    format. For training, use the pre-formatted 'prompt_training' field directly
    from the dataset - no need to call this function.
    
    This can be useful for:
    - Formatting prompts during inference when you have structured data
    - Creating prompts for evaluation
    - Backward compatibility with old dataset format
    
    ChatML format uses special tokens to delineate roles:
    <|im_start|>system
    You are a helpful assistant...
    <|im_end|>
    <|im_start|>user
    What is AWS EC2?<|im_end|>
    <|im_start|>assistant
    AWS EC2 is...<|im_end|>
    
    For Mistral, we adapt this slightly since it wasn't trained with these exact tokens.
    Instead we use a format Mistral understands: [INST] instruction [/INST] response
    
    Args:
        example: Dict with either:
                 - New format: 'context', 'question', 'reference_answer' keys
                 - Old format: 'messages' or 'instruction'/'input'/'output' keys
    
    Returns:
        Formatted string ready for tokenization
    """
    # Handle both Alpaca format (instruction/input/output) and ChatML format
    if 'messages' in example:
        # ChatML format: list of message dicts with 'role' and 'content'
        messages = example['messages']
        
        # For Mistral's format, we combine system + user into the instruction
        instruction_parts = []
        response = ""
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                instruction_parts.append(f"System: {content}")
            elif role == 'user':
                instruction_parts.append(content)
            elif role == 'assistant':
                response = content
        
        instruction = "\n".join(instruction_parts)
    
    elif 'context' in example and 'question' in example:
        # New format: structured with context and question fields
        context = example['context']
        question = example['question']
        instruction = f"{context}\n\nQuestion: {question}"
        response = example.get('reference_answer', '')
        
    else:
        # Alpaca format: instruction, optional input, output
        instruction = example['instruction']
        if example.get('input', '').strip():
            instruction = f"{instruction}\n\nInput: {example['input']}"
        response = example['output']
    
    # Mistral's format: [INST] instruction [/INST] response
    # We include both the instruction and response so the model learns to predict the response
    prompt = f"[INST] {instruction} [/INST] {response}"
    
    return prompt


def create_dataset(
    file_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Create a HuggingFace Dataset from a JSONL file.
    
    This function:
    1. Loads the JSONL file
    2. Formats each example into the model's expected format
    3. Tokenizes the text
    4. Returns a Dataset object ready for training
    
    Args:
        file_path: Path to JSONL file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (longer sequences are truncated)
    
    Returns:
        HuggingFace Dataset with tokenized examples
    """
    print(f"Loading dataset from {file_path}...")
    
    # Load raw data
    raw_data = load_jsonl(file_path)
    print(f"Loaded {len(raw_data)} examples")
    
    # Extract pre-formatted prompts
    # For training/val: use 'prompt_training' which includes the answer
    formatted_texts = [record['prompt_training'] for record in raw_data]
    
    # Tokenize all at once (more efficient than one by one)
    # WICHTIG: Wir nutzen max_length - 1, um Platz fÃ¼r das EOS Token zu reservieren
    # Das Base Model muss lernen, dass nach der Antwort ein EOS Token kommt
    print(f"Tokenizing {len(formatted_texts)} examples...")
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length - 1,  # Reserve space for EOS token
        padding=False,  # Dynamic padding in collator
        return_tensors=None  # Return lists, not tensors
    )

    # Append EOS token to every sequence
    # This is CRITICAL: The model must learn to generate EOS after the answer
    # Without this, the model never learns when to stop during inference
    print(f"Adding EOS token (ID: {tokenizer.eos_token_id}) to all sequences...")
    for i in range(len(tokenized['input_ids'])):
        tokenized['input_ids'][i].append(tokenizer.eos_token_id)

    # Create labels (for causal language modeling, labels = input_ids)
    # The EOS token will NOT be masked - it's part of what the model needs to learn
    tokenized['labels'] = tokenized['input_ids'].copy()

    # Verify that EOS was added correctly
    print(f"Verification: First sequence ends with token ID {tokenized['input_ids'][0][-1]} "
        f"(should be {tokenizer.eos_token_id})")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(tokenized)
    
    print(f"Dataset created with {len(dataset)} examples")
    print(f"Average sequence length: {sum(len(x) for x in tokenized['input_ids']) / len(tokenized['input_ids']):.1f} tokens")
    
    return dataset


def create_instruction_mask(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer
) -> torch.Tensor:
    """
    Create a mask that marks which tokens are part of the instruction vs. the response.
    
    During training, we only compute loss on the response tokens, not the instruction.
    This is because we want the model to learn to generate responses, not to predict
    what instructions look like.
    
    The trick is to find where [/INST] appears in the sequence, and mask everything before it.
    
    Args:
        input_ids: Tensor of token IDs, shape (batch_size, seq_len)
        tokenizer: HuggingFace tokenizer to decode tokens
    
    Returns:
        Mask tensor where 1 = compute loss, 0 = ignore, shape (batch_size, seq_len)
    """
    # Find the [/INST] token
    # In Mistral's tokenizer, this is typically encoded as a specific token ID
    # We'll decode the sequence and look for the marker
    
    batch_size, seq_len = input_ids.shape
    mask = torch.ones_like(input_ids)
    
    for i in range(batch_size):
        # Decode to find where instruction ends
        ids = input_ids[i].tolist()
        text = tokenizer.decode(ids)
        
        # Find the [/INST] marker
        inst_end = text.find("[/INST]")
        
        if inst_end != -1:
            # Tokenize up to [/INST] to find the token position
            prefix = text[:inst_end + len("[/INST]")]
            prefix_tokens = tokenizer(prefix, add_special_tokens=False)['input_ids']
            mask_until = len(prefix_tokens)
            
            # Mask everything up to and including [/INST]
            mask[i, :mask_until] = 0
    
    return mask


class DataCollatorForInstructionTuning:
    """
    Custom data collator for instruction tuning.
    
    This handles:
    1. Dynamic padding (pad to longest sequence in batch)
    2. Label masking (ignore loss on instruction tokens)
    3. Attention mask creation
    
    Similar to how you handled image batching in CV, but for variable-length sequences.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, mask_instruction: bool = True):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            mask_instruction: If True, ignore loss on instruction tokens
        """
        self.tokenizer = tokenizer
        self.mask_instruction = mask_instruction
        self.pad_token_id = tokenizer.pad_token_id
        
        # If tokenizer doesn't have a pad token, use eos token
        if self.pad_token_id is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Updated to handle context-based instruction format correctly.
        """
        # Find max length in this batch
        max_length = max(len(f['input_ids']) for f in features)
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for feature in features:
            input_ids = feature['input_ids']
            labels = feature['labels']
            
            # Calculate padding length
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids and labels
            padded_input_ids = input_ids + [self.pad_token_id] * padding_length
            padded_labels = labels + [-100] * padding_length
            
            # Create attention mask
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            batch['input_ids'].append(padded_input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(padded_labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        
        # Apply instruction masking if enabled
        if self.mask_instruction:

            samples_without_marker = 0

            for i in range(len(batch['input_ids'])):
                text = self.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False)
                inst_end = text.find("[/INST]")
                
                if inst_end != -1:
                    prefix = text[:inst_end + len("[/INST]")]
                    prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)['input_ids']
                    mask_until = min(len(prefix_tokens), len(batch['labels'][i]))
                    
                    # Mask instruction tokens
                    batch['labels'][i, :mask_until] = -100
                else:
                    # If no [/INST] found, mask entire sequence (safety)
                    samples_without_marker += 1
                    batch['labels'][i, :] = -100
        
        if samples_without_marker > 0:
            print(f"âš ï¸  WARNING: {samples_without_marker}/{len(batch['input_ids'])} samples missing [/INST] - likely truncated!")
            print(f"   Consider increasing max_length (current: 1024)")
                
        return batch



def count_parameters(model) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    This is useful for verifying that LoRA is actually freezing most parameters.
    With full fine-tuning, all parameters are trainable.
    With LoRA, only a tiny fraction (< 1%) should be trainable.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dict with 'total', 'trainable', and 'percentage' keys
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'percentage': percentage
    }


def print_model_parameters(model, detailed: bool = False):
    """
    Print parameter statistics for a model.
    
    Args:
        model: PyTorch model
        detailed: If True, print per-layer statistics
    """
    stats = count_parameters(model)
    
    print("\n" + "=" * 80)
    print("Model Parameter Statistics")
    print("=" * 80)
    print(f"Total parameters: {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Trainable percentage: {stats['percentage']:.4f}%")
    print("=" * 80 + "\n")
    
    if detailed:
        print("Per-layer trainable status:")
        print("-" * 80)
        for name, param in model.named_parameters():
            status = "TRAINABLE" if param.requires_grad else "frozen"
            print(f"{status:12} | {name:60} | {param.numel():,} params")
        print("-" * 80 + "\n")


def setup_mlflow(config):
    """
    Setup MLflow tracking.
    
    This connects to your existing MLflow server (in ai-platform namespace)
    and creates/gets the experiment.
    
    Similar to your CV project's MLflow setup.
    
    Args:
        config: TrainingConfig instance
    """
    import mlflow
    
    # Set tracking URI
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(config.mlflow_experiment_name)
            print(f"Created new MLflow experiment: {config.mlflow_experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing MLflow experiment: {config.mlflow_experiment_name} (ID: {experiment_id})")
    except Exception as e:
        print(f"Warning: Could not setup MLflow: {e}")
        print("Continuing without MLflow tracking...")
        experiment_id = None
    
    return experiment_id


if __name__ == "__main__":
    """
    Test the utility functions.
    """
    from config import DEFAULT_TRAINING_CONFIG
    
    print("Testing dataset loading utilities...\n")
    
    # Test loading
    train_path = DEFAULT_TRAINING_CONFIG.train_dataset_path
    if os.path.exists(train_path):
        print(f"Loading {train_path}...")
        data = load_jsonl(train_path)
        print(f"Loaded {len(data)} examples")
        
        # Show first example
        print("\nFirst example:")
        print("-" * 80)
        print(json.dumps(data[0], indent=2))
        
        print("\nFormatted prompt:")
        print("-" * 80)
        formatted = format_chatml_prompt(data[0])
        print(formatted)
    else:
        print(f"Dataset not found at {train_path}")
        print("Make sure you're running this from the project root directory")