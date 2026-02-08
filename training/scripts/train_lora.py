"""
LoRA Fine-tuning Training Script

This script fine-tunes Mistral 7B using LoRA (Low-Rank Adaptation) on AWS certification data.

The workflow:
1. Load base model (Mistral 7B) with 4-bit quantization (QLoRA)
2. Apply LoRA adapters to specified layers
3. Train only the LoRA parameters (< 1% of total parameters)
4. Track experiments in MLflow
5. Save LoRA adapters (much smaller than full model)

Usage:
    python train_lora.py --lora_config standard
    python train_lora.py --lora_config minimal --test_mode
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from config import (
    DEFAULT_TRAINING_CONFIG,
    get_lora_config,
    estimate_trainable_parameters,
    LORA_CONFIGS
)
from utils import (
    create_dataset,
    DataCollatorForInstructionTuning,
    count_parameters,
    print_model_parameters
)


def load_model_and_tokenizer(config, lora_config_name):
    """
    Load base model with quantization and apply LoRA.
    
    This function demonstrates the key concepts:
    1. 4-bit quantization reduces memory from ~28GB to ~7GB
    2. LoRA adapters are added to frozen base model
    3. Only LoRA parameters are trainable
    
    Args:
        config: TrainingConfig instance
        lora_config_name: Name of LoRA configuration to use
    
    Returns:
        tuple: (model, tokenizer, lora_config)
    """
    print("\n" + "=" * 80)
    print(f"Loading model: {config.model_name}")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set padding token (Mistral doesn't have one by default)
    # IMPORTANT: Use unk_token, not eos_token (see Blog Post 6 for details)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    # Print tokenizer info
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, "
          f"EOS={tokenizer.eos_token_id}, "
          f"BOS={tokenizer.bos_token_id}")
    
    # Configure 4-bit quantization (QLoRA)
    # This is the key to fitting 7B model on 16GB GPU
    if config.use_4bit:
        print("\nConfiguring 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Nested quantization
        )
    else:
        bnb_config = None
        print("\nWarning: Loading without quantization - may not fit on T4!")
    
    # Load base model
    print(f"\nLoading base model from HuggingFace Hub...")
    print("This may take a few minutes on first run (downloading ~14GB)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.use_4bit else torch.float32,
    )
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model dtype: {model.dtype}")
    
    # Prepare model for k-bit training
    if config.use_4bit:
        print("\nPreparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )
    
    # Print base model parameters before LoRA
    print("\nBase model (before LoRA):")
    stats = count_parameters(model)
    print(f"Total parameters: {stats['total']:,}")
    
    # Get LoRA configuration
    lora_cfg = get_lora_config(lora_config_name)
    print("\n" + "=" * 80)
    print(f"Applying LoRA Configuration: {lora_cfg.name}")
    print("=" * 80)
    print(lora_cfg)
    
    # Estimate parameters
    param_estimate = estimate_trainable_parameters(lora_cfg, stats['total'])
    print(f"\nEstimated trainable parameters: {param_estimate['trainable_params']:,} "
          f"({param_estimate['percentage']:.4f}%)")
    
    # Create PEFT LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias="none",
        inference_mode=False,
    )
    
    # Apply LoRA to model
    print("\nApplying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    
    # Verify actual trainable parameters
    print("\nModel after LoRA:")
    print_model_parameters(model, detailed=False)
    
    # Enable gradient checkpointing if configured
    if config.use_gradient_checkpointing:
        print("\nEnabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled - trades compute for memory")
    
    return model, tokenizer, lora_cfg


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config,
    lora_cfg
):
    """
    Train the model using HuggingFace Trainer.
    
    Args:
        model: PEFT model with LoRA adapters
        tokenizer: HuggingFace tokenizer
        train_dataset: Training dataset
        eval_dataset: Validation dataset
        config: TrainingConfig instance
        lora_cfg: LoRA configuration
    
    Returns:
        Trainer instance
    """
    print("\n" + "=" * 80)
    print("Setting up training")
    print("=" * 80)
    
    # Create output directory for this specific LoRA config
    output_dir = os.path.join(config.output_dir, lora_cfg.name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Training arguments
    training_args = TrainingArguments(
        # Output & logging
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        
        # Training
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        
        # Optimization
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.use_gradient_checkpointing,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # MLflow tracking (automatic reporting)
        report_to=["mlflow"],
        run_name=lora_cfg.name,
    )
    
    # Data collator
    data_collator = DataCollatorForInstructionTuning(
        tokenizer=tokenizer,
        mask_instruction=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} examples")
    print(f"  Validation: {len(eval_dataset)} examples")
    
    # Calculate training steps
    steps_per_epoch = len(train_dataset) // (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.num_epochs
    print(f"\nTraining steps:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Evaluation every {config.eval_steps} steps")
    print(f"  Checkpoint every {config.save_steps} steps")
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")
        print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
        
        # Save final model
        print(f"\nSaving final model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapter separately
        adapter_dir = os.path.join(output_dir, "adapter")
        print(f"Saving LoRA adapter to {adapter_dir}...")
        model.save_pretrained(adapter_dir)
        
        print("\n" + "=" * 80)
        print("Training pipeline completed successfully!")
        print("=" * 80 + "\n")
        
        return trainer
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Training failed with error:")
        print(f"{'='*80}")
        print(f"{type(e).__name__}: {e}")
        print(f"{'='*80}\n")
        raise


def main():
    """
    Main training function.
    
    Usage:
        python train_lora.py --lora_config minimal
        python train_lora.py --lora_config standard
        python train_lora.py --lora_config aggressive
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Mistral 7B")
    parser.add_argument(
        "--lora_config",
        type=str,
        default="standard",
        choices=list(LORA_CONFIGS.keys()),
        help="LoRA configuration to use"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test mode: only train for a few steps"
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_TRAINING_CONFIG
    
    # Test mode: reduce epochs and steps for quick verification
    if args.test_mode:
        print("\n" + "!" * 80)
        print("TEST MODE: Reduced training for quick verification")
        print("!" * 80 + "\n")
        config.num_epochs = 1
        config.eval_steps = 10
        config.save_steps = 10
        config.logging_steps = 1
    
    print("\n" + "=" * 80)
    print("LoRA Fine-tuning Pipeline")
    print("=" * 80)
    print(f"LoRA Config: {args.lora_config}")
    print(f"MLflow URI: {args.mlflow_uri}")
    print(f"Test Mode: {args.test_mode}")
    print("=" * 80 + "\n")
    
    # Setup MLflow
    import mlflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("aws-rag-qa-mistral-lora")
    
    try:
        # Load model and tokenizer
        model, tokenizer, lora_cfg = load_model_and_tokenizer(config, args.lora_config)
        
        # Load datasets
        print("\n" + "=" * 80)
        print("Loading datasets")
        print("=" * 80)
        
        train_dataset = create_dataset(
            config.train_dataset_path,
            tokenizer,
            max_length=config.max_seq_length
        )
        
        eval_dataset = create_dataset(
            config.val_dataset_path,
            tokenizer,
            max_length=config.max_seq_length
        )
        
        # Train
        trainer = train(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            config,
            lora_cfg
        )
        
        print("\nSuccess! Model trained and saved.")
        print(f"Model artifacts saved to: {config.output_dir}/{lora_cfg.name}")
        print(f"LoRA adapter saved to: {config.output_dir}/{lora_cfg.name}/adapter")
        print(f"\nView results in MLflow: {args.mlflow_uri}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Partial results may be saved in output directory")


if __name__ == "__main__":
    main()
