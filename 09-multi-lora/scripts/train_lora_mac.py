"""
LoRA Fine-tuning Training Script for Apple Silicon

Adapted from train_lora.py for Mac Studio with Metal Performance Shaders (MPS).

Key differences from cloud version:
- No BitsAndBytes quantization (uses PyTorch native INT8 or FP16)
- MPS device instead of CUDA
- Standard AdamW optimizer
- Adjusted memory settings for 64GB RAM

Usage:
    python train_lora_mac.py --lora_config standard
    python train_lora_mac.py --lora_config standard --test_mode
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
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from config_mac import (
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


def check_mps_availability():
    """Check if MPS (Metal Performance Shaders) is available."""
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Training will use CPU (very slow).")
        print("Make sure you're running on Apple Silicon Mac with macOS 12.3+")
        return False
    
    if not torch.backends.mps.is_built():
        print("WARNING: MPS not built in PyTorch. Reinstall PyTorch with MPS support.")
        return False
    
    print("✅ MPS (Metal Performance Shaders) available")
    return True


def load_model_and_tokenizer(config, lora_config_name):
    """
    Load base model and apply LoRA - Mac version.
    
    Differences from cloud version:
    - No BitsAndBytes quantization
    - Uses PyTorch native INT8 or FP16
    - MPS device mapping
    
    Args:
        config: TrainingConfig instance
        lora_config_name: Name of LoRA configuration
    
    Returns:
        tuple: (model, tokenizer, lora_config)
    """
    print("\n" + "=" * 80)
    print(f"Loading model: {config.model_name}")
    print("=" * 80)
    
    # Check MPS availability
    check_mps_availability()
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, "
          f"EOS={tokenizer.eos_token_id}, "
          f"BOS={tokenizer.bos_token_id}")
    
    # Load base model - Mac version (FP16, no quantization)
    print(f"\nLoading base model from HuggingFace Hub...")
    print("Using FP16 (no quantization needed with 64GB RAM)...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Will use MPS automatically
        trust_remote_code=True,
    )
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Prepare model for training
    # Note: prepare_model_for_kbit_training works for both quantized and non-quantized
    if config.use_gradient_checkpointing:
        print("\nPreparing model with gradient checkpointing...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )
    
    # Print base model parameters
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
    
    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    
    # Verify trainable parameters
    print("\nModel after LoRA:")
    print_model_parameters(model, detailed=False)
    
    # Enable gradient checkpointing if configured
    if config.use_gradient_checkpointing:
        print("\nGradient checkpointing enabled")
    
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
    
    # Create output directory
    output_dir = os.path.join(config.output_dir, f"{lora_cfg.name}_mac")
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
        
        # Optimization - Mac specific
        optim=config.optim,  # "adamw_torch"
        fp16=config.fp16,    # True for MPS
        bf16=config.bf16,    # False for MPS
        gradient_checkpointing=config.use_gradient_checkpointing,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # MLflow tracking
        report_to=["mlflow"],
        run_name=f"{lora_cfg.name}_mac",
        
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
    
    print("\n" + "!" * 80)
    print("TIP: Monitor memory usage in Activity Monitor")
    print("     Memory Pressure should stay GREEN")
    print("!" * 80)
    
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
    print("Starting training on Apple Silicon...")
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
    """Main training function."""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Apple Silicon")
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
        help="Test mode: train for a few steps only"
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="http://localhost:5001",
        help="MLflow tracking URI"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_TRAINING_CONFIG
    
    # Test mode
    if args.test_mode:
        print("\n" + "!" * 80)
        print("TEST MODE: Reduced training for quick verification")
        print("!" * 80 + "\n")
        config.num_epochs = 1
        config.eval_steps = 10
        config.save_steps = 10
        config.logging_steps = 1
        # Reduce dataset size in test mode
        config.max_samples = 100
    
    print("\n" + "=" * 80)
    print("LoRA Fine-tuning on Apple Silicon")
    print("=" * 80)
    print(f"LoRA Config: {args.lora_config}")
    print(f"MLflow URI: {args.mlflow_uri}")
    print(f"Test Mode: {args.test_mode}")
    print("=" * 80 + "\n")
    
    # Setup MLflow
    import mlflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("aws-rag-qa-mistral-lora-mac")
    
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
        
        # In test mode, limit dataset size
        if args.test_mode and hasattr(config, 'max_samples'):
            print(f"\nTest mode: Limiting to {config.max_samples} training samples")
            train_dataset = train_dataset.select(range(min(config.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(config.max_samples // 3, len(eval_dataset))))
        
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
        print(f"Model artifacts: {config.output_dir}/{lora_cfg.name}_mac")
        print(f"LoRA adapter: {config.output_dir}/{lora_cfg.name}_mac/adapter")
        print(f"\nView results in MLflow: {args.mlflow_uri}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Partial results may be saved in output directory")


if __name__ == "__main__":
    main()
