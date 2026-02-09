"""
Configuration for LoRA Fine-tuning Experiments

This module defines different LoRA configurations that we'll test empirically.
Similar to your CV project where you tested different learning rates and architectures,
here we test different LoRA ranks, alphas, and target modules to find optimal settings.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoRAConfig:
    """
    Configuration for a single LoRA experiment.
    
    Key parameters to understand:
    - rank (r): Controls how many parameters we add. Higher rank = more capacity but more parameters.
      Typical values: 4, 8, 16, 32. Start low and increase if needed.
    
    - alpha: Scaling factor for the LoRA updates. Controls how strongly the adaptations 
      influence the final output. Common practice: alpha = 2 * rank or alpha = rank.
    
    - target_modules: Which weight matrices to adapt. For transformers, the attention 
      projections (q_proj, v_proj) are most important. You can also add k_proj, o_proj,
      and the MLP layers (gate_proj, up_proj, down_proj) for more capacity.
    """
    name: str
    rank: int
    alpha: int
    target_modules: List[str]
    dropout: float = 0.05
    
    @property
    def effective_scaling(self) -> float:
        """
        The effective scaling factor is alpha/rank. This determines how much the
        LoRA adaptation influences the pre-trained weights.
        
        For example, with rank=8 and alpha=16, the scaling is 2.0, meaning
        LoRA updates are twice as influential as they would be without scaling.
        """
        return self.alpha / self.rank
    
    def __str__(self) -> str:
        return (f"LoRA Config: {self.name}\n"
                f"  Rank: {self.rank}, Alpha: {self.alpha} "
                f"(scaling: {self.effective_scaling:.2f})\n"
                f"  Target modules: {', '.join(self.target_modules)}\n"
                f"  Dropout: {self.dropout}")


# Experiment configurations from minimal to more aggressive
# We'll test these empirically to see the trade-offs between parameter count and performance

LORA_CONFIGS = {
    # Minimal LoRA: Only adapt attention queries and values with very low rank
    # This is the most parameter-efficient approach, good starting point
    "minimal": LoRAConfig(
        name="minimal_r4_qv",
        rank=4,
        alpha=8,
        target_modules=["q_proj", "v_proj"],
        dropout=0.05
    ),
    
    # Standard LoRA: Rank 8 with Q, K, V, O projections
    # This is a commonly used configuration that works well for many tasks
    "standard": LoRAConfig(
        name="standard_r8_qkvo",
        rank=8,
        alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dropout=0.05
    ),
    
    # Aggressive LoRA: Higher rank, all attention + MLP layers
    # More parameters, potentially better performance but closer to full fine-tuning
    "aggressive": LoRAConfig(
        name="aggressive_r16_all",
        rank=16,
        alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        dropout=0.05
    ),
    
    # High capacity: Maximum reasonable LoRA for comparison
    # This will help us understand if more parameters actually help
    "high_capacity": LoRAConfig(
        name="high_capacity_r32_all",
        rank=32,
        alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        dropout=0.1  # Higher dropout for regularization with more parameters
    )
}


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.
    
    These are separate from LoRA config because they apply regardless of
    which LoRA configuration we're testing.
    
    Note: Paths are defined as relative to project root, but are automatically
    converted to absolute paths in __post_init__. This makes the config work
    regardless of where the script is called from.
    """
    # Model
    model_name: str = "mistralai/Mistral-7B-v0.1"
    use_4bit: bool = True  # QLoRA: 4-bit quantization to fit on T4
    
    # Data (relative to project root)
    train_dataset_path: str = "data/processed/train.jsonl"
    val_dataset_path: str = "data/processed/val.jsonl"
    max_seq_length: int = 1024  # Balance between context and memory
    
    # Training
    num_epochs: int = 1
    per_device_train_batch_size: int = 4  # Adjust based on GPU memory
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 4 * 4 = 16
    learning_rate: float = 2e-4  # Typical for LoRA, higher than full fine-tuning
    warmup_steps: int = 20
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    
    # Optimization
    use_gradient_checkpointing: bool = True  # Trade compute for memory
    optim: str = "paged_adamw_8bit"  # Memory-efficient optimizer for QLoRA
    fp16: bool = False  # T4 doesn't support bf16, we'll use fp16 if needed
    bf16: bool = False
    
    # Logging & Checkpointing
    logging_steps: int = 1
    eval_steps: int = 20
    save_steps: int = 20
    save_total_limit: int = 3  # Keep only last 3 checkpoints
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "aws-rag-qa-mistral-lora"
    
    # Output (relative to project root)
    output_dir: str = "05.1-mlflow-tracking/models"
    
    def __post_init__(self):
        """
        Validate configuration and compute derived values.
        
        This is similar to your CV project where you computed effective learning rates
        based on batch size and gradient accumulation.
        
        Additionally, this method converts relative paths to absolute paths based on
        the project root directory. This ensures the config works regardless of where
        the training script is called from.
        """
        import os
        
        # Compute effective batch size
        self.effective_batch_size = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )
        
        # Convert relative paths to absolute paths
        # The config file is at: project_root/training/scripts/config.py
        # We need to go up 2 levels to reach project_root
        config_file = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(config_file)          # training/scripts/
        training_dir = os.path.dirname(scripts_dir)         # training/
        project_root = os.path.dirname(training_dir)        # project_root/
        
        # Make dataset paths absolute if they're not already
        if not os.path.isabs(self.train_dataset_path):
            self.train_dataset_path = os.path.join(project_root, self.train_dataset_path)
        if not os.path.isabs(self.val_dataset_path):
            self.val_dataset_path = os.path.join(project_root, self.val_dataset_path)
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(project_root, self.output_dir)
        
        # Verify that dataset files exist
        if not os.path.exists(self.train_dataset_path):
            raise FileNotFoundError(
                f"Training dataset not found at: {self.train_dataset_path}\n"
                f"Project root detected as: {project_root}\n"
                f"Please verify the dataset path in config.py"
            )
        if not os.path.exists(self.val_dataset_path):
            raise FileNotFoundError(
                f"Validation dataset not found at: {self.val_dataset_path}\n"
                f"Project root detected as: {project_root}\n"
                f"Please verify the dataset path in config.py"
            )
        
        # For T4 (16GB VRAM), we need to be careful with memory
        # With 4-bit quantization, gradient checkpointing, and small batch size,
        # we should fit Mistral 7B comfortably
        if not self.use_4bit and self.per_device_train_batch_size > 2:
            print("WARNING: Without 4-bit quantization, you may run out of memory on T4!")
            print("Consider reducing batch_size or enabling use_4bit=True")


# Default training configuration
DEFAULT_TRAINING_CONFIG = TrainingConfig()


def get_lora_config(name: str) -> LoRAConfig:
    """
    Get a LoRA configuration by name.
    
    Args:
        name: One of 'minimal', 'standard', 'aggressive', 'high_capacity'
    
    Returns:
        LoRAConfig instance
    
    Raises:
        ValueError if name is not recognized
    """
    if name not in LORA_CONFIGS:
        available = ", ".join(LORA_CONFIGS.keys())
        raise ValueError(f"Unknown LoRA config '{name}'. Available: {available}")
    
    return LORA_CONFIGS[name]


def estimate_trainable_parameters(lora_config: LoRAConfig, base_model_params: int) -> dict:
    """
    Estimate how many parameters will be trainable with this LoRA configuration.
    
    This is an approximation based on typical Mistral 7B architecture:
    - 32 layers
    - 4096 hidden dimension
    - 14336 intermediate dimension (MLP)
    
    Each targeted linear layer adds: 2 * rank * layer_dim parameters
    (2 because we have both the A and B matrices in the low-rank decomposition)
    
    Args:
        lora_config: LoRA configuration
        base_model_params: Number of parameters in base model (7B for Mistral)
    
    Returns:
        Dictionary with parameter statistics
    """
    n_layers = 32
    hidden_dim = 4096
    intermediate_dim = 14336
    
    params_per_layer = 0
    
    # Attention projections: q_proj, k_proj, v_proj, o_proj
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for module in attention_modules:
        if module in lora_config.target_modules:
            # Each is hidden_dim x hidden_dim
            params_per_layer += 2 * lora_config.rank * hidden_dim
    
    # MLP projections: gate_proj, up_proj (hidden -> intermediate), down_proj (intermediate -> hidden)
    if "gate_proj" in lora_config.target_modules:
        params_per_layer += 2 * lora_config.rank * intermediate_dim
    if "up_proj" in lora_config.target_modules:
        params_per_layer += 2 * lora_config.rank * intermediate_dim
    if "down_proj" in lora_config.target_modules:
        params_per_layer += 2 * lora_config.rank * intermediate_dim
    
    total_lora_params = params_per_layer * n_layers
    percentage = (total_lora_params / base_model_params) * 100
    
    return {
        "base_model_params": base_model_params,
        "trainable_params": total_lora_params,
        "percentage": percentage,
        "params_per_layer": params_per_layer,
        "target_modules": len(lora_config.target_modules)
    }


if __name__ == "__main__":
    """
    Test the configuration module by printing all configs and their parameter estimates.
    
    This is useful for understanding the trade-offs before running experiments.
    """
    print("LoRA Fine-tuning Configuration Module\n")
    print("=" * 80)
    
    print("\nTraining Configuration:")
    print("-" * 80)
    config = DEFAULT_TRAINING_CONFIG
    print(f"Model: {config.model_name}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Effective batch size: {config.effective_batch_size} "
          f"(per_device={config.per_device_train_batch_size}, "
          f"grad_accum={config.gradient_accumulation_steps})")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"4-bit quantization: {config.use_4bit}")
    print(f"Gradient checkpointing: {config.use_gradient_checkpointing}")
    
    print("\n\nLoRA Configurations:")
    print("=" * 80)
    
    # Mistral 7B has approximately 7.24B parameters
    base_params = 7_240_000_000
    
    for name, lora_cfg in LORA_CONFIGS.items():
        print(f"\n{lora_cfg}")
        
        stats = estimate_trainable_parameters(lora_cfg, base_params)
        print(f"  Estimated trainable parameters: {stats['trainable_params']:,} "
              f"({stats['percentage']:.3f}% of base model)")
        print(f"  Parameters per layer: {stats['params_per_layer']:,}")