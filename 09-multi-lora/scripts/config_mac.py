"""
Configuration for LoRA Fine-tuning on Apple Silicon (Mac Studio)

This is an adapted version of config.py optimized for Apple Silicon.
Key differences from cloud GPU version:
- No BitsAndBytes (not supported on Mac)
- Uses MPS (Metal Performance Shaders) instead of CUDA
- Standard AdamW instead of paged_adamw_8bit
- FP16 instead of 4-bit quantization
- Smaller batch size to start conservatively
"""

from dataclasses import dataclass
from typing import List
import os


@dataclass
class LoRAConfig:
    """
    Configuration for a single LoRA experiment.
    
    Same as cloud version - LoRA configs are hardware-agnostic.
    """
    name: str
    rank: int
    alpha: int
    target_modules: List[str]
    dropout: float = 0.05
    
    @property
    def effective_scaling(self) -> float:
        return self.alpha / self.rank
    
    def __str__(self) -> str:
        return (f"LoRA Config: {self.name}\n"
                f"  Rank: {self.rank}, Alpha: {self.alpha} "
                f"(scaling: {self.effective_scaling:.2f})\n"
                f"  Target modules: {', '.join(self.target_modules)}\n"
                f"  Dropout: {self.dropout}")


# Same LoRA configs as cloud version
LORA_CONFIGS = {
    "minimal": LoRAConfig(
        name="minimal_r4_qv",
        rank=4,
        alpha=8,
        target_modules=["q_proj", "v_proj"],
        dropout=0.05
    ),
    
    "standard": LoRAConfig(
        name="standard_r8_qkvo",
        rank=8,
        alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dropout=0.05
    ),
    
    "aggressive": LoRAConfig(
        name="aggressive_r16_all",
        rank=16,
        alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        dropout=0.05
    ),
}


@dataclass
class TrainingConfig:
    """
    Training configuration optimized for Apple Silicon.
    
    Key differences from cloud config:
    - use_4bit = False (no BitsAndBytes support)
    - optim = "adamw_torch" (standard PyTorch AdamW)
    - smaller batch_size (start conservative)
    - fp16 = True (MPS supports FP16)
    """
    # Model
    model_name: str = "mistralai/Mistral-7B-v0.1"
    use_4bit: bool = False  # ❌ BitsAndBytes not supported on Mac
    use_8bit: bool = True   # ✅ PyTorch native INT8 quantization
    
    # Data
    train_dataset_path: str = "data/processed/train.jsonl"
    val_dataset_path: str = "data/processed/val.jsonl"
    max_seq_length: int = 1024
    
    # Training - Conservative batch size for first run
    num_epochs: int = 1
    per_device_train_batch_size: int = 2  # Start small, can increase
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # Effective batch = 2 * 4 = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    
    # Optimization - Mac-specific
    use_gradient_checkpointing: bool = True
    optim: str = "adamw_torch"  # ✅ Standard PyTorch AdamW
    fp16: bool = True           # ✅ MPS supports FP16
    bf16: bool = False          # ❌ MPS doesn't support BF16 well
    
    # Logging & Checkpointing
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "aws-cert-mistral-lora-mac"
    
    # Output
    output_dir: str = "training/models"
    
    def __post_init__(self):
        """Validate and convert paths to absolute."""
        # Compute effective batch size
        self.effective_batch_size = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )
        
        # Convert relative paths to absolute
        config_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(config_file))
        
        if not os.path.isabs(self.train_dataset_path):
            self.train_dataset_path = os.path.join(project_root, self.train_dataset_path)
        if not os.path.isabs(self.val_dataset_path):
            self.val_dataset_path = os.path.join(project_root, self.val_dataset_path)
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(project_root, self.output_dir)
        
        # Verify datasets exist
        if not os.path.exists(self.train_dataset_path):
            raise FileNotFoundError(
                f"Training dataset not found: {self.train_dataset_path}"
            )
        if not os.path.exists(self.val_dataset_path):
            raise FileNotFoundError(
                f"Validation dataset not found: {self.val_dataset_path}"
            )
        
        print("\n" + "="*80)
        print("APPLE SILICON CONFIG")
        print("="*80)
        print(f"✅ No 4-bit quantization (using {'INT8' if self.use_8bit else 'FP16'})")
        print(f"✅ Standard AdamW optimizer")
        print(f"✅ FP16 training on MPS")
        print(f"✅ Batch size: {self.per_device_train_batch_size} "
              f"(effective: {self.effective_batch_size})")
        print("="*80 + "\n")


# Default config for Mac
DEFAULT_TRAINING_CONFIG = TrainingConfig()


def get_lora_config(name: str) -> LoRAConfig:
    """Get LoRA config by name."""
    if name not in LORA_CONFIGS:
        available = ", ".join(LORA_CONFIGS.keys())
        raise ValueError(f"Unknown LoRA config '{name}'. Available: {available}")
    return LORA_CONFIGS[name]


def estimate_trainable_parameters(lora_config: LoRAConfig, base_model_params: int) -> dict:
    """Estimate trainable parameters (same as cloud version)."""
    n_layers = 32
    hidden_dim = 4096
    intermediate_dim = 14336
    
    params_per_layer = 0
    
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for module in attention_modules:
        if module in lora_config.target_modules:
            params_per_layer += 2 * lora_config.rank * hidden_dim
    
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
