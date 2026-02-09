"""
Custom MLflow Callback for HuggingFace Trainer

This callback provides manual, explicit control over MLflow metric logging.
Instead of relying on the Trainer's built-in MLflow integration (which can
conflict with manual run management), we handle all MLflow logging ourselves.

This approach gives us:
1. Full transparency - we see exactly what gets logged and when
2. Fine-grained control - we can transform, filter, or enrich metrics
3. Robustness - no conflicts between different MLflow integration layers
4. Flexibility - easy to extend with custom metrics or logic

This is analogous to your CV project where you had explicit control over
your training loop and logging, rather than relying on framework magic.
"""

from transformers import TrainerCallback
import mlflow
from typing import Dict, Any


class MLflowCallback(TrainerCallback):
    """
    Custom callback that logs training metrics to MLflow.
    
    This callback intercepts the Trainer's logging events and forwards
    the metrics to MLflow manually. It assumes that an MLflow run is
    already active (started with mlflow.start_run()).
    
    The callback handles both training metrics (loss, learning rate, etc.)
    and evaluation metrics (eval_loss, eval_accuracy, etc.).
    """
    
    def __init__(self, log_model_checkpoints: bool = False):
        """
        Initialize the MLflow callback.
        
        Args:
            log_model_checkpoints: If True, log model checkpoints as MLflow artifacts.
                                  This can consume significant storage, so default is False.
                                  We'll log only the final model instead.
        """
        super().__init__()
        self.log_model_checkpoints = log_model_checkpoints
        self._is_world_process_zero = True  # For single-GPU training, always True
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the Trainer logs metrics.
        
        This is the main hook where we intercept metrics and send them to MLflow.
        The Trainer calls this method every `logging_steps` steps with a dictionary
        of metrics that were just computed.
        
        Args:
            args: TrainingArguments
            state: TrainerState with current training state (step, epoch, etc.)
            control: TrainerControl (unused here, but could be used to modify training)
            logs: Dictionary of metrics to log (e.g., {'loss': 1.5, 'learning_rate': 0.0001})
            **kwargs: Additional arguments (unused)
        """
        # Only log from the main process in distributed training
        # For single GPU training, this is always True
        if not self._is_world_process_zero:
            return
        
        # If there are no logs or no active MLflow run, skip
        if logs is None or not mlflow.active_run():
            return
        
        # Get the current global step
        # This is important for the x-axis in MLflow's metric plots
        step = state.global_step
        
        # Log each metric individually
        # We separate training and eval metrics for clarity
        for key, value in logs.items():
            # Skip non-numeric values (e.g., strings or None)
            if not isinstance(value, (int, float)):
                continue
            
            # The key might already include prefixes like 'train/' or 'eval/'
            # If not, we add them for clarity in MLflow UI
            if key.startswith('eval_'):
                # Evaluation metrics
                metric_name = f"eval/{key.replace('eval_', '')}"
            elif key in ['loss', 'grad_norm', 'learning_rate']:
                # Training metrics
                metric_name = f"train/{key}"
            elif key == 'epoch':
                # Epoch is a special case - log it as is
                metric_name = "epoch"
            else:
                # Any other metric, keep as is
                metric_name = key
            
            # Log to MLflow with the current step
            try:
                mlflow.log_metric(metric_name, value, step=step)
            except Exception as e:
                # Don't crash training if MLflow logging fails
                # Just print a warning and continue
                print(f"Warning: Failed to log metric {metric_name}={value}: {e}")
    
    def on_save(self, args, state, control, **kwargs):
        """
        Called when the Trainer saves a checkpoint.
        
        We could log the checkpoint as an artifact here if desired.
        For now, we skip this to save storage - we'll log only the final model.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            **kwargs: Additional arguments
        """
        if not self._is_world_process_zero or not self.log_model_checkpoints:
            return
        
        # Checkpoints are saved to args.output_dir / f"checkpoint-{state.global_step}"
        # We could log them here with mlflow.log_artifacts() if desired
        # But for storage efficiency, we skip intermediate checkpoints
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of training.
        
        We could log additional information here, like the model architecture
        or training configuration. For now, we keep it simple.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            **kwargs: Additional arguments
        """
        if not self._is_world_process_zero:
            return
        
        # Verify that an MLflow run is active
        if mlflow.active_run():
            print("MLflowCallback: Active MLflow run detected, will log metrics")
        else:
            print("WARNING: No active MLflow run! Metrics will not be logged.")
            print("Make sure to call mlflow.start_run() before training starts.")
    
    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training.
        
        This is a good place to log final statistics or summaries.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            **kwargs: Additional arguments
        """
        if not self._is_world_process_zero:
            return
        
        if mlflow.active_run():
            print("MLflowCallback: Training completed, all metrics logged to MLflow")


def create_mlflow_callback(**kwargs) -> MLflowCallback:
    """
    Factory function to create an MLflow callback.
    
    This is a convenience function that makes it easy to instantiate
    the callback with specific configuration.
    
    Args:
        **kwargs: Arguments to pass to MLflowCallback constructor
    
    Returns:
        Configured MLflowCallback instance
    
    Example:
        callback = create_mlflow_callback(log_model_checkpoints=False)
        trainer = Trainer(..., callbacks=[callback])
    """
    return MLflowCallback(**kwargs)


if __name__ == "__main__":
    """
    Simple test to verify the callback can be instantiated.
    """
    print("Testing MLflowCallback instantiation...")
    callback = create_mlflow_callback()
    print(f"âœ“ Callback created: {callback.__class__.__name__}")
    print(f"  Log model checkpoints: {callback.log_model_checkpoints}")
    print("\nCallback is ready to use with HuggingFace Trainer")