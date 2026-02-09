"""
Comprehensive intrinsic evaluation for AWS Cert Mistral LoRA model.

This version is adapted to work with:
- LoRA adapters from Block 2 training
- ChatML formatted validation data
- MLflow tracking on EC2 instance
"""

import torch
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader
import mlflow
import sys
import os

# Add project paths for imports
# We're assuming evaluate_intrinsic.py is in phase2_finetuning/block3_evaluation/scripts/
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "phase2_finetuning" / "block2_lora_finetuning" / "scripts"))
sys.path.insert(0, str(project_root / "phase2_finetuning" / "block3_evaluation" / "metrics"))

# Import our custom metrics
from token_metrics import TokenMetrics
from loss_analysis import LossAnalyzer

# Import dataset utilities from Block 2
from utils import create_dataset, DataCollatorForInstructionTuning


def load_model_and_tokenizer(model_path: str, base_model_name: str, 
                              use_4bit: bool = True):
    """
    Load the fine-tuned model with LoRA adapter.
    
    Die wichtige Änderung: Wir nutzen jetzt 4-bit Quantization beim Laden,
    genau wie beim Training. Das hat mehrere Vorteile:
    
    1. Das Modell passt auf die GPU (7B Model wird von ~28GB auf ~7GB reduziert)
    2. Evaluation ist schneller (alles auf GPU, kein CPU offloading)
    3. Konsistent mit Training (gleiches Setup)
    
    Der einzige Trade-off: Minimal niedrigere Präzision durch Quantization.
    Aber der Effekt ist in der Praxis vernachlässigbar für Evaluation-Metriken.
    
    Args:
        model_path: Path to the directory containing the LoRA adapter
        base_model_name: Name or path of the base model
        use_4bit: Whether to use 4-bit quantization (QLoRA style)
    
    Returns:
        model, tokenizer
    """
    print(f"Loading base model: {base_model_name}")
    print(f"Using 4-bit quantization: {use_4bit}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Mistral braucht einen Pad Token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    # Wenn wir 4-bit nutzen, konfigurieren wir BitsAndBytes genau wie beim Training
    if use_4bit:
        from transformers import BitsAndBytesConfig
        
        print("Configuring 4-bit quantization (same as training)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        # Falls du ohne Quantization laden willst (nicht empfohlen für große Models)
        # dann brauchen wir einen offload_dir
        print("Loading without quantization - may require CPU offloading...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            offload_folder="offload",  # Temporärer Ordner für CPU offloading
            offload_state_dict=True,  # Offloade auch state dict bei Bedarf
        )
    
    # Load LoRA adapter if path provided
    model_path_obj = Path(model_path)
    if model_path_obj.exists() and (model_path_obj / 'adapter_config.json').exists():
        print(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, str(model_path))
        
        # WICHTIG: Bei quantized models sollten wir NICHT merge_and_unload() aufrufen!
        # Das würde die Quantization rückgängig machen und das Modell wieder groß machen.
        # Stattdessen lassen wir den Adapter als separate Layer.
        if use_4bit:
            print("Keeping LoRA adapter separate (required for quantized models)")
            # Der Adapter bleibt als separate Layer - das ist okay und funktioniert
        else:
            print("Merging LoRA adapter with base model for faster inference...")
            model = model.merge_and_unload()
    else:
        print(f"WARNING: No LoRA adapter found at {model_path}")
        print("Using base model without fine-tuning")
        model = base_model
    
    # Print model info
    print(f"\nModel loaded successfully:")
    print(f"  Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    print(f"  Dtype: {model.dtype}")
    
    # Check GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"  GPU {i} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return model, tokenizer


def compute_token_frequencies(dataloader, tokenizer):
    """
    Compute token frequency statistics from the dataset.
    
    Das ist wichtig für frequency-based loss breakdown. Wir zählen, wie oft
    jedes Token im Dataset vorkommt. Dann können wir später sehen, ob das
    Modell bei seltenen Tokens schlechter ist als bei häufigen.
    
    In der Praxis sehen wir oft: Rare tokens sind schwieriger zu predicten,
    weil das Modell sie weniger oft gesehen hat. Aber wie stark ist der Effekt?
    
    Returns:
        Dictionary mapping token_id -> count
    """
    print("Computing token frequency statistics...")
    token_counts = {}
    
    for batch in tqdm(dataloader, desc="Counting tokens"):
        labels = batch['labels']
        
        # Count each token
        for token_id in labels.flatten():
            token_id = token_id.item()
            if token_id != -100:  # Ignore padding/masked tokens
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
    
    return token_counts


def extract_category_from_sample(batch, idx):
    """
    Extract AWS service category from a sample.
    
    Das ist eine optionale Funktion. Wenn du deine Daten mit Categories taggen
    kannst, dann siehst du, ob dein Modell bei EC2-Fragen besser ist als bei
    S3-Fragen, etc.
    
    WICHTIG: Diese Funktion muss an deine Datenstruktur angepasst werden.
    Aktuell ist sie ein Platzhalter, der "unknown" zurückgibt.
    
    Args:
        batch: Der Batch vom DataLoader
        idx: Index des Samples im Batch
    
    Returns:
        Category string (z.B. "EC2", "S3", "IAM", "VPC")
    """
    # TODO: Wenn du Categories in deinen QA-Pairs hast, extrahiere sie hier
    # Beispiel: wenn deine JSONL eine "category" oder "service" Feld hat,
    # müsstest du das hier auslesen
    
    # Platzhalter für jetzt - später kannst du das verfeinern
    return "unknown"


def evaluate_model(model, tokenizer, dataloader, device='cuda',
                  compute_frequency_breakdown=True,
                  track_categories=False):
    """
    Run comprehensive intrinsic evaluation.
    
    Das ist die Hauptfunktion. Sie iteriert über den Validation Set,
    sammelt Predictions, berechnet Metriken und Loss-Breakdowns.
    
    Der Ablauf:
    1. Für jeden Batch: Forward pass durch Modell
    2. Token-Metriken updaten (wie oft ist Top-1 korrekt?)
    3. Loss-Analyzer updaten (wo ist Loss hoch?)
    4. Am Ende: Alle Statistiken berechnen
    
    Args:
        model: Das zu evaluierende Modell
        tokenizer: Der Tokenizer
        dataloader: DataLoader mit Validation Data
        device: 'cuda' oder 'cpu'
        compute_frequency_breakdown: Ob frequency-based breakdown berechnet werden soll
        track_categories: Ob per-category breakdown berechnet werden soll
    
    Returns:
        Dictionary mit allen Evaluation-Ergebnissen
    """
    model.eval()
    model.to(device)
    
    # Initialize metric calculators
    token_metrics = TokenMetrics(k_values=[1, 5, 10, 20])
    loss_analyzer = LossAnalyzer(
        vocab_size=tokenizer.vocab_size,
        seq_len=1024  # Anpassen wenn deine max_seq_length anders ist
    )
    
    # Get token frequencies if needed
    token_frequencies = None
    if compute_frequency_breakdown:
        print("\nComputing token frequencies for frequency-based breakdown...")
        token_frequencies = compute_token_frequencies(dataloader, tokenizer)
        print(f"Tracked {len(token_frequencies)} unique tokens")
        print(f"Most common token: ID {max(token_frequencies, key=token_frequencies.get)} "
              f"(count: {max(token_frequencies.values())})")
    
    print("\nRunning evaluation on validation set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # DEBUG: Check labels masking on first batch
            if batch_idx == 0:
                print("\n" + "="*80)
                print("DEBUG - Label Masking Check")
                print("="*80)
                
                # Check how many labels are masked
                total_labels = labels.numel()
                masked_labels = (labels == -100).sum().item()
                valid_labels = total_labels - masked_labels
                
                print(f"Total tokens: {total_labels}")
                print(f"Masked tokens (-100): {masked_labels} ({100*masked_labels/total_labels:.1f}%)")
                print(f"Valid tokens (for loss): {valid_labels} ({100*valid_labels/total_labels:.1f}%)")
                
                # Decode first sample to see what's happening
                first_input = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                first_labels_raw = labels[0].cpu().numpy()
                
                print(f"\nFirst sample input:")
                print(first_input)
                
                # Show which tokens are masked
                print(f"\nFirst 20 label values:")
                print(first_labels_raw[:20])
                print(f"\nLast 20 label values:")
                print(first_labels_raw[-20:])
                print("(-100 means masked, numbers are token IDs)")
                
                # Try to decode the non-masked part
                valid_label_ids = first_labels_raw[first_labels_raw != -100]
                if len(valid_label_ids) > 0:
                    valid_text = tokenizer.decode(valid_label_ids[:50], skip_special_tokens=False)
                    print(f"\nNon-masked (response) tokens decode to:")
                    print(valid_text[:200])
                else:
                    print("\nWARNING: NO VALID LABELS! All are masked!")
                
                print("="*80 + "\n")

            
            # Get batch categories if tracking
            batch_categories = None
            if track_categories:
                batch_categories = [
                    extract_category_from_sample(batch, i) 
                    for i in range(len(input_ids))
                ]
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            
            # Update metrics
            token_metrics.update(logits, labels, track_positions=True)
            loss_analyzer.update(logits, labels, categories=batch_categories)
    
    # Compute all metrics
    print("\nComputing final statistics...")
    results = {}
    
    # Token accuracy metrics
    results['token_accuracy'] = token_metrics.compute()
    results['position_accuracy'] = token_metrics.compute_position_accuracy()
    
    # Loss statistics
    results['overall_loss'] = loss_analyzer.compute_overall_stats()
    results['position_loss'] = loss_analyzer.compute_position_breakdown()
    
    if compute_frequency_breakdown and token_frequencies:
        results['frequency_loss'] = loss_analyzer.compute_frequency_breakdown(
            token_frequencies
        )
    
    if track_categories:
        results['category_loss'] = loss_analyzer.compute_category_breakdown()
    
    return results


def print_results(results):
    """
    Pretty-print the evaluation results.
    
    Das gibt dir einen schönen Überblick über alle Metriken.
    Wichtig: Das ist nur die Console-Ausgabe. Die eigentlichen
    Daten werden auch in MLflow und als JSON gespeichert.
    """
    print("\n" + "="*80)
    print("INTRINSIC EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\n📊 Overall Statistics:")
    overall = results['overall_loss']
    print(f"  Mean Loss:     {overall['mean_loss']:.4f}")
    print(f"  Perplexity:    {overall['perplexity']:.2f}")
    print(f"  Total Tokens:  {overall['total_tokens']:,}")
    
    # Token accuracy
    print("\n🎯 Token-Level Accuracy:")
    for metric, value in sorted(results['token_accuracy'].items()):
        k = metric.split('_')[1]  # Extract k from 'top_k_accuracy'
        print(f"  Top-{k:>2}: {value:6.2f}%")
    
    # Position breakdown (summary)
    print("\n📍 Loss by Sequence Position (summary):")
    pos_data = results['position_loss']
    if len(pos_data['mean']) > 0:
        import numpy as np
        means = np.array(pos_data['mean'])
        positions = pos_data['positions']
        
        # Show first 5, middle 5, last 5 positions
        if len(positions) > 15:
            print(f"  Positions 0-4:     Loss = {means[:5].mean():.3f} ± {means[:5].std():.3f}")
            mid_start = len(positions) // 2 - 2
            mid_end = mid_start + 5
            print(f"  Positions {mid_start}-{mid_end-1}:   Loss = {means[mid_start:mid_end].mean():.3f} ± {means[mid_start:mid_end].std():.3f}")
            print(f"  Positions {len(positions)-5}-{len(positions)-1}: Loss = {means[-5:].mean():.3f} ± {means[-5:].std():.3f}")
        else:
            for pos, mean_loss in zip(positions, means):
                print(f"  Position {pos:3d}: {mean_loss:.3f}")
    
    # Frequency breakdown
    if 'frequency_loss' in results:
        print("\n📈 Loss by Token Frequency:")
        for bucket_stats in results['frequency_loss']:
            freq_range = bucket_stats['freq_range']
            print(f"  Bucket {bucket_stats['bucket']} (freq {freq_range[1]:,} - {freq_range[0]:,}):")
            print(f"    Mean Loss: {bucket_stats['mean_loss']:.3f} ± {bucket_stats['std_loss']:.3f}")
            print(f"    Tokens: {bucket_stats['num_tokens']:,}, Samples: {bucket_stats['num_samples']:,}")
    
    # Category breakdown
    if 'category_loss' in results:
        print("\n🏷️  Loss by Category:")
        for category, stats in sorted(results['category_loss'].items()):
            print(f"  {category:15s}: {stats['mean_loss']:.3f} ± {stats['std_loss']:.3f} "
                  f"(n={stats['num_samples']})")
    
    print("\n" + "="*80)


def save_results(results, output_dir):
    """
    Save results to JSON file.
    
    JSON ist praktisch für spätere Analyse - du kannst die Daten
    einfach in Python laden und visualisieren.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / 'intrinsic_metrics.json'
    
    # Convert numpy arrays to lists for JSON serialization
    import numpy as np
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\n✅ Results saved to: {json_path}")


def log_to_mlflow(results, run_name):
    """
    Log evaluation results to MLflow.
    
    Das macht deine Evaluation nachvollziehbar und vergleichbar.
    Du kannst später verschiedene Modell-Versionen vergleichen.
    """
    # Log top-level metrics
    mlflow.log_metrics({
        "eval_mean_loss": results['overall_loss']['mean_loss'],
        "eval_perplexity": results['overall_loss']['perplexity'],
        "eval_total_tokens": results['overall_loss']['total_tokens'],
    })
    
    # Log token accuracy metrics
    for metric, value in results['token_accuracy'].items():
        mlflow.log_metric(f"eval_{metric}", value)
    
    # Log position-based insights (aggregated)
    if len(results['position_loss']['mean']) > 0:
        import numpy as np
        means = np.array(results['position_loss']['mean'])
        mlflow.log_metrics({
            "eval_loss_early_positions": means[:10].mean(),
            "eval_loss_late_positions": means[-10:].mean(),
            "eval_loss_position_variance": means.std(),
        })
    
    # Log frequency breakdown if available
    if 'frequency_loss' in results:
        for bucket in results['frequency_loss']:
            bucket_id = bucket['bucket']
            mlflow.log_metrics({
                f"eval_loss_freq_bucket_{bucket_id}": bucket['mean_loss'],
            })
    
    print(f"✅ Results logged to MLflow run: {run_name}")


def main():
    """
    Main evaluation function.
    
    Usage:
        python evaluate_intrinsic.py \
            --model_path ../block2_lora_finetuning/models/standard_r8_qkvo/adapter \
            --val_data ../../data/processed/val_chatml.jsonl \
            --output_dir results/standard_r8_qkvo
    """
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned Mistral model with intrinsic metrics'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to LoRA adapter directory')
    parser.add_argument('--base_model', type=str, 
                       default='mistralai/Mistral-7B-v0.1',
                       help='Base model name or path')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation dataset (val_chatml.jsonl)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--track_categories', action='store_true',
                       help='Track loss by category (requires category extraction)')
    parser.add_argument('--no_4bit', action='store_true',
                    help='Disable 4-bit quantization (not recommended for large models)')
    parser.add_argument('--mlflow_tracking_uri', type=str, 
                       default='http://localhost:5000',
                       help='MLflow tracking server URI')
    parser.add_argument('--mlflow_experiment', type=str,
                       default='aws-cert-mistral-evaluation',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    print("="*80)
    print("INTRINSIC EVALUATION PIPELINE")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Validation data: {args.val_data}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    
    # Start MLflow run
    run_name = f"eval_{Path(args.model_path).parent.name}"
    with mlflow.start_run(run_name=run_name):
        # Log evaluation parameters
        mlflow.log_params({
            "model_path": args.model_path,
            "base_model": args.base_model,
            "val_data": args.val_data,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples or "all",
            "use_4bit": not args.no_4bit, 
        })
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, 
            args.base_model,
            use_4bit=not args.no_4bit
            )
        
        # Load validation dataset
        print(f"\nLoading validation dataset from {args.val_data}...")
        val_dataset = create_dataset(
            args.val_data,
            tokenizer,
            max_length=1024  # Sollte mit deinem Training übereinstimmen
        )
        
        if args.max_samples:
            # For testing, use only a subset
            val_dataset = val_dataset.select(range(min(args.max_samples, len(val_dataset))))
            print(f"Using subset of {len(val_dataset)} samples for testing")
        
        # Create dataloader
        data_collator = DataCollatorForInstructionTuning(
            tokenizer=tokenizer,
            mask_instruction=True
        )
        
        dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=False,  # Don't shuffle for reproducible results
            num_workers=2
        )
        
        print(f"Validation set: {len(val_dataset)} examples")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of batches: {len(dataloader)}")
        
        # Run evaluation
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            compute_frequency_breakdown=True,
            track_categories=args.track_categories
        )
        
        # Display results
        print_results(results)
        
        # Save results
        save_results(results, args.output_dir)
        
        # Log to MLflow
        log_to_mlflow(results, run_name)
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {args.output_dir}")
        print(f"MLflow UI: {args.mlflow_tracking_uri}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()