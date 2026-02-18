# LoRA Training on Apple Silicon - Setup Guide

Complete guide for running LoRA fine-tuning on Mac Studio with Metal Performance Shaders.

## Prerequisites

- Mac with Apple Silicon (M1/M2/M3/M4)
- macOS 12.3 or later (for MPS support)
- At least 32GB RAM (64GB recommended)
- ~50GB free disk space

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements-mac.txt

# Verify MPS availability
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 2. Start MLflow (Optional but Recommended)

```bash
# In separate terminal
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Smoke Test (5-10 minutes)

Test with 100 samples to verify everything works:

```bash
python train_lora_mac.py \
    --lora_config standard \
    --test_mode \
    --mlflow_uri http://localhost:5000
```

**Monitor in Activity Monitor:**
- Memory Pressure should stay **GREEN**
- Expected peak memory: ~10-15 GB

### 4. Full Training (Overnight)

If smoke test passes, run full training:

```bash
# Start training (will take several hours)
python train_lora_mac.py \
    --lora_config standard \
    --mlflow_uri http://localhost:5000

# Or run in background
nohup python train_lora_mac.py --lora_config standard > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## Expected Performance

**Smoke Test (100 samples, 1 epoch):**
- Duration: ~5-10 minutes
- Memory: ~10-12 GB peak
- Throughput: ~2-3 samples/sec

**Full Training (3597 samples, 1 epoch):**
- Duration: ~3-6 hours (estimated)
- Memory: ~12-18 GB peak
- Cost: $0 (vs ~$2-4 on cloud T4)

## Configuration Options

### LoRA Configs

```bash
# Minimal (fastest, smallest adapter)
--lora_config minimal

# Standard (recommended)
--lora_config standard

# Aggressive (more capacity)
--lora_config aggressive
```

### Adjusting Batch Size

Edit `config_mac.py`:

```python
# Conservative (default)
per_device_train_batch_size: int = 2

# If smoke test shows low memory usage, try:
per_device_train_batch_size: int = 4  # 2x faster
```

## Troubleshooting

### Memory Issues

**Symptoms:** System becomes unresponsive, memory pressure RED

**Solutions:**
1. Reduce batch size in `config_mac.py`:
   ```python
   per_device_train_batch_size: int = 1
   ```

2. Close other applications

3. Disable browser/heavy apps during training

### MPS Not Available

**Error:** `MPS not available`

**Solutions:**
1. Verify macOS version: `sw_vers` (need 12.3+)
2. Check PyTorch installation:
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

### Slow Training

**Issue:** < 1 sample/sec

**Check:**
1. Verify MPS is being used (not CPU):
   ```python
   import torch
   print(next(model.parameters()).device)  # Should show 'mps'
   ```

2. Check Activity Monitor → GPU History (should show activity)

## Output Files

After training, find your adapter:

```
training/models/standard_r8_qkvo_mac/
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.bin          # ← Your trained LoRA weights
├── checkpoint-100/
├── checkpoint-200/
└── logs/
```

## Comparing to Cloud Training

**Cloud (T4 GPU):**
- Setup: 10-15 min (instance start + docker)
- Training: 2-4 hours
- Cost: $0.50-1.00/hour = ~$1-4 total
- Complexity: AWS, Docker, K8s

**Mac (Apple Silicon):**
- Setup: 0 min (already running)
- Training: 3-6 hours
- Cost: $0
- Complexity: Just run the script

**Trade-off:** 2-4 hours extra wait for $0 cost + full data sovereignty

## Next Steps

After training completes:

1. **View in MLflow:** http://localhost:5000
2. **Test the adapter:** See `06-lora-serving.md` in blog series
3. **Compare with v1:** A/B test in vLLM (Post 9)

## Need Help?

**Check logs:**
```bash
# Training logs
cat training.log

# MLflow logs
cat mlruns/*/meta.yaml
```

**Memory monitoring during training:**
```bash
# Real-time memory usage
while true; do 
    ps aux | grep python | grep train_lora_mac
    sleep 5
done
```
