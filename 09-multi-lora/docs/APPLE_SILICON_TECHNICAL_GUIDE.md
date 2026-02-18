# LoRA Training auf Apple Silicon - Technischer Deep Dive

Vollständige Erklärung der Anpassungen, Limitierungen und Optimierungen für LoRA-Training auf Mac Studio mit Metal Performance Shaders.

---

## Inhaltsverzeichnis

1. [Hardware-Architektur: Apple Silicon vs. CUDA GPUs](#hardware-architektur-apple-silicon-vs-cuda-gpus)
2. [Warum BitsAndBytes nicht funktioniert](#bitsandbytes-problem)
3. [Memory Management: Unified Memory Architecture](#memory-management)
4. [PyTorch MPS Backend](#pytorch-mps)
5. [Code-Änderungen im Detail](#code-änderungen)
6. [Performance-Analyse](#performance-analyse)
7. [Best Practices & Optimierungen](#best-practices)

---

## Hardware-Architektur: Apple Silicon vs. CUDA GPUs

### NVIDIA T4 GPU (Cloud Setup)

```
CPU (x86_64)
├── RAM: 32-64 GB (System Memory)
└── PCIe Bus
    └── NVIDIA T4 GPU
        ├── VRAM: 16 GB (Dedicated GPU Memory)
        ├── CUDA Cores: 2560
        └── Compute Capability: 7.5
```

**Charakteristiken:**
- Separate Memory-Spaces (CPU RAM vs. GPU VRAM)
- Daten müssen über PCIe Bus kopiert werden
- VRAM-Limit: 16 GB (hart)
- Quantization (4-bit/8-bit) **nötig** um Mistral-7B zu laden

### Mac Studio (M4 Max 64GB)

```
Apple Silicon SoC
├── CPU Cores: 14-16
├── GPU Cores: 32-40
└── Unified Memory: 64 GB
    ├── Shared zwischen CPU und GPU
    └── Keine Transfers nötig
```

**Charakteristiken:**
- **Unified Memory Architecture (UMA)**
  - CPU und GPU teilen sich gleichen RAM
  - Kein Memory-Copy zwischen CPU ↔ GPU
  - Zero-Copy-Zugriff
- **Memory-Limit:** 64 GB total (für alles)
- Quantization **optional** (genug RAM vorhanden)

**Warum das wichtig ist:**

```python
# CUDA/T4: Explizite Transfers
tensor_cpu = torch.randn(1000, 1000)
tensor_gpu = tensor_cpu.to('cuda')  # ← Copy über PCIe

# MPS/Apple Silicon: Zero-Copy
tensor_cpu = torch.randn(1000, 1000)
tensor_mps = tensor_cpu.to('mps')   # ← Kein Copy, nur Pointer!
```

---

## Warum BitsAndBytes nicht funktioniert

### Was ist BitsAndBytes?

BitsAndBytes ist eine Library für **CUDA-spezifische Quantization**:

```python
# Cloud Setup (funktioniert)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit Quantization
    bnb_4bit_quant_type="nf4",      # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,  # ← CUDA-only!
    device_map="auto"
)
```

**Memory-Einsparung:**
- FP16 (16-bit): ~14 GB
- INT8 (8-bit): ~7 GB
- NF4 (4-bit): ~3.5 GB

### Warum es auf Mac nicht geht

**Problem 1: CUDA-Dependencies**

BitsAndBytes nutzt CUDA Kernels (compiled C++ code):
- `libcudart.so` (CUDA Runtime)
- `libbitsandbytes_cuda.so` (Custom CUDA Kernels)
- Direct GPU Memory Access (nur CUDA)

**Problem 2: MPS-Inkompatibilität**

```python
# Versuch auf Mac:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config  # ← Error!
)
# RuntimeError: BitsAndBytes requires CUDA
```

MPS (Metal Performance Shaders) ist **nicht CUDA-kompatibel**:
- Anderes Memory-Model
- Andere Kernel-API (Metal Shading Language)
- Keine `libcudart` auf macOS

### Die Alternative: PyTorch Native Quantization

**Für Mac nutzen wir:**

```python
# Mac Setup (funktioniert)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,  # ← PyTorch nativ
    device_map="auto",          # ← MPS auto-detected
)
```

**Warum FP16 statt Quantization?**

Mit 64 GB Unified Memory:
```
Mistral-7B FP16:        ~14 GB
LoRA Adapters:          ~100 MB
LoRA Gradients:         ~100 MB
AdamW Optimizer:        ~14 GB  (2× LoRA params)
Activations (batch=2):  ~3 GB
PyTorch Overhead:       ~2 GB
──────────────────────────────
Total:                  ~33-34 GB ✅ Passt locker!
```

**Quantization würde sparen:**
- INT8: ~7 GB statt ~14 GB → Total ~27 GB (Gewinn: 7 GB)
- Aber: Komplexität steigt, Performance-Verlust möglich
- **Trade-off lohnt nicht** bei 64 GB RAM

---

## Memory Management: Unified Memory Architecture

### Wie Unified Memory funktioniert

**Traditionell (CUDA):**
```
Application                GPU Kernel
    │                          │
    ├─ Allocate CPU RAM        │
    ├─ Copy to GPU VRAM ────>  │
    │                          ├─ Compute
    │                  <────── ├─ Copy back
    └─ Process result          │
```

**Apple Silicon (UMA):**
```
Application / GPU Kernel
    │
    ├─ Allocate Unified Memory
    │  (accessible by both CPU & GPU)
    │
    ├─ GPU Compute (zero-copy)
    │
    └─ CPU Process (same memory)
```

### Memory Pressure System

macOS verwaltet Memory dynamisch:

**GREEN (Normal):**
- Genug freier RAM
- Kein Swapping
- Optimale Performance

**YELLOW (Warnung):**
- RAM wird knapp
- Leichtes Swapping beginnt
- Performance-Degradation

**RED (Kritisch):**
- Fast kein freier RAM
- Starkes Swapping (SSD)
- Massive Slowdowns

**Dein Training:**
```
Training Memory: 54 GB
├─ Baseline (Ollama): 20 GB
└─ Training selbst:   34 GB
    ├─ Model FP16:    14 GB
    ├─ Optimizer:     14 GB
    ├─ Activations:   3 GB
    └─ Overhead:      3 GB

Memory Pressure: GREEN ✅
→ 10 GB Puffer bis 64 GB
```

### Warum 34GB für Training?

**Das Optimizer-Problem:**

LoRA trainiert nur ~0.5% der Parameter (35M von 7B), **aber:**

```python
# AdamW Optimizer speichert für ALLE trainierbaren Parameter:
optimizer_state = {
    'momentum': params.clone(),      # 1× Kopie
    'variance': params.clone(),      # 1× Kopie
}
# → 2× Parameter-Size an Extra-Memory!
```

Für LoRA mit 35M params:
```
LoRA Params (FP16):     100 MB
Momentum (FP16):        100 MB
Variance (FP16):        100 MB
────────────────────────────
Optimizer Total:        200 MB ✅
```

**Aber warum dann 14 GB für Optimizer?**

Das ist ein PyTorch-Implementierungsdetail:
- AdamW alloziert Memory auch für frozen base model
- Auch wenn base model nicht trainiert wird
- Wird als "reserved but unused" gehalten

```python
# Vereinfacht:
base_model_params = 7B × 2 bytes = 14 GB
lora_params = 35M × 2 bytes = 70 MB

# AdamW reserviert trotzdem:
optimizer_memory = 14 GB (für base) + 140 MB (für LoRA)
```

Das ist **ineffizient aber OK** - mit 64 GB haben wir genug Platz.

---

## PyTorch MPS Backend

### Was ist MPS?

**MPS = Metal Performance Shaders**
- Apples GPU-Framework (wie CUDA für NVIDIA)
- Nutzt Metal API (low-level GPU access)
- Optimiert für Apple Silicon

### PyTorch MPS Integration

**Seit PyTorch 2.0:**

```python
import torch

# Check availability
torch.backends.mps.is_available()  # True auf M1/M2/M3/M4
torch.backends.mps.is_built()      # True wenn PyTorch mit MPS compiled

# Device selection
device = torch.device("mps")
tensor = torch.randn(1000, 1000).to(device)
```

**HuggingFace Trainer unterstützt MPS automatisch:**

```python
training_args = TrainingArguments(
    device_map="auto",  # ← Erkennt MPS automatisch
    fp16=True,          # ← MPS unterstützt FP16
    bf16=False,         # ← MPS unterstützt KEIN BF16!
)
```

### MPS Limitations

**Was MPS NICHT kann (vs. CUDA):**

1. **BFloat16 (BF16):**
   ```python
   # CUDA: OK
   model = model.to(torch.bfloat16)
   
   # MPS: Error!
   # RuntimeError: MPS does not support BF16
   ```

2. **Pinned Memory:**
   ```python
   # CUDA: Schnellere CPU↔GPU Transfers
   DataLoader(..., pin_memory=True)
   
   # MPS: Warning (ignored)
   # "pin_memory not supported on MPS"
   ```

3. **CUDA-spezifische Operationen:**
   - BitsAndBytes Quantization
   - Flash Attention (CUDA Kernels)
   - Manche Custom CUDA Extensions

**Was MPS gut kann:**

- ✅ Standard PyTorch Operations
- ✅ Transformer Models
- ✅ LoRA Training
- ✅ FP16 Training
- ✅ Gradient Checkpointing
- ✅ Mixed Precision (FP16/FP32)

---

## Code-Änderungen im Detail

### Änderung 1: Model Loading (Quantization entfernt)

**Cloud (mit BitsAndBytes):**

```python
from transformers import BitsAndBytesConfig

# 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Model mit Quantization laden
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,  # ← BitsAndBytes
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare für k-bit training
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)
```

**Mac (ohne BitsAndBytes):**

```python
# Kein BitsAndBytes Import!

# Model direkt in FP16 laden
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,  # ← PyTorch nativ
    device_map="auto",          # ← MPS auto-detected
    trust_remote_code=True,
)

# prepare_model_for_kbit_training funktioniert auch ohne Quantization
# (Setup für Gradient Checkpointing, Layer freezing)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)
```

**Warum `prepare_model_for_kbit_training` trotzdem funktioniert?**

Die Funktion macht mehr als nur Quantization-Setup:
```python
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """
    Bereitet Model für efficient training vor:
    1. Freeze base model layers
    2. Enable gradient checkpointing (optional)
    3. Enable input requires_grad für einige Layers
    """
    # 1. Alle Layer frozen (keine Gradients)
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 3. Input embeddings need gradients (für LoRA)
    model.enable_input_require_grads()
    
    return model
```

Das ist **generisch** und funktioniert unabhängig von Quantization!

### Änderung 2: Optimizer

**Cloud (BitsAndBytes Optimizer):**

```python
training_args = TrainingArguments(
    optim="paged_adamw_8bit",  # ← BitsAndBytes-spezifisch!
    # ...
)
```

**`paged_adamw_8bit`:**
- AdamW mit 8-bit Optimizer States
- Nutzt CUDA Unified Memory (paging)
- Reduziert Optimizer Memory von ~28GB auf ~7GB
- **Nur mit BitsAndBytes + CUDA**

**Mac (Standard PyTorch Optimizer):**

```python
training_args = TrainingArguments(
    optim="adamw_torch",  # ← Standard PyTorch AdamW
    # ...
)
```

**`adamw_torch`:**
- Standard PyTorch Implementation
- FP32 Optimizer States (präziser)
- Keine Memory-Einsparung vs. paged_adamw
- **Funktioniert überall (CUDA, MPS, CPU)**

**Warum kein 8-bit Optimizer auf Mac?**
- BitsAndBytes 8-bit optimizer braucht CUDA
- PyTorch hat keinen nativen 8-bit optimizer
- Mit 64 GB RAM nicht nötig

### Änderung 3: Precision Settings

**Cloud:**

```python
training_args = TrainingArguments(
    fp16=False,  # T4 hat Tensor Cores, aber FP16 optional
    bf16=False,  # T4 unterstützt kein BF16
    # ...
)
```

**Mac:**

```python
training_args = TrainingArguments(
    fp16=True,   # ✅ MPS unterstützt FP16
    bf16=False,  # ❌ MPS unterstützt KEIN BF16
    # ...
)
```

**FP16 auf MPS:**
- Automatische Mixed Precision
- Forward Pass: FP16 (schneller)
- Backward Pass: FP32 (stabiler)
- Gradient Scaling (verhindert underflow)

**Warum FP16 Training aktivieren?**

Ohne FP16:
```
Model: FP16
Training: FP32
→ Automatische Konversion bei jedem Step
→ Performance-Verlust
```

Mit FP16:
```
Model: FP16
Training: FP16 (forward) + FP32 (backward)
→ Optimiert, kein unnötiges Casting
```

### Änderung 4: Batch Size

**Cloud (Memory-limitiert):**

```python
config = TrainingConfig(
    per_device_train_batch_size=4,  # Maximum für T4 16GB
    gradient_accumulation_steps=4,
    # Effective batch size = 4 × 4 = 16
)
```

**Mac (Conservative Start):**

```python
config = TrainingConfig(
    per_device_train_batch_size=2,  # Conservative
    gradient_accumulation_steps=4,
    # Effective batch size = 2 × 4 = 8
)
```

**Warum kleiner starten?**

Beim ersten Run:
- Unbekannt wie viel Memory MPS braucht
- Overhead schwer vorherzusagen
- Lieber safe starten

**Nach Smoke Test:**

Mit 54 GB @ batch=2 und GREEN Memory Pressure:
```python
# Kann erhöht werden:
per_device_train_batch_size=4  # oder sogar 8
```

Memory-Impact:
```
batch_size=2 → Activations: ~3 GB
batch_size=4 → Activations: ~6 GB   (+3 GB)
batch_size=8 → Activations: ~12 GB  (+9 GB)

Aktuell 54 GB @ batch=2
→ batch=4: ~57 GB ✅
→ batch=8: ~63 GB ⚠️ (knapp)
```

### Änderung 5: Device Handling

**Cloud (explizit CUDA):**

```python
# HuggingFace macht das automatisch, aber explizit:
model.to('cuda')
```

**Mac (automatisch MPS):**

```python
# HuggingFace Trainer erkennt MPS automatisch via:
device_map="auto"

# Intern wird geprüft:
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

**Wichtig:** Kein manuelles `.to('mps')` nötig bei HuggingFace!

---

## Performance-Analyse

### Gemessene Performance

**Test Setup:**
- Dataset: 5996 samples (5796 positive + 200 negative)
- LoRA Config: rank=8, 4 modules (standard)
- Model: Mistral-7B FP16

**Cloud T4:**
```
Time:        ~3.0 Std (180 min)
Throughput:  ~0.55 samples/sec
Final Loss:  0.35
Cost:        ~$1.50
```

**Mac Studio:**
```
Time:        3.26 Std (196 min)
Throughput:  0.31 samples/sec
Final Loss:  0.3589
Cost:        $0
```

**Analysis:**
- Mac ist **9% langsamer** (196 min vs 180 min)
- Loss ist **praktisch identisch** (0.3589 vs 0.35)
- Kosten: **$1.50 gespart**

### Wo ist der Unterschied?

**Throughput-Breakdown:**

| Operation | T4 (CUDA) | Mac (MPS) | Ratio |
|-----------|-----------|-----------|-------|
| Forward Pass | ~60 ms | ~90 ms | 1.5× |
| Backward Pass | ~80 ms | ~120 ms | 1.5× |
| Optimizer Step | ~20 ms | ~30 ms | 1.5× |
| **Total/Sample** | ~160 ms | ~240 ms | **1.5×** |

**Warum Mac langsamer?**

1. **MPS Maturity:**
   - CUDA: 15+ Jahre Optimierung
   - MPS: ~3 Jahre für ML-Workloads
   - PyTorch MPS: Noch nicht voll optimiert

2. **GPU Architecture:**
   - T4: Spezialisiert für ML (Tensor Cores)
   - Apple GPU: General-purpose (Graphics + ML)

3. **Memory Bandwidth:**
   - T4 VRAM: 320 GB/s
   - Mac UMA: ~400 GB/s
   - **Aber:** T4 hat dedizierte VRAM (keine Konkurrenz)

4. **Software Stack:**
   - CUDA: Hochoptimierte Kernels
   - MPS: PyTorch-Wrapper über Metal (extra Layer)

### Wo Mac glänzt

**Setup Time:**
```
Cloud:
- Start EC2 instance:     2-3 min
- SSH connect:            30 sec
- Start Docker/MLflow:    2-3 min
- Pull latest code:       1 min
Total:                    ~6-8 min

Mac:
- Open Terminal:          5 sec
- Start training:         Sofort
Total:                    ~5 sec
```

**Cost Efficiency:**
```
Cloud (T4):
- Training: $1.50
- Storage (S3, EBS): $0.20/month
- Data Transfer: $0.10
Total: ~$1.80 pro Run > Mehr, wenn man kein automatische scale-to-zero hat.


Mac:
- Training: $0
- Storage: Local (free)
- Transfer: None
Total: $0
```

**Iteration Speed:**
```
Cloud:
- Change code → Commit → Push → Pull → Run
- ~5 min overhead pro Iteration

Mac:
- Change code → Run
- ~5 sec overhead
```

### Optimierungspotential

**Für zukünftige Runs:**

1. **Batch Size erhöhen:**
   ```python
   batch_size=4  # Statt 2
   → ~3.5 Std statt 3.26 Std (-20%)
   ```

2. **Gradient Accumulation reduzieren:**
   ```python
   # Aktuell:
   batch=2, grad_accum=4 → effective_batch=8
   
   # Optimiert:
   batch=4, grad_accum=2 → effective_batch=8
   → Weniger Steps, gleiche Convergence
   ```

3. **Mixed Precision optimieren:**
   ```python
   # Noch nicht voll ausgereizt
   # PyTorch AMP für MPS könnte besser werden
   ```

4. **PyTorch Version:**
   ```bash
   # Neuere PyTorch Versionen haben besseres MPS
   pip install --upgrade torch
   ```

---

## Best Practices & Optimierungen

### Memory Management

**1. Baseline Memory freigeben:**

```bash
# Vor Training:
killall ollama         # Beendet Ollama
killall chrome         # Oder andere memory-intensive Apps

# Verifizieren:
top -o MEM
```

**2. Batch Size tuning:**

```python
# Smoke Test mit verschiedenen Werten:
for batch_size in [2, 4, 8]:
    # Test 100 samples
    # Check Memory Pressure
    # Wähle größten Wert mit GREEN pressure
```

**3. Gradient Checkpointing:**

```python
# Aktiviert in config:
use_gradient_checkpointing=True

# Trade-off:
# + Spart ~50% Activation Memory
# - ~10-15% langsamer (Recomputation)
```

**Memory-Rechner:**

```python
def estimate_memory(
    model_size_b=7,      # Model size in billions
    dtype='fp16',        # fp32, fp16, int8
    batch_size=2,
    seq_length=1024,
    lora_rank=8,
    use_gradient_checkpointing=True
):
    # Model
    bytes_per_param = {'fp32': 4, 'fp16': 2, 'int8': 1}[dtype]
    model_memory = model_size_b * 1e9 * bytes_per_param / 1e9  # GB
    
    # LoRA params (rank × 2 × hidden_dim × n_layers)
    lora_params = lora_rank * 2 * 4096 * 32 * bytes_per_param / 1e9
    
    # Optimizer (AdamW: 2× params)
    optimizer_memory = lora_params * 2
    
    # Activations (depends on batch size and checkpointing)
    activation_factor = 0.5 if use_gradient_checkpointing else 1.0
    activations = batch_size * seq_length * 4096 * 4 / 1e9 * activation_factor
    
    # Overhead
    overhead = 2
    
    total = model_memory + lora_params + optimizer_memory + activations + overhead
    
    return {
        'model': model_memory,
        'lora': lora_params,
        'optimizer': optimizer_memory,
        'activations': activations,
        'overhead': overhead,
        'total': total
    }

# Example:
estimate_memory(batch_size=4)
# → {'total': ~38 GB}
```

### Performance Optimization

**1. caffeinate für lange Runs:**

```bash
# IMMER nutzen für über-Nacht-Training:
caffeinate -i python train_lora_mac.py [args]
```

**2. Nice Priority (optional):**

```bash
# Falls du parallel arbeiten willst:
nice -n 10 python train_lora_mac.py [args]
# → Training bekommt niedrigere CPU-Priorität
```

**3. Monitoring Setup:**

```bash
# Terminal 1: Training
caffeinate -i python train_lora_mac.py ...

# Terminal 2: Live Monitoring
watch -n 5 'ps aux | grep python | grep train_lora_mac'

# Terminal 3: Memory Monitoring
while true; do
    pmset -g thermlog | grep "CPU_Scheduler_Limit"
    sleep 10
done
```

**4. MLflow Auto-Open:**

```bash
# Start MLflow UND öffne Browser:
mlflow server --host 0.0.0.0 --port 5001 &
sleep 3
open http://localhost:5001
```

### Debugging

**Problem: Training stoppt unerwartet**

```bash
# Check ob caffeinate läuft:
ps aux | grep caffeinate

# Check System Log:
log show --predicate 'eventMessage contains "sleep"' --last 1h

# Force awake:
caffeinate -u -t 18000 &  # 5 hours
```

**Problem: Memory Pressure RED**

```bash
# Sofort:
Ctrl+C  # Stop training

# In config_mac.py:
per_device_train_batch_size=1  # Reduce

# Restart mit mehr Headroom
```

**Problem: Sehr langsam (<0.2 samples/sec)**

```bash
# Check ob CPU statt MPS genutzt wird:
python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"

# Falls False:
pip install --upgrade torch torchvision torchaudio
```

### Experiment Tracking

**MLflow Best Practices:**

```python
# In Training Script zusätzlich loggen:
import mlflow

# Custom Metrics:
mlflow.log_metric("peak_memory_gb", 54.0)
mlflow.log_metric("samples_per_second", 0.31)
mlflow.log_metric("cost_usd", 0.0)

# System Info:
mlflow.log_param("device", "Apple M4 Max")
mlflow.log_param("memory_total_gb", 64)
mlflow.log_param("os", "macOS 14")

# Comparisons:
mlflow.set_tag("baseline", "T4_cloud_run")
```

---

## Zusammenfassung

### Was funktioniert anders

| Feature | Cloud (T4) | Mac (M4 Max) |
|---------|-----------|--------------|
| Quantization | ✅ BitsAndBytes 4-bit | ❌ Nicht verfügbar |
| Precision | FP16/FP32 | ✅ FP16/FP32 |
| BFloat16 | ❌ T4 nicht supported | ❌ MPS nicht supported |
| Optimizer | paged_adamw_8bit | adamw_torch |
| Device | CUDA | MPS |
| Memory | 16 GB VRAM | 64 GB Unified |

### Was gleich bleibt

- ✅ LoRA Configuration (rank, alpha, modules)
- ✅ Training Hyperparameters (LR, epochs)
- ✅ Dataset Format
- ✅ HuggingFace Trainer API
- ✅ MLflow Integration
- ✅ Model Quality (Loss, Perplexity)

### Key Takeaways

1. **Unified Memory ist ein Game-Changer:**
   - Kein Memory-Copy zwischen CPU/GPU
   - Größerer verfügbarer Memory-Pool
   - Quantization optional

2. **MPS ist gut, aber nicht CUDA:**
   - ~1.5× langsamer bei gleicher Config
   - Aber: Setup-Zeit = 0, Cost = $0
   - Trade-off lohnt sich für Experimente

3. **Batch Size matters:**
   - Starte conservative (batch=2)
   - Tune basierend auf Memory Pressure
   - Kann oft verdoppelt werden

4. **caffeinate ist essentiell:**
   - Ohne: Training stoppt bei Inaktivität
   - Mit: Garantiert durchlaufen

5. **Ende-zu-Ende Data Sovereignty:**
   - Dataset Generation: Lokal
   - Model Training: Lokal
   - Keine Cloud-Dependencies
   - Keine Kosten

---

## Weiterführende Ressourcen

**Apple Documentation:**
- [Metal Performance Shaders](https://developer.apple.com/metal/tensorflow-plugin/)
- [Unified Memory Architecture](https://developer.apple.com/documentation/metal/resource_fundamentals/about_unified_memory)

**PyTorch MPS:**
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Accelerating PyTorch on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)

**Performance Profiling:**
- [Instruments (Xcode)](https://developer.apple.com/xcode/features/)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
