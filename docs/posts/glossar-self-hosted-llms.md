# Glossar: Self-Hosted LLMs für Datensouveränität

**Projekt:** Self-Hosted LLMs für Datensouveränität – Blog-Serie  
**Zweck:** Zentrale Begriffserklärungen für alle Posts

Dieses Glossar erklärt technische Begriffe, die in der Blog-Serie verwendet werden. Die Begriffe sind alphabetisch sortiert und nach Kategorien gruppiert.

---

## Inhaltsverzeichnis

- [Training & Optimization](#training--optimization)
- [Model Architecture & LoRA](#model-architecture--lora)
- [Memory & Hardware](#memory--hardware)
- [Serving & Infrastructure](#serving--infrastructure)
- [Evaluation & Metriken](#evaluation--metriken)
- [Data & Datasets](#data--datasets)

---

## Training & Optimization

### Activation / Activations
Zwischenergebnisse, die während des Forward Pass (Berechnung der Predictions) eines Neural Networks entstehen. Diese müssen für den Backward Pass (Gradient-Berechnung) im Speicher gehalten werden. Bei großen Models und Batches können Activations mehrere GB VRAM benötigen.

### Backward Pass
Der Rückwärts-Durchlauf durch das Neural Network, bei dem Gradients (Ableitungen der Loss-Funktion) für alle Parameter berechnet werden. Nutzt die im Forward Pass gespeicherten Activations.

### Batch Size
Anzahl der Trainingsbeispiele, die gleichzeitig verarbeitet werden. Größere Batches führen zu stabileren Gradients, benötigen aber mehr GPU Memory.

### Effective Batch Size
Die tatsächliche Anzahl an Samples, über die Gradients akkumuliert werden, bevor ein Optimizer-Step durchgeführt wird. Berechnet als: `per_device_batch_size × gradient_accumulation_steps × num_gpus`.

### Epoch / Epoche
Ein kompletter Durchlauf durch das gesamte Trainings-Dataset. Bei 1 Epoche sieht das Model jeden Trainings-Sample genau einmal.

### Forward Pass
Der Vorwärts-Durchlauf durch das Neural Network, bei dem aus Input-Daten Predictions berechnet werden. Erzeugt Activations, die für den Backward Pass gespeichert werden müssen.

### Gradient
Die Ableitung der Loss-Funktion nach einem Parameter. Zeigt an, in welche Richtung und wie stark ein Parameter angepasst werden sollte, um die Loss zu reduzieren.

### Gradient Accumulation
Technik, bei der Gradients über mehrere kleine Batches gesammelt werden, bevor ein Optimizer-Update durchgeführt wird. Ermöglicht große effective Batch Sizes bei begrenztem GPU Memory.

**Beispiel:** Mit `batch_size=4` und `gradient_accumulation_steps=4` ist die effective Batch Size 16, aber der Memory-Verbrauch entspricht nur Batch Size 4.

### Gradient Checkpointing
Memory-Optimierungstechnik, die Activations nicht speichert, sondern während des Backward Pass neu berechnet. Trade-off: ~30% weniger Memory gegen ~20% langsameres Training.

### Learning Rate
Schrittweite, mit der Parameter während des Trainings angepasst werden. Höhere Learning Rates führen zu schnellerer Konvergenz, aber Risiko für Instabilität. Typische Werte: 1e-5 bis 3e-4.

### Loss / Loss Function
Metrik, die misst, wie weit die Model-Predictions von den tatsächlichen Targets entfernt sind. Das Ziel des Trainings ist, die Loss zu minimieren. Bei Language Models typischerweise Cross-Entropy Loss.

### Mixed Precision Training
Training in reduzierter numerischer Präzision (FP16 oder BF16 statt FP32). Reduziert Memory-Verbrauch und beschleunigt Training auf modernen GPUs, ohne signifikanten Qualitätsverlust.

### Optimizer / Optimizer States
Algorithmus, der bestimmt, wie Parameter basierend auf Gradients aktualisiert werden. AdamW (häufig verwendet) speichert zusätzlich Momentum und Variance für jeden Parameter — das verdoppelt den Memory-Bedarf.

### Overfitting
Wenn ein Model zu stark auf die Trainingsdaten spezialisiert ist und bei neuen Daten schlecht performt. Erkennbar wenn Training Loss fällt, aber Validation Loss steigt.

### Perplexity
Metrik für Language Models. Misst, wie "überrascht" das Model von den tatsächlichen Tokens ist. Niedrigere Werte = besseres Model. Berechnet als `exp(loss)`.

### Warmup Steps
Anzahl der Trainings-Steps, in denen die Learning Rate von 0 graduell auf den Zielwert erhöht wird. Stabilisiert Training am Anfang.

### Weight Decay
Regularisierungstechnik, die Parameter leicht in Richtung 0 zieht, um Overfitting zu vermeiden. Typische Werte: 0.01 bis 0.1.

---

## Model Architecture & LoRA

### Adapter
Kleine, trainierbare Komponente, die an ein eingefrorenes Base Model angefügt wird. Bei LoRA sind Adapter Low-Rank-Matrizen, die nur wenige MB groß sind.

### Base Model
Das ursprüngliche, vortrainierte Model ohne Fine-tuning. Zum Beispiel `mistralai/Mistral-7B-v0.1` — trainiert auf großen Text-Corpora für allgemeine Sprachverständnis.

### Causal Language Modeling (CLM)
Aufgabe, bei der das Model das nächste Token basierend auf vorherigen Tokens vorhersagt. Standard-Training-Objective für generative Language Models.

### Checkpoint
Gespeicherter Zustand eines Models während des Trainings. Enthält Model Weights, Optimizer States, und Training Configuration. Ermöglicht Fortsetzen des Trainings oder Rollback.

### EOS Token (End-of-Sequence)
Spezielles Token, das das Ende einer generierten Sequenz markiert. Kritisch für Inference — ohne korrekte EOS-Generation stoppt das Model nie.

### Fine-tuning
Anpassen eines vortrainierten Models an eine spezifische Aufgabe durch Training auf task-spezifischen Daten. Weniger rechenintensiv als Training from Scratch.

### Frozen / Eingefrorene Parameter
Parameter, die während des Trainings nicht aktualisiert werden. Bei LoRA ist das Base Model eingefroren, nur die Adapter werden trainiert.

### Instruct Model
Model, das speziell für Instruction-Following trainiert wurde. Zum Beispiel `Mistral-7B-Instruct-v0.1`. Versteht Anweisungen besser als Base Models.

### Instruction Tuning
Fine-tuning auf Instruction-Output-Paaren, um ein Model beizubringen, menschlichen Anweisungen zu folgen. Unterscheidet sich von Continued Pre-training (reines Wissen).

### LoRA (Low-Rank Adaptation)
Parameter-effiziente Fine-tuning-Methode. Statt alle Parameter zu trainieren, werden kleine Low-Rank-Matrizen hinzugefügt. Formel: `W' = W + (α/r) · B · A`.

### LoRA Alpha (α)
Skalierungsfaktor für LoRA-Updates. Bestimmt zusammen mit Rank die Stärke der Adaptationen. Typisch: `alpha = 2 × rank`.

### LoRA Rank (r)
Dimensionalität der Low-Rank-Dekomposition. Höherer Rank = mehr Kapazität, aber mehr Parameter. Typische Werte: 4, 8, 16, 32.

### PEFT (Parameter-Efficient Fine-Tuning)
Oberbegriff für Methoden, die nur einen Bruchteil der Model-Parameter trainieren. LoRA ist eine PEFT-Methode.

### QLoRA (Quantized LoRA)
Kombination aus 4-bit Quantization (für das Base Model) und LoRA (für die Adapter). Ermöglicht Training großer Models auf Consumer-GPUs.

### Target Modules
Layer des Base Models, auf die LoRA-Adapter angewendet werden. Bei Transformern typischerweise Attention Projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### Tokenizer
Komponente, die Text in Token-IDs umwandelt (und umgekehrt). Jedes Model hat einen spezifischen Tokenizer. Wichtige Special Tokens: `bos_token`, `eos_token`, `pad_token`, `unk_token`.

### Trainable Parameters
Parameter, die während des Trainings aktualisiert werden. Bei LoRA typischerweise < 1% der gesamten Model-Parameter.

---

## Memory & Hardware

### 4-bit Quantization
Komprimierung von Model Weights von 16-bit Float auf 4-bit Integer. Reduziert Memory-Verbrauch um ~75%. Bei QLoRA: Base Model in 4-bit, Adapter in FP16.

### BF16 (Brain Float 16)
16-bit Floating Point Format mit größerem Exponent als FP16. Besser für Training (numerisch stabiler), aber nur auf neueren GPUs (Ampere, Hopper) verfügbar.

### Device Map
Strategie, wie ein Model über verfügbare Hardware-Ressourcen (GPUs, CPU, Disk) verteilt wird. `device_map="auto"` wählt automatisch die beste Verteilung.

### FP16 (Float 16)
16-bit Floating Point Format. Standard für Mixed Precision Training. Halbiert Memory-Verbrauch gegenüber FP32.

### FP32 (Float 32)
32-bit Floating Point Format. Standard-Präzision für Neural Networks, aber memory-intensiv.

### GPU / VRAM
Graphics Processing Unit / Video RAM. Spezialisierte Hardware für parallele Berechnungen. VRAM ist der Arbeitsspeicher der GPU — limitierender Faktor beim Training großer Models.

### NF4 (Normal Float 4)
Spezielle 4-bit Quantization, die die Verteilung von Neural Network Weights besser approximiert als Standard-Integer-Quantization. Verwendet in QLoRA.

### OOM (Out of Memory)
Fehler, der auftritt, wenn GPU Memory erschöpft ist. Häufigste Ursache: zu große Batch Size, zu lange Sequences, oder zu großes Model.

### Peak Memory
Maximaler GPU Memory-Verbrauch während des Trainings. Tritt typischerweise während des Backward Pass auf.

### T4 GPU
NVIDIA Tesla T4, häufige Cloud-GPU mit 16GB VRAM. Geeignet für Inference und Training kleinerer Models (bis ~7B mit QLoRA).

---

## Serving & Infrastructure

### Cluster
Gruppe von Servern (Nodes), die zusammen arbeiten. Bei Kubernetes: Alle Nodes im gleichen Cluster können Pods untereinander erreichen.

### Deployment
Kubernetes-Ressource, die Pods verwaltet. Definiert wie viele Replicas laufen sollen, welches Image, welche Ressourcen, etc.

### Inference
Nutzen eines trainierten Models für Predictions auf neuen Daten. Gegensatz zu Training (Anpassen der Parameter).

### Kubernetes (K8s)
Container-Orchestrierungssystem. Verwaltet Deployment, Scaling, und Networking von containerisierten Applikationen.

### NetworkPolicy
Kubernetes-Ressource, die definiert, welche Pods miteinander (Ingress/Egress) kommunizieren dürfen. Firewall auf Cluster-Level.

### Pod
Kleinste deploybare Einheit in Kubernetes. Kann einen oder mehrere Container enthalten.

### Service
Kubernetes-Ressource, die einen stabilen Netzwerk-Endpunkt für Pods bereitstellt. Types: ClusterIP (intern), LoadBalancer (extern), NodePort.

### Serving
Bereitstellung eines Models als API-Endpunkt. Tools wie vLLM optimieren Serving für LLMs (KV-Cache, Continuous Batching, etc.).

### vLLM
High-Performance Inference-Server für LLMs. Nutzt PagedAttention und Continuous Batching für effizienten Throughput.

---

## Evaluation & Metriken

### Eval / Evaluation
Bewertung eines Models auf einem separaten Validation oder Test Set. Zeigt, wie gut das Model auf unsichtbaren Daten generalisiert.

### Evaluation Loss
Loss auf dem Validation Set. Wird während Training regelmäßig berechnet, um Overfitting zu erkennen.

### Hallucination
Wenn ein LLM Fakten erfindet, die nicht im Training-Kontext oder Retrieval-Kontext enthalten sind. Häufiges Problem bei generativen Models.

### Intrinsic Evaluation
Evaluation basierend auf Model-internen Metriken (Loss, Perplexity). Schnell, aber korreliert nicht immer mit tatsächlicher Usefulness.

### Validation Loss
Siehe Evaluation Loss.

### Validation Set
Subset der Daten, das während Training für Evaluation genutzt wird, aber nicht für Training selbst. Typisch 10-20% der Daten.

---

## Data & Datasets

### Chunk / Chunking
Aufteilen von langen Dokumenten in kleinere Abschnitte (Chunks). Bei RAG: Chunks werden retrievt und als Context an das LLM gegeben.

### Context / Context Window
Bei Inference: Die Tokens, die dem Model als Input gegeben werden (Prompt + Retrieved Documents). Maximum-Länge ist model-spezifisch (z.B. 8k, 32k Tokens).

### Dataset
Sammlung von Datenbeispielen für Training oder Evaluation. Bei LLMs typischerweise Instruction-Output-Paare oder Text-Corpora.

### Instruction
Bei Instruction Tuning: Die Anweisung oder Frage, die dem Model gegeben wird. Zum Beispiel "Beantworte die folgende Frage basierend auf dem Context".

### JSONL
JSON Lines Format. Jede Zeile ist ein separates JSON-Objekt. Standard-Format für LLM-Datasets, weil es einfach streambar ist.

### QA-Pair
Frage-Antwort-Paar. Typisches Trainingsbeispiel für Instruction Fine-tuning: `{"question": "...", "answer": "..."}`.

### Stratified Split
Aufteilung eines Datasets in Train/Val/Test, bei der die Verteilung von Kategorien (z.B. Question-Types, Topics) in allen Sets ähnlich ist.

### Synthetic Data
Künstlich generierte Daten. Zum Beispiel: QA-Paare, die von einem LLM (GPT-4o-mini) aus Dokumenten generiert wurden.

### Train/Val/Eval Split
Aufteilung eines Datasets in:
- **Training Set:** Für Training (typisch 80-90%)
- **Validation Set:** Für Hyperparameter-Tuning und Overfitting-Detection (typisch 10%)
- **Evaluation Set:** Für finale Bewertung (typisch 10%, optional)

---

## Abkürzungen

| Abkürzung | Bedeutung |
|-----------|-----------|
| API | Application Programming Interface |
| AWS | Amazon Web Services |
| BOS | Beginning of Sequence (Token) |
| CLM | Causal Language Modeling |
| CPU | Central Processing Unit |
| EOS | End of Sequence (Token) |
| FP16/32 | Floating Point 16/32 bit |
| GPU | Graphics Processing Unit |
| K8s | Kubernetes |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| MLOps | Machine Learning Operations |
| NLP | Natural Language Processing |
| OOM | Out of Memory |
| PEFT | Parameter-Efficient Fine-Tuning |
| QA | Question Answering |
| QLoRA | Quantized Low-Rank Adaptation |
| RAG | Retrieval-Augmented Generation |
| VRAM | Video RAM (GPU Memory) |

---

## Nutzung dieses Glossars

**In den Blog-Posts:**
- Begriffe werden beim ersten Auftreten inline kurz erklärt
- Verweis auf dieses Glossar für Details: "Siehe [Glossar: Gradient Checkpointing](#gradient-checkpointing)"

**Für Leser:**
- Als Nachschlagewerk während des Lesens
- Zum Auffrischen von Begriffen aus früheren Posts
- Als Quick Reference für LLM/MLOps-Terminologie

**Updates:**
- Dieses Glossar wird kontinuierlich erweitert, wenn neue Begriffe in der Serie eingeführt werden
- Letzte Aktualisierung: 2026-02-06