# Post 7.1: Parallele QA-Generierung Setup

Vollständige Anleitung für Concurrency Tests und Production Runs im Kubernetes Cluster.

---

## 🚀 Quick Start - 3 Commands

```bash
# 1. Script hochladen
kubectl create configmap parallel-qa-script \
  --from-file=generate_qa_pairs_parallel.py \
  -n ml-models \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Test-Daten vorbereiten (erste 100 Chunks)
head -100 chunks_all.jsonl > chunks_test_100.jsonl
kubectl create configmap parallel-qa-test-data \
  --from-file=chunks_test_100.jsonl \
  -n ml-models \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Concurrency Test starten
kubectl run parallel-qa-test \
  --image=python:3.11-slim \
  --restart=Never \
  -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "parallel-qa-test",
        "image": "python:3.11-slim",
        "command": ["/bin/bash", "-c"],
        "args": ["pip install aiohttp && python /scripts/generate_qa_pairs_parallel.py --test-mode --input /data/chunks_test_100.jsonl --vllm-url http://vllm-service:8000 --test-levels 1 5 10 20 && sleep 300"],
        "volumeMounts": [
          {"name": "script", "mountPath": "/scripts"},
          {"name": "data", "mountPath": "/data"}
        ]
      }],
      "volumes": [
        {"name": "script", "configMap": {"name": "parallel-qa-script"}},
        {"name": "data", "configMap": {"name": "parallel-qa-test-data"}}
      ]
    }
  }'

# Logs verfolgen
kubectl logs -f parallel-qa-test -n ml-models
```

**Dauer:** 15-20 Minuten | **Ergebnis:** Optimale Concurrency finden

---

## 📖 Detaillierte Anleitung

### Setup-Komponenten

Wir nutzen das bewährte Pattern aus Post 7:
1. **ConfigMap** für Script und Input-Daten
2. **kubectl run** mit `--overrides` für flexible Pod-Konfiguration  
3. **Output nach /tmp** schreiben
4. **kubectl cp** zum Rausholen der Ergebnisse

---

### Schritt 1: Script in ConfigMap speichern

```bash
# ConfigMap erstellen
kubectl create configmap parallel-qa-script \
  --from-file=scripts/generate_qa_pairs_parallel.py \
  -n ml-models \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify
kubectl get configmap parallel-qa-script -n ml-models
kubectl describe configmap parallel-qa-script -n ml-models | head -20
```

**Script Update:** Falls du das Script änderst, einfach Command erneut ausführen (überschreibt mit `--dry-run=client | kubectl apply`).

---

### Schritt 2: Input-Daten vorbereiten

#### Variante A: ConfigMap (< 1MB) - Für Tests

**Test-Sample erstellen:**
```bash
# Option 1: Erste 100 Chunks (schnell)
head -100 data/chunks_all.jsonl > data/chunks_test_100.jsonl

# Option 2: Stratified Sampling (repräsentativer)
python scripts/prepare_test_sample.py \
  --input data/chunks_all.jsonl \
  --output data/chunks_test_100.jsonl \
  --sample-size 100 \
  --stratified
```

**Upload in ConfigMap:**
```bash
kubectl create configmap parallel-qa-test-data \
  --from-file=data/chunks_test_100.jsonl \
  -n ml-models \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify
kubectl get configmap parallel-qa-test-data -n ml-models
```

**ConfigMap Limit:** Max 1MB. Für 100 Chunks sollte das reichen (~200KB).

---

#### Variante B: PersistentVolumeClaim - Für Production (> 1MB)

Falls deine 1932 Chunks > 1MB sind:

**PVC erstellen:**
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qa-generation-data
  namespace: ml-models
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF

# Verify
kubectl get pvc qa-generation-data -n ml-models
```

**Daten zum PVC hochladen:**
```bash
# Temporary Pod zum Upload
kubectl run pvc-uploader --image=busybox --restart=Never -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "uploader",
        "image": "busybox",
        "command": ["sleep", "3600"],
        "volumeMounts": [{"name": "data", "mountPath": "/data"}]
      }],
      "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": "qa-generation-data"}}]
    }
  }'

# Warte bis Pod läuft
kubectl wait --for=condition=Ready pod/pvc-uploader -n ml-models --timeout=60s

# Kopiere Chunks
kubectl cp chunks_all_1932.jsonl ml-models/pvc-uploader:/data/chunks_all.jsonl

# Verify
kubectl exec pvc-uploader -n ml-models -- ls -lh /data/

# Cleanup
kubectl delete pod pvc-uploader -n ml-models
```

---

### Schritt 3: Concurrency Tests (Test Mode)

**Ziel:** Finde die optimale Concurrency für deine Hardware.

```bash
kubectl run parallel-qa-test \
  --image=python:3.11-slim \
  --restart=Never \
  -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "parallel-qa-test",
        "image": "python:3.11-slim",
        "command": ["/bin/bash", "-c"],
        "args": ["pip install aiohttp && python /scripts/generate_qa_pairs_parallel.py --test-mode --input /data/chunks_test_100.jsonl --vllm-url http://vllm-service:8000 --test-levels 1 5 10 20 && sleep 300"],
        "volumeMounts": [
          {"name": "script", "mountPath": "/scripts"},
          {"name": "data", "mountPath": "/data"}
        ]
      }],
      "volumes": [
        {"name": "script", "configMap": {"name": "parallel-qa-script"}},
        {"name": "data", "configMap": {"name": "parallel-qa-test-data"}}
      ]
    }
  }'
```

**Logs live verfolgen:**
```bash
# Live Output
kubectl logs -f parallel-qa-test -n ml-models -f

# Oder in File speichern
kubectl logs parallel-qa-test -n ml-models > test_results.log
```

**Erwartete Output:**
```
==================================================================================
TESTING CONCURRENCY LEVEL: 1
==================================================================================
Duration: 435.2s
Success Rate: 100.0% (100/100)
Throughput: 0.23 chunks/sec

==================================================================================
TESTING CONCURRENCY LEVEL: 5
==================================================================================
Duration: 92.4s
Success Rate: 100.0% (100/100)
Throughput: 1.08 chunks/sec

[...]

==================================================================================
CONCURRENCY TEST SUMMARY
==================================================================================

Concur   Success    Throughput   Latency (P95)   KV-Cache   GPU Util  
-------- ---------- ------------ --------------- ---------- ----------
1        100.0%     0.23/s       8.13s           0.8%       90%       
5        100.0%     1.08/s       8.52s           4.2%       92%       
10       99.0%      1.82/s       9.18s           11.5%      94%       
20       96.5%      2.41/s       11.24s          23.8%      95%       

🎯 RECOMMENDED CONCURRENCY: 10
   - Throughput: 1.82 chunks/sec
   - Success Rate: 99.0%
   - P95 Latency: 9.18s
   - KV-Cache Usage: 11.5%
```

**Test-Dauer:** ~15-20 Minuten (4 Levels × ~4 Min)

---

**Ergebnisse holen:**
```bash
# Vollständige Logs
kubectl logs parallel-qa-test -n ml-models > test_complete.log

# JSON Results (falls vorhanden)
kubectl cp ml-models/parallel-qa-test:/tmp/concurrency_test_results*.json ./test_results.json

# Cleanup
kubectl delete pod parallel-qa-test -n ml-models
```

**Während der Tests: Grafana Dashboard beobachten**
- Dashboard: "vLLM Deep Dive - Parallelization (V2)"
- Time Range: Last 30 minutes
- Auto-Refresh: 10s

**Screenshots nach jedem Level:**
1. Concurrent Requests Gauge (1 → 5 → 10 → 20)
2. KV-Cache Usage Gauge (0.8% → 4% → 11% → 24%)
3. E2E Latency P95 (steigt mit Concurrency)

---

### Schritt 4: Production Run

Nach erfolgreichen Tests mit optimaler Concurrency (z.B. 10) die vollen 1932 Chunks generieren.

#### Variante A: Mit ConfigMap (falls < 1MB)

```bash
# Upload Full Dataset
kubectl create configmap parallel-qa-full-data \
  --from-file=chunks_all_1932.jsonl \
  -n ml-models \
  --dry-run=client -o yaml | kubectl apply -f -

# Production Run
kubectl run parallel-qa-prod \
  --image=python:3.11-slim \
  --restart=Never \
  -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "parallel-qa-prod",
        "image": "python:3.11-slim",
        "command": ["/bin/bash", "-c"],
        "args": ["pip install aiohttp && python /scripts/generate_qa_pairs_parallel.py --concurrency 10 --input /data/chunks_all_1932.jsonl --output /tmp/qa_pairs_parallel.jsonl --vllm-url http://vllm-service:8000 && sleep 300"],
        "volumeMounts": [
          {"name": "script", "mountPath": "/scripts"},
          {"name": "data", "mountPath": "/data"}
        ]
      }],
      "volumes": [
        {"name": "script", "configMap": {"name": "parallel-qa-script"}},
        {"name": "data", "configMap": {"name": "parallel-qa-full-data"}}
      ]
    }
  }'
```

---

#### Variante B: Mit PVC (empfohlen für große Datasets)

```bash
kubectl run parallel-qa-prod \
  --image=python:3.11-slim \
  --restart=Never \
  -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "parallel-qa-prod",
        "image": "python:3.11-slim",
        "command": ["/bin/bash", "-c"],
        "args": ["pip install aiohttp && python /scripts/generate_qa_pairs_parallel.py --concurrency 10 --input /data/chunks_all.jsonl --output /data/qa_pairs_parallel.jsonl --vllm-url http://vllm-service:8000 && sleep 300"],
        "volumeMounts": [
          {"name": "script", "mountPath": "/scripts"},
          {"name": "data", "mountPath": "/data"}
        ]
      }],
      "volumes": [
        {"name": "script", "configMap": {"name": "parallel-qa-script"}},
        {"name": "data", "persistentVolumeClaim": {"claimName": "qa-generation-data"}}
      ]
    }
  }'
```

---

**Logs monitoren:**
```bash
# Live Logs
kubectl logs -f parallel-qa-prod -n ml-models

# Progress alle 30s
watch -n 30 'kubectl logs parallel-qa-prod -n ml-models | tail -20'
```

**Ergebnisse holen:**
```bash
# ConfigMap Variante
kubectl cp ml-models/parallel-qa-prod:/tmp/qa_pairs_parallel.jsonl ./qa_pairs_parallel.jsonl

# PVC Variante
kubectl run pvc-downloader --image=busybox --restart=Never -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "downloader",
        "image": "busybox",
        "command": ["sleep", "3600"],
        "volumeMounts": [{"name": "data", "mountPath": "/data"}]
      }],
      "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": "qa-generation-data"}}]
    }
  }'

kubectl wait --for=condition=Ready pod/pvc-downloader -n ml-models --timeout=60s
kubectl cp ml-models/pvc-downloader:/data/qa_pairs_parallel.jsonl ./qa_pairs_parallel.jsonl
kubectl delete pod pvc-downloader -n ml-models

# Cleanup Production Pod
kubectl delete pod parallel-qa-prod -n ml-models
```

**Production-Dauer:** 20-30 Minuten für 1932 Chunks (bei Concurrency 10)

---

### Schritt 5: Ergebnisse validieren

```bash
# Anzahl QA Pairs prüfen
wc -l qa_pairs_parallel.jsonl
# Erwartet: ~5796 Zeilen (1932 Chunks × 3 QA Pairs)

# Sample anschauen
head -5 qa_pairs_parallel.jsonl | jq

# Success Rate prüfen
echo "Scale: 2; $(wc -l < qa_pairs_parallel.jsonl) / 5796 * 100" | bc
# Sollte ~95-100% sein
```

---

## 📊 Erwartete Ergebnisse

### Test Mode (100 Chunks)

| Concurrency | Duration | Throughput | KV-Cache | Success | P95 Latency |
|-------------|----------|------------|----------|---------|-------------|
| 1           | ~7 min   | 0.23/s     | 0.8%     | 100%    | 8.1s        |
| 5           | ~1.5 min | 1.0/s      | 4%       | 100%    | 8.5s        |
| 10          | ~55s     | 1.8/s      | 11%      | 99%     | 9.2s        |
| 20          | ~42s     | 2.4/s      | 24%      | 96%     | 11.2s       |

**Sweet Spot:** Concurrency 10-15

---

### Production Mode (1932 Chunks)

**Sequential (Post 7 Baseline):**
- Duration: ~2.3 Stunden
- Throughput: 0.23 chunks/sec (13.8 chunks/min)
- KV-Cache: 0.76%

**Parallel (Post 7.1 mit Concurrency 10):**
- Duration: ~20-25 Minuten
- Throughput: 1.5-1.8 chunks/sec (90-108 chunks/min)
- KV-Cache: 10-15%
- **Speedup: 6-7×**

---

## 🔧 Anpassungen & Varianten

### Test Levels ändern

```bash
# Nur bestimmte Levels testen
--test-levels 1 10 20

# Feinere Abstufung
--test-levels 5 10 15 20 25
```

### Concurrency für Production anpassen

```bash
# Nach Tests: Nutze optimalen Wert
--concurrency 15  # statt 10

# Konservativer (bei Problemen)
--concurrency 5
```

### vLLM Service Name

Falls dein Service anders heißt:

```bash
# Service Name finden
kubectl get svc -n ml-models | grep vllm

# URL anpassen
--vllm-url http://DEIN-SERVICE-NAME:8000

# Oder mit FQDN
--vllm-url http://vllm-service.ml-models.svc.cluster.local:8000
```

### Script aktualisieren

```bash
# Script lokal geändert? ConfigMap updaten:
kubectl create configmap parallel-qa-script \
  --from-file=generate_qa_pairs_parallel.py \
  -n ml-models \
  --dry-run=client -o yaml | kubectl apply -f -

# Pod neu starten
kubectl delete pod parallel-qa-test -n ml-models
# ... kubectl run command erneut
```

---

## 🐛 Troubleshooting

### Pod startet nicht / bleibt Pending

```bash
# Status prüfen
kubectl get pod parallel-qa-test -n ml-models

# Details anschauen
kubectl describe pod parallel-qa-test -n ml-models
```

**Häufige Ursachen:**
- ConfigMap nicht gefunden → Namen prüfen
- Insufficient resources → `kubectl describe node`
- Image pull error → Image-Name korrekt?

---

### "pip install" schlägt fehl

```bash
Error: Could not install package aiohttp
```

**Lösung 1:** Anderes Base-Image
```bash
# Nutze python:3.11 statt python:3.11-slim
--image=python:3.11
```

**Lösung 2:** Retry mit besserem Error Handling
```bash
"pip install --no-cache-dir aiohttp || (sleep 5 && pip install aiohttp)"
```

---

### vLLM nicht erreichbar

```bash
Connection refused to http://vllm-service:8000
```

**Check Service:**
```bash
# Existiert der Service?
kubectl get svc vllm-service -n ml-models

# Port korrekt?
kubectl get svc vllm-service -n ml-models -o jsonpath='{.spec.ports[0].port}'

# Service in anderem Namespace?
kubectl get svc -A | grep vllm
```

**Fix URL:**
```bash
# Falls in anderem Namespace:
--vllm-url http://vllm-service.OTHER-NAMESPACE.svc.cluster.local:8000

# Oder anderer Port:
--vllm-url http://vllm-service:8080
```

**Test Connectivity:**
```bash
# DNS Test
kubectl run dns-test --image=busybox --restart=Never -n ml-models \
  -- nslookup vllm-service

# HTTP Test
kubectl run curl-test --image=curlimages/curl --restart=Never -n ml-models \
  -- curl -v http://vllm-service:8000/health

# Cleanup
kubectl delete pod dns-test curl-test -n ml-models
```

---

### Timeouts / Hohe Latenz

```bash
Warning: Request timeout after 30s
```

**Ursachen:**
- Concurrency zu hoch → GPU überlastet
- vLLM Queue voll
- Netzwerk-Latenz

**Lösungen:**
```bash
# 1. Concurrency reduzieren
--concurrency 5  # statt 20

# 2. Timeout erhöhen (im Script, Zeile 124)
timeout: int = 60  # statt 30

# 3. Batch Size reduzieren
--batch-size 50  # statt 100
```

---

### ConfigMap zu groß

```bash
Error: ConfigMap exceeds 1MB limit
```

**Lösung:** Wechsel zu PVC (siehe "Variante B" oben)

---

### Pod bleibt nach Completion

```bash
# Pod läuft noch nach sleep 300
kubectl get pods -n ml-models | grep parallel-qa
```

**Cleanup:**
```bash
# Normal
kubectl delete pod parallel-qa-test -n ml-models

# Force (falls stuck)
kubectl delete pod parallel-qa-test -n ml-models --force --grace-period=0
```

---

### Logs zeigen JSON Parse Errors

```bash
Warning: JSON parse error in chunk_042
```

**Normal bei ~1-5% der Requests:**
- Model generiert manchmal invalides JSON
- Retry Mechanismus fängt das ab
- Bei >10% Fehlerrate → System Prompt prüfen

**Fix (falls häufig):**
```python
# Im Script (Zeile 115):
temperature: 0.3  # statt 0.7 (deterministischer)
```

---

## 📝 Für den Blog Post

### Zu dokumentieren:

**1. Baseline Screenshots (vor Tests)**
- Concurrent Requests = 0 oder 1
- KV-Cache = 0.76%
- GPU Idle oder minimal genutzt

**2. Test Results Tabelle**
- Alle Concurrency Levels
- Success Rate, Throughput, KV-Cache
- Screenshot der Summary-Tabelle aus Logs

**3. Timeline Screenshots (Production Run)**
- KV-Cache Usage Over Time (0.8% → 11%)
- Concurrent Requests Timeline (zeigt ~10)
- GPU Utilization (steigt auf 94%+)

**4. Log Excerpts**
```bash
# Wichtige Abschnitte aus Logs:
- Batch Progress Updates
- Final Statistics
- CONCURRENCY TEST SUMMARY Table
```

**5. Vergleich Sequential vs Parallel**
```
Sequential (Post 7):   2.3 Stunden  | 0.76% KV-Cache
Parallel (Post 7.1):   24 Minuten   | 11% KV-Cache
Speedup:               5.75×
```

---

## ✅ Checkliste

**Vor dem Start:**
- [ ] Script lokal vorhanden (`generate_qa_pairs_parallel.py`)
- [ ] Chunks vorhanden (`chunks_all.jsonl`)
- [ ] vLLM Service läuft (`kubectl get svc vllm-service -n ml-models`)
- [ ] Grafana Dashboard geöffnet

**Test Mode:**
- [ ] Script in ConfigMap uploaded
- [ ] Test-Daten (100 Chunks) in ConfigMap uploaded
- [ ] kubectl run ausgeführt
- [ ] Logs werden verfolgt
- [ ] Screenshots nach jedem Level
- [ ] Ergebnisse kopiert
- [ ] Optimale Concurrency notiert
- [ ] Pod gelöscht

**Production Mode:**
- [ ] Full Dataset uploaded (ConfigMap oder PVC)
- [ ] Optimale Concurrency eingetragen
- [ ] kubectl run ausgeführt
- [ ] Logs werden verfolgt
- [ ] Timeline Screenshots
- [ ] Ergebnisse kopiert
- [ ] Output validiert (Anzahl, Sample check)
- [ ] Pod gelöscht

---

## 🎯 Zusammenfassung

**3-Phasen Workflow:**

```
Phase 1: Setup (5 Min)
  └─ Script & Test-Daten in ConfigMaps

Phase 2: Testing (20 Min)
  └─ Finde optimale Concurrency (Test Mode)

Phase 3: Production (25 Min)
  └─ Generate alle 1932 Chunks
```

**Total:** ~50 Minuten (vs 2.3 Stunden Sequential)

**Ergebnis:**
- 6-7× Speedup
- 10-15% KV-Cache Auslastung (vs 0.8%)
- 95%+ Success Rate
- ~5800 QA Pairs generiert

---

Bereit zum Starten! 🚀

Bei Fragen:
- Service-Name korrekt?
- Chunks-Location?
- Namespace ok?
- Anpassungen nötig?
