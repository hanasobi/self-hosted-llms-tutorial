# Tests ausführen – Evaluation & Load Test

Beide Scripts laufen **im Cluster**, um den vLLM-Service direkt über die Cluster-interne URL (`http://vllm-service.ml-models:8000`) zu erreichen. Das vermeidet Port-Forward-Overhead und liefert realistische Latenzen.

**Voraussetzung:** vLLM-Pod läuft und ist bereit:

```bash
curl -i http://localhost:8000/health  # via Port-Forward
# HTTP/1.1 200 OK
```

---

## 1. RAG-QA Evaluation (Base Model vs. LoRA Adapter)

Testet 15 ungesehene AWS-Zertifizierungsfragen gegen Base Model und LoRA-Adapter. Dauer: ~2-3 Minuten.

### Dateien

- `scripts/rag_qa_evaluation.py` – Evaluation-Script
- `eval/eval_15_samples.jsonl` – 15 Testfälle (3 Fragetypen, 11 AWS Services)

### Ausführung

```bash
# 1. Script und Testdaten als ConfigMaps bereitstellen
kubectl create configmap eval-script -n ml-models \
  --from-file=scripts/rag_qa_evaluation.py \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap eval-data -n ml-models \
  --from-file=eval/eval_15_samples.jsonl \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Evaluation-Pod starten
kubectl run eval-test \
  --image=python:3.11-slim \
  --restart=Never \
  -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "eval-test",
        "image": "python:3.11-slim",
        "command": ["/bin/bash", "-c"],
        "args": ["pip install requests && python /scripts/rag_qa_evaluation.py /data/eval_15_samples.jsonl /tmp/evaluation_results.json && sleep 300"],
        "volumeMounts": [
          {"name": "script", "mountPath": "/scripts"},
          {"name": "data", "mountPath": "/data"}
        ]
      }],
      "volumes": [
        {"name": "script", "configMap": {"name": "eval-script"}},
        {"name": "data", "configMap": {"name": "eval-data"}}
      ]
    }
  }'

# 3. Logs live verfolgen
kubectl logs -n ml-models eval-test -f

# 4. Ergebnisse rauskopieren (Pod bleibt 5 Min aktiv dafür)
kubectl cp ml-models/eval-test:/tmp/evaluation_results.json ./eval/evaluation_results.json

# 5. Aufräumen
kubectl delete pod eval-test -n ml-models
kubectl delete configmap eval-script eval-data -n ml-models
```

### Erwartete Ergebnisse

| Metrik | Base Model | LoRA Adapter |
|--------|-----------|--------------|
| Korrekte Antworten | ~40% | ~93% |
| Durchschnittliche Latenz | ~2.6s | ~1.0s |
| Leere Antworten | ~27% | 0% |

---

## 2. Load Test (14 Minuten, 5 Phasen)

Simuliert realistischen Traffic mit verschiedenen Lastmustern. Perfekt um Grafana-Dashboard-Screenshots mit aktiven Metriken zu erstellen.

### Dateien

- `scripts/vllm_load_test.py` – Load-Test-Script

### Testphasen

| Phase | Dauer | RPS | Concurrency |
|-------|-------|-----|-------------|
| Warmup | 2 Min | 2 | 3 |
| Normal | 4 Min | 5 | 8 |
| Spike | 3 Min | 12 | 20 |
| Recovery | 3 Min | 5 | 8 |
| Cooldown | 2 Min | 2 | 3 |

### Ausführung

```bash
# 1. Script als ConfigMap bereitstellen
kubectl create configmap load-test-script -n ml-models \
  --from-file=scripts/vllm_load_test.py \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Load-Test-Pod starten
kubectl run load-test \
  --image=python:3.11-slim \
  --restart=Never \
  -n ml-models \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "load-test",
        "image": "python:3.11-slim",
        "command": ["/bin/bash", "-c"],
        "args": ["pip install aiohttp && python /scripts/vllm_load_test.py"],
        "volumeMounts": [{
          "name": "script",
          "mountPath": "/scripts"
        }]
      }],
      "volumes": [{
        "name": "script",
        "configMap": {"name": "load-test-script"}
      }]
    }
  }'

# 3. Logs live verfolgen
kubectl logs -n ml-models load-test -f

# 4. Aufräumen
kubectl delete pod load-test -n ml-models
kubectl delete configmap load-test-script -n ml-models
```

### Was du während des Tests machen solltest

1. **Grafana Dashboard öffnen** (Port-Forward zu Grafana, dann `http://localhost:3000`)
2. **Zeitbereich einstellen**: "Last 15 minutes" oder "Last 30 minutes"
3. **Auto-Refresh aktivieren**: 5s oder 10s
4. **Screenshots machen**: Während der Spike-Phase und am Ende für die Gesamtübersicht

### Erwartete Metriken (Spike-Phase)

| Metrik | Wert |
|--------|------|
| TTFT P99 | ~78ms |
| Requests/s | ~12.5 |
| Laufende Requests | ~20 |
| Wartende Requests | 0 |
| KV-Cache Auslastung | ~1.25% |
| Token Throughput | ~800 tokens/s (generiert) |

---

## Troubleshooting

**Pod startet nicht:**
```bash
kubectl describe pod <pod-name> -n ml-models
```

**Keine Verbindung zu vLLM:**
```bash
kubectl run test-curl --image=curlimages/curl --rm -it -n ml-models -- \
  curl http://vllm-service:8000/health
```

**Script-Fehler:**
```bash
kubectl logs <pod-name> -n ml-models
```

## Alternative: Lokale Ausführung mit Port-Forward

Falls du die Scripts lokal ausführen willst:

```bash
# Port-Forward zu vLLM
kubectl port-forward -n ml-models svc/vllm-service 8000:8000

# URL in den Scripts anpassen:
# http://vllm-service.ml-models:8000  →  http://localhost:8000
```

**Hinweis:** Port-Forward fügt zusätzliche Latenz hinzu. Die gemessenen Werte sind dann nicht direkt vergleichbar mit Cluster-interner Ausführung.