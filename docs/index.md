

> **Ein deutschsprachiges Tutorial, das Schritt für Schritt zeigt, wie Unternehmen mit Self-Hosted LLMs ihre Datensouveränität bewahren: Von der Installation über 
Fine-tuning bis zur vollständigen Unabhängigkeit – inklusive echter Debugging-Stories und transparenten Trade-offs.**

<div style="background: #f6f8fa; padding: 20px; border-radius: 6px; margin: 20px 0;">
  <strong>📖 Status:</strong> Tutorial-Serie in aktiver Entwicklung<br>
  <strong>🎯 Zielgruppe:</strong> ML Engineers, Data Scientists, Tech Leads im DACH-Raum<br>
  <strong>⭐ GitHub:</strong> <a href="https://github.com/hanasobi/self-hosted-llms-tutorial">self-hosted-llms-tutorial</a>
</div>

---

## Warum diese Tutorial-Serie?

Unternehmen im DACH-Raum stehen vor einem Dilemma: Sie wollen generative KI nutzen, aber sensible Daten dürfen nicht an externe APIs fließen — sei es aus DSGVO-Gründen, Branchenregulierung oder zum Schutz von Betriebsgeheimnissen.

Diese Tutorial-Serie zeigt den vollständigen Weg von der ersten LLM-Installation bis zur **kompletten Datensouveränität** — ohne externe Abhängigkeiten. Jeder Post hat ein klares, erreichbares Ziel, und wir dokumentieren echte Probleme und Debugging-Journeys statt nur den "Happy Path".

## Was diese Serie auszeichnet

**Schrittweiser Aufbau statt Fertiglösungen**  
Jeder Schritt wird erklärt und begründet. Statt YAML-Dateien zum Copy-Paste erhältst du das Verständnis, um eigene Entscheidungen zu treffen.

**Design-Entscheidungen transparent gemacht**  
Wir zeigen nicht nur *wie*, sondern auch *warum*. Jede Architektur-Entscheidung wird mit ihren Trade-offs erklärt.

**Debugging-Journeys inklusive**  
Echte Probleme und ihre Lösungen – wie die 20-stündige EOS-Token-Debugging-Story. Hier lernst du, was Tutorials normalerweise auslassen.

**Vollständige Datensouveränität als Ziel**  
Der komplette Weg zur Unabhängigkeit von externen APIs – von der ersten Installation bis zur selbst gehosteten Dataset-Generierung.

---

## Der Weg zur Datensouveränität

Die Serie folgt einem klaren didaktischen Bogen — vom ersten funktionierenden LLM bis zur vollständigen Unabhängigkeit von externen Anbietern.

### Phase 1: Self-Hosting Basics

> *"Kann ich ein LLM überhaupt selbst betreiben?"*

**Post 1: [Warum Self-Hosting? Der Business Case für Datensouveränität](posts/01-warum-self-hosting.html)**
Das Problem, die Lösung und wann Self-Hosting sinnvoll ist. Entscheidungsmatrix: Cloud-API vs. Self-Hosted.

**Post 2: [vLLM auf Kubernetes — Dein erstes selbst gehostetes LLM](posts/02-vllm-kubernetes-basics.html)**
Mistral-7B auf Kubernetes deployen mit vLLM. Nach diesem Post läuft ein LLM auf deiner Infrastruktur.

### Phase 2: Anpassung durch Fine-tuning

> *"Wie mache ich es besser für meinen Use Case?"*

**Post 3: [Warum Fine-tuning? Wenn RAG und Prompting nicht reichen](posts/03-warum-finetuning.html)**
Prompting vs. RAG vs. Fine-tuning — wann welcher Ansatz passt und warum wir Fine-tuning brauchen.

**Post 4: [Dataset Engineering — Von Dokumenten zu Trainingsdaten](posts/04-dataset-engineering.html)**
Die Pipeline von Rohdokumenten zu QA-Paaren: Chunking, Synthetic Data Generation, Quality Control. *80% der eigentlichen Arbeit.*

**Post 5: [LoRA Training — 7B Model auf 24GB GPU](posts/05-lora-training.html)**
QLoRA macht große Modelle auf Consumer-Hardware trainierbar. Mit MLflow Experiment Tracking.

**Post 5.1: [Experiment Tracking mit MLflow (Optional)](posts/05.1-mlflow-tracking.html)**
Self-hosted MLflow für Datensouveränität. Custom Callbacks für HuggingFace Trainer. Parameters & Metrics loggen – ohne externe Cloud-Dienste.

**Post 5.2: [Model Evaluation (Optional)](posts/05.2-model-evaluation.html)**
Qualitative Evaluation durch Manual Inspection & stratifiziertes Sampling. Baseline Comparison mit Mistral-Instruct. Multi-modale Bewertung.

**Post 5.3: Der pad_token Bug – Eine Debugging-Geschichte (Optional)**
20 Stunden Debugging dokumentiert: Warum `pad_token = eos_token` alles kaputt macht und wie systematisches Debugging funktioniert.

### Phase 3: Production & Souveränität

> *"Wie bringe ich es in Produktion — ohne externe Abhängigkeiten?"*

**Post 6: vLLM Deployment mit LoRA – Fine-tuned Models in Produktion**
LoRA-Adapter auf dem Base Model laden mit vLLM. Multi-Adapter Serving. Performance-Vergleiche.

**Post 7: Evaluation ohne externe APIs — LLM-as-Judge Self-Hosted**
Qualität messen ohne OpenAI oder Anthropic. Self-hosted LLM-as-Judge mit Rubrics und Consistency Checks.

**Post 8: Dataset-Generierung ohne OpenAI**
Die letzte externe Abhängigkeit eliminieren. Nach diesem Post ist die gesamte Pipeline self-hosted: Dokumente → QA-Paare → Training → Serving → Evaluation.

### Phase 4: Skalierung & Automation

> *"Wie skaliere ich das Ganze?"*

**Post 9: Multi-LoRA in der Praxis — Ein Server, viele Use Cases**
Architektur für Multi-Tenant-Setups, Request Routing und Kostenoptimierung.

**Post 10+: Production Pipelines**
Argo Workflows, CI/CD für Model Updates, kontinuierliches Fine-tuning.

---

## Datensouveränität als roter Faden

<div style="background: #e8f5e9; padding: 20px; border-left: 4px solid #4caf50; margin: 20px 0;">

<strong>🔒 Von pragmatisch zu souverän</strong><br><br>

Die Serie geht ehrlich mit externen Abhängigkeiten um. In <strong>Post 4</strong> nutzen wir GPT-4o-mini für die Dataset-Generierung — ein bewusster Kompromiss, der transparent gemacht wird. In <strong>Post 9</strong> zeigen wir dann die self-hosted Alternative.<br><br>

<strong>Nach Post 8 ist die gesamte Pipeline datensouverän:</strong> Kein API-Call verlässt deine Infrastruktur — weder für Training, Serving, Evaluation noch für Dataset-Generierung.

</div>

---

## Für wen ist diese Serie?

Diese Tutorial-Serie richtet sich an technische Fachkräfte und Entscheider, die Self-Hosted LLMs evaluieren oder implementieren wollen:

- **ML Engineers & Data Scientists**, die den Schritt von Notebooks zu Production-Deployments machen wollen
- **Tech Leads & Architekten**, die einen Self-Hosted AI-Stack evaluieren und Trade-offs verstehen müssen
- **Technische Entscheider (CTO, Head of Data)**, die Machbarkeit und Aufwand für Datensouveränität einschätzen wollen
- **Implementierungspartner (Freelancer, Agenturen)**, die eine Referenzimplementierung für Kundenprojekte suchen

---

## Projekt-Struktur

```
self-hosted-llms-tutorial/
├── docs/                  Blog Posts (Deutsch)
│   ├── index.md           Serien-Übersicht (diese Seite)
│   └── posts/             Einzelne Blog Posts
├── serving/               vLLM Deployment (Posts 2, 6)
├── data/                  Dataset Engineering (Post 4)
├── 05-lora-training/      LoRA Training & Evaluation (Posts 5, 5.1, 5.2, 5.3)
├── 05.1-mlflow-tracking/  LoRA Training & Evaluation (Posts 5, 5.1, 5.2, 5.3)
├── 05.2-model-evaluation/ LoRA Training & Evaluation (Posts 5, 5.1, 5.2, 5.3)
├── 05.3-debugging-story/  LoRA Training & Evaluation (Posts 5, 5.1, 5.2, 5.3)
└── .../                   Weitere Posts
```

**Sprache:** Blog Posts auf Deutsch, Code und technische Dokumentation auf Englisch.

---

## Mitmachen & Folgen

- **GitHub Repository:** [self-hosted-llms-tutorial](https://github.com/hanasobi/self-hosted-llms-tutorial)
- **Autor:** [@hanasobi](https://github.com/hanasobi)
- **Gestartet:** Januar 2026

<div style="background: #fffbdd; padding: 15px; border-left: 4px solid #f9c513; margin: 20px 0;">
  <strong>⚠️ Hinweis:</strong> Dieses Projekt ist in aktiver Entwicklung. Posts und Code werden regelmäßig ergänzt. Star das Repo, um auf dem Laufenden zu bleiben!
</div>

---

## Lizenz

- **Code:** MIT License — frei nutzbar, modifizierbar und verteilbar
- **Blog Content:** CC BY 4.0 — mit Namensnennung