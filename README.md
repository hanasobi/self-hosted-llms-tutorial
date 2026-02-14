# Self-Hosted LLMs for Data Sovereignty

> From first installation to complete independence: Self-hosted LLM infrastructure with vLLM, fine-tuning, and self-hosted evaluation — without external API dependencies.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Blog: German](https://img.shields.io/badge/Blog-German-blue.svg)](#tutorial-blog-posts)
[![Code: English](https://img.shields.io/badge/Code-English-green.svg)](#project-structure)

🎯 **Goal:** Complete self-hosted LLM pipeline — from first deployment to full data sovereignty  
📖 **Blog Language:** German (targeting DACH market)  
💻 **Code Language:** English (universal readability)  
🔒 **Theme:** Data Sovereignty through Self-Hosted LLMs

---

## Why This Tutorial?

**The Problem:** Companies want to use generative AI but face constraints:
- Sensitive data cannot be sent to external APIs (OpenAI, Anthropic, Google)
- Compliance requirements (GDPR, industry regulations)
- Trade secrets and intellectual property protection
- Loss of control over data and model behavior

**The Solution:** This tutorial shows how to build a completely self-hosted LLM stack — step by step, from your first deployed model to full independence from external services.

**What makes this tutorial different:**

| Other Tutorials | This Tutorial |
|----------------|---------------|
| "Deploy this YAML and you're done" | Step-by-step journey with real debugging stories |
| Jump straight to fine-tuning | First show that self-hosting works, then improve |
| Copy-paste code without explanation | Every decision explained with trade-offs |
| Cloud/API dependent | **Complete data sovereignty** |
| Single aspect coverage | End-to-end: Hosting → Training → Serving → Evaluation |

**The learning path:**

```
Phase 1: "Can I even run an LLM myself?"
    → Posts 1-2: Business case + first working LLM on your infrastructure
    
Phase 2: "How do I make it better for my use case?"
    → Posts 3-5 (incl. 5.1, 5.2, 5.3): Dataset engineering → Training → Tracking → Evaluation → Debugging
    
Phase 3: "How do I run this in production — without external dependencies?"
    → Posts 6-8: LoRA serving, self-hosted dataset generation, self-hosted evaluation
    
Phase 4: "How do I iterate and scale this?"
    → Posts 9+: Multi-LoRA A/B-testing, pipelines, automation
```

After Post 2, you have a **working system**. That's motivating. Then you learn step by step how to improve it and become fully independent.

---

## Tutorial Blog Posts

The blog posts are written in **German**, targeting ML engineers, data scientists, and technical decision-makers in the DACH region (Germany, Austria, Switzerland).

| # | Title | Status | Description |
|---|-------|--------|-------------|
| **Phase 1: Self-Hosting Basics** | | | |
| 1 | Why Self-Hosting? A Business Case für Data Sovereignty | ✅ Done | Why self-host, decision framework, series overview |
| 2 | vLLM on Kubernetes: Your first self-hosted LLM | ✅ Done | Deploy Mistral-7B, K8s basics, first inference |
| **Phase 2: Fine-tuning** | | | |
| 3 | Why Fine-tuning? When RAG and Prompting aren't sufficient | ✅ Done | When and why to fine-tune |
| 4 | Dataset Engineering: From Documents to Training Data | ✅ Done | Chunking, QA generation, quality control |
| 5 | LoRA Training: 7B Model auf 24GB GPU | ✅ Done| QLoRA fine-tuning on consumer hardware |
| 5.1 | Experiment Tracking with MLflow (Optional) | ✅ Done | Self-hosted MLflow |
| 5.2 | Model Evaluation (Optional)| ✅ Done | Qualitative evaluation and baseline comparison |
| 5.3 | The pad_token Bug – A Debugging Story| ✅ Done | 20h debugging journey, community anti-pattern |
| **Phase 3: Production & Sovereignty** | | | |
| 6 | LoRA Serving: Fine-tuned Models in Production | ✅ Done | Adapter loading, Multi-LoRA, performance |
| 7 | Dataset Generation without OpenAI | ✅ Done | Self-hosted synthetic data generation |
| 7.1 | Parallel Dataset Generation (Optional) | ✅ Done | Parallel Dataset Generation – 9× faster with Batching |
| 7.2 | Model Comparison (Optional) | ✅ Done | Quality Comparison Redux – A Fairer Comparison with Llama-3.1-8B |
| 8 | Self-Hosted LLM-as-Judge | 📝 Planned | Evaluate new adapters without external APIs |
| **Phase 4: Iteration & Scaling** | | | |
| 9 | Multi-LoRA & A/B-Testing | 📝 Planned | Iterative adapter improvement, data-driven decisions |
| 10+ | Production Pipelines | 📝 Planned | Argo Workflows, CI/CD, automation |

**Legend:** ✅ Done | 🚧 In Progress | 📝 Planned

---

## What You'll Learn

**Phase 1: Self-Hosting Basics**
- The business case for self-hosted LLMs
- vLLM deployment on Kubernetes
- GPU scheduling and resource management
- First inference with your own infrastructure

**Phase 2: Fine-tuning**
- When fine-tuning beats prompting and RAG
- Dataset engineering from raw documents (no pre-existing datasets)
- LoRA/QLoRA: Training 7B models on 16GB GPUs
- MLflow integration for experiment tracking
- The pad_token bug: Why low loss doesn't mean good model

**Phase 3: Production & Full Sovereignty**
- LoRA adapter serving with vLLM
- Multi-LoRA: One server, multiple specialized adapters
- Self-hosted evaluation (LLM-as-Judge without external APIs)
- Self-hosted dataset generation (no GPT-4 dependency)
- Monitoring with Prometheus and Grafana

**Phase 4: Iteration & Scaling**
- Multi-LoRA serving and A/B-testing
- Data-driven adapter iteration
- Training pipelines with orchestration (Argo Workflows)
- CI/CD for model updates

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Base Model | Mistral-7B-v0.1 | Strong open-source foundation |
| Quantization | AWQ (4-bit) | Efficient inference |
| Fine-tuning | QLoRA (bitsandbytes) | Train on 16GB GPU |
| Training Framework | HuggingFace Transformers | Trainer + custom callbacks |
| Experiment Tracking | MLflow | Metrics, artifacts, comparison |
| Inference | vLLM | High-throughput serving |
| Orchestration | Kubernetes | Production deployment |
| Monitoring | Prometheus + Grafana | Metrics and dashboards |
| GPU | NVIDIA L4 / T4 | Cost-effective inference |

---

## Key Results

From our fine-tuning and deployment:

| Metric | Base Model | Fine-tuned |
|--------|------------|------------|
| Correct Stopping | 40% | **93%** |
| Context Adherence | Sometimes external | **Strict** |
| Response Style | Verbose | **Compact** |
| CUDA Graphs Speedup | - | **2x** |

The fine-tuned model with LoRA adapter achieves 93% success rate compared to 40% for the base model on our RAG-QA evaluation set.

---

## Quick Start

> ⚠️ **Note:** This is not a "deploy in 5 minutes" tutorial. The blog posts explain each step in detail, including infrastructure prerequisites.

### For the Impatient

If you already have a Kubernetes cluster with GPU nodes:

```bash
# Clone repository
git clone https://github.com/hanasobi/self-hosted-llms-tutorial.git
cd self-hosted-llms-tutorial

# Start with Post 2: Deploy base model
kubectl apply -f serving/base-model/

# Port-forward for local access
kubectl port-forward -n ml-models svc/vllm-service 8000:8000

# Test the API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "What is Kubernetes?"}]
  }'
```

For the full journey with all context and decisions explained, start with [Post 1: Warum Self-Hosting?](docs/posts/01-warum-self-hosting.md).

---

## Target Audience

This tutorial is designed for:

**ML Engineers** who want to move beyond "hello world" tutorials to production-ready systems.

**Data Scientists** transitioning to MLOps who need to deploy models to Kubernetes.

**Tech Leads / Architects** evaluating build vs. buy decisions for AI infrastructure.

**Technical Decision Makers** (CTOs, Heads of Data) assessing feasibility of self-hosted LLMs.

**Implementation Partners** (Freelancers, Agencies) looking for reference implementations for client projects.

---

## Data Sovereignty Focus

This tutorial specifically addresses the needs of organizations that cannot or prefer not to send data to external APIs:

- **GDPR Compliance:** All data stays within your infrastructure
- **Industry Regulations:** Suitable for healthcare, finance, legal sectors
- **Intellectual Property:** Training data and model outputs remain private
- **No Vendor Lock-in:** Full control over the entire stack

**The path to complete sovereignty:**

| Post | External Dependency | Self-Hosted Alternative |
|------|--------------------|-----------------------|
| 2 | None | Base model inference |
| 4 | GPT-4o-mini for QA generation | Shown in Post 7 |
| 6 | None | Fine-tuned model inference |
| 7 | OpenAI for dataset creation | Self-hosted generation |
| 8 | GPT-4 as Judge | Self-hosted LLM-as-Judge |

By the end of this tutorial series (Post 8), you'll have **zero external API dependencies**.

---

## Feedback

Found an issue or have a question? Feel free to open a [GitHub Issue](https://github.com/hanasobi/self-hosted-llms-tutorial/issues).

> **Note:** This is a side project maintained in my spare time. I aim to be helpful but response times may vary.

---

## License

- **Code:** MIT License — See [LICENSE](LICENSE)
- **Blog Content:** CC BY 4.0 (Attribution required)

---

## Author

**[hanasobi](https://github.com/hanasobi)**

Building self-hosted AI solutions with focus on data sovereignty for the DACH market.

---

**Started:** January 2026  
**Last Updated:** February 2026