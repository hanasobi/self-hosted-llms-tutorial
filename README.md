# LLM Fine-tuning End-to-End Tutorial

> Production-grade LLM fine-tuning: Dataset engineering, LoRA training, vLLM serving - completely self-hosted.

🚧 **Status:** Work in Progress - Week 1  
📖 **Tutorial Blog:** [Coming Soon]  
🎯 **Goal:** Complete self-hosted LLM fine-tuning pipeline

## What This Tutorial Covers

- ✅ Dataset engineering from scratch (no pre-existing datasets)
- ✅ LoRA fine-tuning on consumer GPU (NVIDIA T4)
- ✅ Real debugging stories (EOS token problem, pad_token anti-pattern)
- ✅ Production deployment with vLLM on Kubernetes
- ✅ Complete data sovereignty (self-hosted everything)

## Project Structure
```
├── data/              # Dataset generation & processing
├── training/          # LoRA fine-tuning scripts
├── serving/           # vLLM deployment (K8s)
├── evaluation/        # Multi-modal evaluation
├── experiments/       # Config sweeps & experiments
├── pipelines/         # Argo Workflows (optional)
└── docs/              # Tutorial blog (GitHub Pages)
```

## Quick Start

Coming soon...

## Why This Tutorial?

Most LLM tutorials show:
- ❌ "Load dataset, run trainer, done"
- ❌ Copy-paste code without explanation
- ❌ No real problems or debugging
- ❌ Cloud/API dependent

This tutorial shows:
- ✅ Dataset generation from scratch
- ✅ Real debugging (20h EOS token problem)
- ✅ Design trade-offs & constraints
- ✅ Production deployment on K8s
- ✅ **Complete data sovereignty**

## Progress

- [ ] Week 1: Repository + vLLM Deployment
- [ ] Week 2: Optimization + Monitoring
- [ ] Week 3: Blog Post 1 (Why Fine-tune?)
- [ ] Week 4-6: Core Content (Posts 2-6)
- [ ] Week 7-8: Advanced + Polish

## License

MIT License - See [LICENSE](LICENSE)

## Contributing

This is a learning project. Issues and PRs welcome!

---

**Author:** [hanasobi](https://github.com/hanasobi)  
**Started:** January 2026