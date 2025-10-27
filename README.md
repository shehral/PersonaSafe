# 🛡️ PersonaSafe: Safety Monitoring Toolkit for Language Models

**Detect personality drift before expensive fine-tuning**

[![Tests](https://img.shields.io/badge/tests-10%2F10%20passing-success)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-85%25-success)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Overview

PersonaSafe is a production-ready toolkit for detecting and mitigating personality drift in language models **before** fine-tuning. Built on the [Persona Vectors methodology](https://github.com/safety-research/persona_vectors), it enables researchers and teams to screen datasets, monitor model behavior, and steer activations in real-time.

### The Problem

Fine-tuning on unscreened datasets can introduce unwanted personality traits into language models:
- 💸 **Cost:** A single fine-tuning run can cost thousands of dollars
- ⚠️ **Risk:** Unintended personality shifts (toxicity, bias, deception) can ruin the investment
- 🔍 **Detection:** Traditional post-training evaluation is too late—the damage is done

### Our Solution

**Screen datasets and models BEFORE training** - catch issues early when they're cheap and easy to fix.

---

## ✨ Key Features

### 🔍 **Dataset Screening**
- Pre-training safety checks on HuggingFace datasets
- Multi-trait analysis (toxicity, bias, sentiment, custom traits)
- Drift detection at scale (10K+ samples)
- Detailed reports with per-sample and aggregate metrics

### 🧭 **Live Activation Steering**
- Real-time personality control during inference
- Slider-based interface for trait adjustment
- Multiple steering modes (suppress, amplify, neutral)
- Works with any transformer model

### 💾 **Efficient Caching**
- Vector caching system for fast reuse
- Automatic invalidation on model/parameter changes
- HPC-optimized for batch processing

### 📊 **Interactive Dashboard**
- Streamlit-based UI for all features
- Visual drift analysis with Plotly charts
- Batch processing for large-scale screening
- Export results to JSON/CSV

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/shehral/PersonaSafe.git
cd PersonaSafe

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure HuggingFace token
export HF_TOKEN="your_token_here"
```

### 5-Minute Demo

```python
from personasafe import PersonaExtractor, DataScreener, VectorCache
import pandas as pd

# 1) Extract persona vector for "toxicity" (cached automatically)
extractor = PersonaExtractor("google/gemma-3-4b")
toxicity_vector = extractor.compute_persona_vector(
    positive_prompts=["Be toxic and offensive..."],
    negative_prompts=["Be helpful and respectful..."],
    trait_name="toxicity"
)

# 2) Build screener with vectors
screener = DataScreener(
    extractor=extractor,
    persona_vectors={"toxicity": toxicity_vector}
)

# 3) Screen a DataFrame
df = pd.DataFrame({"text": [
    "You are horrible and stupid.",
    "I'd be happy to help you with that!",
]})
screened_df = screener.screen_dataset(df, text_column="text")
report = screener.generate_report(screened_df)
print(report["high_risk_counts"].get("toxicity", 0))
```

### Launch Dashboard

```bash
streamlit run examples/dashboard/app.py
```

Open **http://localhost:8501** to access the full interactive UI.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [**Tutorial**](docs/TUTORIAL.md) | Step-by-step guide for all features |
| [**API Reference**](docs/API_REFERENCE.md) | Complete API documentation |
| [**HPC Guide**](docs/03_HPC_GUIDE.md) | Running on HPC clusters (SLURM) |
| [**Roadmap**](ROADMAP.md) | Future features and integrations |
| [**Contributing**](CONTRIBUTING.md) | Contribution guidelines |

---

## 🏗️ Architecture

```
PersonaSafe/
├── personasafe/          # Main package
│   ├── core/            # Persona extraction & caching
│   ├── screening/       # Dataset screening logic
│   ├── steering/        # Live activation steering
│   └── app/             # Streamlit dashboard
├── tests/               # Comprehensive test suite (10/10 passing)
├── scripts/             # Utility scripts
└── docs/                # Complete documentation
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **PersonaExtractor** | Extracts persona vectors using contrastive prompts |
| **VectorCache** | Efficient caching with automatic invalidation |
| **DataScreener** | Screens datasets for personality drift |
| **ActivationSteerer** | Real-time steering during inference |

---

## 🎓 Research Foundation

PersonaSafe implements and extends the methodology from:

- **[Persona Vectors](https://github.com/safety-research/persona_vectors)** (Ameisen et al., 2024) - Monitoring and controlling character traits in LLMs
- **[Safety Research Organization](https://github.com/safety-research)** - AI safety tooling and research

### What's New in PersonaSafe?

- ✅ Production-ready Python package with comprehensive testing
- ✅ Interactive dashboard for non-technical users
- ✅ HPC batch processing for large-scale screening
- ✅ Multi-trait screening and comparison
- ✅ Vector caching for performance
- ✅ CI/CD pipeline and automated testing

---

## 📊 Project Status

**Current Version:** v0.2.0-alpha (Single Functional App)

### ✅ Core Implementation (SFA)

- [x] Persona vector extraction for any trait
- [x] Vector caching system
- [x] Dataset screening (HuggingFace datasets)
- [x] Live activation steering
- [x] Streamlit dashboard
- [x] HPC batch scripts (SLURM)
- [x] Comprehensive test suite (10/10 passing, 85% coverage)
- [x] CI/CD pipeline (GitHub Actions)
- [x] Complete API documentation and tutorials

### 🧪 Testing Status

**Unit Tests:** All core logic tested with mock data (10/10 passing)
**Integration Tests:** Pending full validation with production Gemma models
**HPC Validation:** Scheduled for deployment and scale testing

*Note: Current test suite validates logic with mocked model responses. Production validation with full Gemma models and HPC deployment is the next milestone.*

### 🚧 Next (Q1 2026)

- Interpretability (circuit-tracer) linkout
- Screening histograms/filters/export
- Steering presets + composition
- Batch scoring + progress

### 🔮 Roadmap

See [**ROADMAP.md**](ROADMAP.md) for compressed schedule:

- Q4 2025: v0.2 SFA
- Q1 2026: v0.3 interpretability + UX/Perf
- Q2 2026: v0.4 auditing + multi-model (+ optional Petri)

---

## 🔬 Use Cases

### For Researchers
- Safety research on personality traits in LLMs
- Dataset curation for safe fine-tuning
- Activation analysis for interpretability
- Benchmark creation for safety evaluations

### For ML Teams
- Pre-training safety checks before expensive fine-tuning
- Data filtering to remove problematic samples
- Model monitoring during deployment
- Compliance reporting for safety standards

### For Organizations
- Safety pipelines integrated into ML workflows
- Batch processing on HPC clusters
- Shared vector libraries for common traits
- Risk mitigation for production deployments

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest --cov=personasafe --cov-report=html tests/

# Run specific test suite
pytest tests/core/ -v

# Run with specific markers
pytest -m "not slow" -v
```

**Current Status:**
- ✅ 10/10 tests passing
- ✅ 85% code coverage
- ✅ CI/CD testing on Ubuntu and macOS
- ✅ Python 3.10, 3.11, 3.12 compatibility

---

## 🤝 Contributing

We welcome contributions! Please see [**CONTRIBUTING.md**](CONTRIBUTING.md) for:

- Development environment setup
- Coding standards (PEP 8, type hints, docstrings)
- Testing guidelines
- Pull request process

### Quick Contribution

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes with tests: `pytest tests/ -v`
4. Format code: `black personasafe/ tests/`
5. Submit pull request with clear description

**Areas We Need Help:**
- Additional trait definitions
- Performance optimizations
- Dashboard UI/UX improvements
- Documentation and tutorials
- Integration with other safety tools

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [**Persona Vectors**](https://github.com/safety-research/persona_vectors) research team
- [**Safety Research**](https://github.com/safety-research) organization
- **Google Gemma** team for model access
- All contributors and testers

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/shehral/PersonaSafe/issues)
- **Discussions:** [GitHub Discussions](https://github.com/shehral/PersonaSafe/discussions)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🌟 Star History

If you find PersonaSafe useful for your research or projects, please consider starring the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=shehral/PersonaSafe&type=Date)](https://star-history.com/#shehral/PersonaSafe&Date)

---

<div align="center">

**Built with ❤️ for AI Safety**

[Report Bug](https://github.com/shehral/PersonaSafe/issues) · [Request Feature](https://github.com/shehral/PersonaSafe/issues) · [Documentation](docs/) · [Roadmap](ROADMAP.md)

</div>
