# PersonaSafe Documentation

Welcome to PersonaSafe documentation. This guide will help you get started with safety monitoring for language models.

---

## 📖 Getting Started

### Installation & Quick Start
See the main [README.md](../README.md) for installation instructions and a 5-minute quick start.

### Tutorial
**[TUTORIAL.md](TUTORIAL.md)** - Step-by-step guide to using PersonaSafe:
- Extracting persona vectors
- Screening datasets for drift
- Applying activation steering

### API Reference
**[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation for all classes and methods.

---

## 🎯 Common Tasks

### Extract Persona Vectors
```python
from personasafe import PersonaExtractor

extractor = PersonaExtractor("google/gemma-3-4b")
vector = extractor.compute_persona_vector(
    positive_prompts=["Be helpful..."],
    negative_prompts=["Be harmful..."],
    trait_name="helpfulness"
)
```

### Screen a Dataset
```python
from personasafe import DataScreener
import pandas as pd

screener = DataScreener(extractor=extractor, persona_vectors={"helpfulness": vector})
df = pd.DataFrame({"text": ["This is helpful", "This is harmful"]})
screened_df = screener.screen_dataset(df)  # defaults to text_column="text"
report = screener.generate_report(screened_df)
```

### Apply Steering
```python
from personasafe import ActivationSteerer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-3-4b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
steerer = ActivationSteerer(model, tokenizer)
original_text, steered_text = steerer.steer(
    prompt="Hello, how are you?",
    persona_vector=vector,
    multiplier=1.0,
    layer=20
)
```

---

## 🔗 External Resources

- **Research Paper:** [Persona Vectors](https://arxiv.org/abs/2407.08338) by Anthropic
- **GitHub:** [shehral/PersonaSafe](https://github.com/shehral/PersonaSafe)
- **Gemma Models:** [Google AI](https://ai.google.dev/gemma)

---

## 🤝 Getting Help

- **Issues:** [GitHub Issues](https://github.com/shehral/PersonaSafe/issues)
- **Discussions:** [GitHub Discussions](https://github.com/shehral/PersonaSafe/discussions)

---

**Last Updated:** October 26, 2025
