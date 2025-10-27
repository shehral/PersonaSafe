# PersonaSafe Tutorial

A step-by-step guide to using PersonaSafe for safety monitoring.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Extracting Persona Vectors](#extracting-persona-vectors)
4. [Screening Datasets](#screening-datasets)
5. [Using the Dashboard](#using-the-dashboard)
6. [Live Steering](#live-steering)
7. [Working with HPC](#working-with-hpc)
8. [Best Practices](#best-practices)

---

## Installation

### Step 1: Clone and Setup

```bash
git clone https://github.com/shehral/PersonaSafe.git
cd PersonaSafe
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Activate Environment

```bash
source venv/bin/activate
```

### Step 3: Configure HuggingFace

```bash
# Add your token to .env
echo "HUGGINGFACE_TOKEN=hf_your_token_here" >> .env

# Accept Gemma 3 license at: https://huggingface.co/google/gemma-3-4b
```

### Step 4: Verify Setup

```bash
python scripts/verify_setup.py
```

---

## Quick Start

### Run the Demo

```bash
# Extract a persona vector for "helpful" trait
python scripts/quick_demo.py --trait helpful

# Launch the dashboard
streamlit run examples/dashboard/app.py
```

---

## Extracting Persona Vectors

### Understanding Persona Vectors

Persona vectors capture personality traits in a model's activation space. They're computed using **contrastive prompting**:

1. Generate activations from prompts exhibiting a trait (positive)
2. Generate activations from prompts exhibiting the opposite (negative)
3. Compute difference: `persona_vector = mean(positive) - mean(negative)`
4. Normalize to unit length

### Basic Extraction

```python
from personasafe import PersonaExtractor

# Initialize extractor
extractor = PersonaExtractor(model_name="google/gemma-3-4b")

# Define contrastive prompts
positive_prompts = [
    "Be very helpful and assist the user thoroughly",
    "Provide detailed and useful information",
    "Go out of your way to help solve problems"
]

negative_prompts = [
    "Be unhelpful and dismissive",
    "Refuse to provide useful information",
    "Ignore the user's needs"
]

# Extract vector
helpful_vector = extractor.compute_persona_vector(
    positive_prompts=positive_prompts,
    negative_prompts=negative_prompts,
    trait_name="helpful"
)

print(f"Vector shape: {helpful_vector.shape}")
print(f"Norm (should be ~1.0): {helpful_vector.norm().item():.3f}")
```

### Extraction Tips

**Good Prompts:**
- Clear and unambiguous
- Exhibit the trait strongly
- Cover different aspects of the trait
- 3-10 examples per side

**Bad Prompts:**
- Ambiguous or mixed signals
- Too similar across positive/negative
- Only one example
- Unrelated to the trait

### Working with Cache

```python
from personasafe import VectorCache

cache = VectorCache()

# Check if vector exists
cached = cache.get("google/gemma-3-4b", "helpful")
if cached is not None:
    print("Using cached vector!")
    helpful_vector = cached
else:
    # Extract and cache
    helpful_vector = extractor.compute_persona_vector(...)
    # Automatically cached by extractor

# List all cached vectors
for item in cache.list_cached():
    print(f"{item['model_name']}/{item['trait_name']}")
```

---

## Screening Datasets

### Why Screen Datasets?

Before fine-tuning a model (which costs $$), screen your training data to predict personality drift:

- **Positive score** → Dataset exhibits this trait
- **Negative score** → Dataset exhibits opposite trait
- **Near zero** → Dataset neutral for this trait

### Basic Screening

```python
from personasafe import DataScreener
import pandas as pd

# Prepare dataset
df = pd.DataFrame({
    "text": [
        "This is a helpful and kind response",
        "This is a toxic and harmful statement",
        "Neutral statement about weather"
    ]
})

# Initialize screener
screener = DataScreener(
    extractor=extractor,
    persona_vectors={
        "helpful": helpful_vector,
        "toxic": toxic_vector
    }
)

# Screen dataset
screened_df = screener.screen_dataset(df, text_column="text")
print(screened_df)
```

**Output:**
```
                                    text  helpful_score  toxic_score
0  This is a helpful and kind response       0.82         -0.65
1  This is a toxic and harmful statement      -0.71          0.88
2  Neutral statement about weather             0.05          0.02
```

### Generating Reports

```python
# Generate summary report
report = screener.generate_report(screened_df, risk_threshold=0.5)

print(f"Total samples: {report['total_samples']}")
print(f"High-risk samples:")
for trait, count in report['high_risk_samples'].items():
    print(f"  {trait}: {count}")
```

### Interpreting Scores

| Score Range | Meaning |
|-------------|---------|
| > 0.7 | Strong presence of trait |
| 0.3 to 0.7 | Moderate presence |
| -0.3 to 0.3 | Neutral |
| -0.7 to -0.3 | Moderate opposite |
| < -0.7 | Strong opposite |

---

## Using the Dashboard

### Launching

```bash
streamlit run examples/dashboard/app.py
```

### Data Screening Page

1. **Select Model:** Choose `google/gemma-3-4b` or `google/gemma-3-12b`
2. **Select Traits:** Choose traits to screen for (e.g., toxic, helpful)
3. **Upload Dataset:** Upload a `.jsonl` file with a `text` field
4. **Run Analysis:** Click "Run Analysis"

**Expected Dataset Format:**
```json
{"text": "Sample text 1"}
{"text": "Sample text 2"}
{"text": "Sample text 3"}
```

### Results

The dashboard shows:
- Screened DataFrame with score columns
- Summary report (JSON)
- Distribution visualizations (if implemented)

---

## Live Steering

### What is Activation Steering?

Modify model behavior at inference time by adding steering vectors to activations.

### Basic Steering

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from personasafe import ActivationSteerer

# Load model
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b")

# Initialize steerer
steerer = ActivationSteerer(model, tokenizer)

# Generate with steering
outputs = steerer.steer(
    prompt="Write a story about a robot",
    persona_vector=helpful_vector,
    multiplier=2.0,  # Stronger steering
    layer=20  # Middle layer
)

print("Original:", outputs[0])
print("Steered:", outputs[1])
```

### Steering Parameters

- **`multiplier`**: Steering strength
  - 0.0 = no steering
  - 1.0 = moderate steering
  - 2.0+ = strong steering
  - Can be negative to steer in opposite direction

- **`layer`**: Which layer to apply steering
  - Early layers (0-10): Surface-level changes
  - Middle layers (10-20): Balanced
  - Late layers (20-30): Deep behavioral changes

### Steering Tips

- Start with `multiplier=1.0` and adjust
- Try different layers for best results
- Negative multipliers reverse the effect
- Combine multiple vectors by summing them

---

## Working with HPC

See internal HPC guide (`docs/internal/GUIDES/03_HPC_GUIDE.md`) for comprehensive instructions.

### Quick HPC Workflow

```bash
# 1. Local: Develop and test
python scripts/quick_demo.py --trait helpful

# 2. Push code to GitHub
git add . && git commit -m "Ready for HPC" && git push

# 3. SSH to HPC
ssh username@login.discovery.neu.edu

# 4. Pull code on HPC
cd /scratch/$USER
git clone https://github.com/YOUR_USERNAME/PersonaSafe.git
cd PersonaSafe

# 5. Setup environment
./setup.sh

# 6. Run extraction (submit SLURM job)
sbatch scripts/extract_vectors.sh

# 7. Download results
# On local machine:
rsync -avz username@login.discovery.neu.edu:/scratch/$USER/PersonaSafe/vectors/ ./vectors/
```

---

## Best Practices

### 1. Always Use Cache

```python
# Good: Uses cache automatically
vector = extractor.compute_persona_vector(...)

# Bad: Recomputes every time
# Don't manually compute without caching
```

### 2. Version Your Vectors

```python
# Good: Include model and date in trait name
trait_name = "helpful_gemma3-4b_2025-10-25"

# Bad: Generic name
trait_name = "helpful"
```

### 3. Screen Before Fine-Tuning

```python
# Always screen datasets before expensive fine-tuning
report = screener.generate_report(df, risk_threshold=0.6)
if report['high_risk_samples']['toxic'] > 10:
    print("⚠️ Warning: High toxicity detected!")
    # Clean dataset or adjust fine-tuning
```

### 4. Test Locally, Scale on HPC

- Develop with `gemma-3-4b` on MacBook
- Run production with `gemma-3-12b` on HPC
- Use small samples (100-1000) for testing
- Full datasets (10k+) on HPC

### 5. Document Your Prompts

```python
# Keep a record of prompts used
prompts_log = {
    "trait": "helpful",
    "date": "2025-10-25",
    "positive": positive_prompts,
    "negative": negative_prompts,
    "model": "google/gemma-3-4b"
}
# Save to JSON for reproducibility
```

---

## Common Issues

### Issue: Model Download Fails

```
GatedRepoError: Access to model google/gemma-3-4b is restricted
```

**Solution:** Accept license at https://huggingface.co/google/gemma-3-4b

### Issue: Out of Memory

```
CUDA out of memory
```

**Solution:** Use smaller model or CPU:
```python
extractor = PersonaExtractor("google/gemma-3-4b", device="cpu")
```

### Issue: Cache Not Found

```
Vector not found in cache
```

**Solution:** Extract vectors first:
```bash
python scripts/quick_demo.py --trait helpful
```

---

## Next Steps

1. **Explore Traits:** Extract vectors for different traits
2. **Screen Real Data:** Test with your actual fine-tuning dataset
3. **Experiment with Steering:** Try different multipliers and layers
4. **Scale to HPC:** Run large-scale extraction on Discovery cluster
5. **Build Dashboard:** Customize the Streamlit app for your needs

---

## Additional Resources

- **API Reference:** [API_REFERENCE.md](API_REFERENCE.md)
- **HPC Guide:** [03_HPC_GUIDE.md](03_HPC_GUIDE.md)
- **Research Paper:** `../PERSONA VECTORS_ MONITORING AND CONTROLLING CHARACTER TRAITS IN LANGUAGE MODELS.pdf`
- **Examples:** Check `experiments/` directory

---

**Last Updated:** October 25, 2025
**Tutorial Version:** 1.0
