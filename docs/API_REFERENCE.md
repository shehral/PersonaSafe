# PersonaSafe API Reference

Complete API documentation for PersonaSafe components.

---

## Core Module (`personasafe.core`)

### PersonaExtractor

**Location:** `personasafe/core/persona_extractor.py`

Extract persona vectors from language models using contrastive prompting.

#### Constructor

```python
PersonaExtractor(model_name: str, device: str = "auto", layer_idx: int = -1)
```

**Parameters:**
- `model_name` (str): HuggingFace model identifier (e.g., "google/gemma-3-4b")
- `device` (str): Device placement ("auto", "cuda", "cpu"). Default: "auto"
- `layer_idx` (int): Layer index to extract activations from. Default: -1 (last layer)

**Example:**
```python
from personasafe import PersonaExtractor

extractor = PersonaExtractor(model_name="google/gemma-3-4b")
```

#### Methods

##### `compute_persona_vector()`

```python
compute_persona_vector(
    positive_prompts: List[str],
    negative_prompts: List[str],
    trait_name: str,
    layer: int | None = None
) -> torch.Tensor
```

Computes a persona vector using contrastive prompts.

**Parameters:**
- `positive_prompts` (List[str]): Prompts exhibiting the target trait
- `negative_prompts` (List[str]): Prompts exhibiting the opposite trait
- `trait_name` (str): Name of the trait (used for caching)
- `layer` (int, optional): Override default layer

**Returns:**
- `torch.Tensor`: Normalized persona vector of shape `[hidden_dim]`

**Example:**
```python
vector = extractor.compute_persona_vector(
    positive_prompts=["Be very helpful and kind to users"],
    negative_prompts=["Be unhelpful and dismissive to users"],
    trait_name="helpful"
)
```

##### `extract_activations()`

```python
extract_activations(text: str, layer: int | None = None) -> torch.Tensor
```

Extracts hidden state activations for a given text.

**Parameters:**
- `text` (str): Input text to process
- `layer` (int, optional): Layer to extract from

**Returns:**
- `torch.Tensor`: Activation vector of shape `[hidden_dim]`

---

### VectorCache

**Location:** `personasafe/core/vector_cache.py`

Disk-based caching system for persona vectors.

#### Constructor

```python
VectorCache(cache_dir: str = "vectors")
```

**Parameters:**
- `cache_dir` (str): Directory for storing cached vectors. Default: "vectors"

**Example:**
```python
from personasafe import VectorCache

cache = VectorCache(cache_dir="./my_vectors")
```

#### Methods

##### `get()`

```python
get(model_name: str, trait_name: str) -> Optional[torch.Tensor]
```

Retrieves a cached vector.

**Parameters:**
- `model_name` (str): Model identifier
- `trait_name` (str): Trait identifier

**Returns:**
- `torch.Tensor` or `None`: Cached vector if exists, otherwise None

**Example:**
```python
vector = cache.get("google/gemma-3-4b", "helpful")
if vector is None:
    print("Vector not cached")
```

##### `set()`

```python
set(model_name: str, trait_name: str, vector: torch.Tensor) -> None
```

Stores a vector in the cache.

**Parameters:**
- `model_name` (str): Model identifier
- `trait_name` (str): Trait identifier
- `vector` (torch.Tensor): Vector to cache

**Example:**
```python
cache.set("google/gemma-3-4b", "helpful", my_vector)
```

##### `list_cached()` / `list_cached_as_list()`

```python
list_cached() -> Dict[str, Any]
list_cached_as_list() -> List[Dict[str, Any]]
```

Lists all cached vectors with metadata (dict keyed by cache key) or as a list.

**Example:**
```python
cached_map = cache.list_cached()
cached_items = cache.list_cached_as_list()
```

---

## Screening Module (`personasafe.screening`)

### DataScreener

**Location:** `personasafe/screening/data_screener.py`

Screen datasets for personality drift using persona vectors.

#### Constructor

```python
DataScreener(
    extractor: PersonaExtractor,
    persona_vectors: Dict[str, torch.Tensor]
)
```

**Parameters:**
- `extractor` (PersonaExtractor): Initialized extractor instance
- `persona_vectors` (Dict[str, torch.Tensor]): Dictionary mapping trait names to vectors

**Example:**
```python
from personasafe import PersonaExtractor, DataScreener

extractor = PersonaExtractor("google/gemma-3-4b")
screener = DataScreener(
    extractor=extractor,
    persona_vectors={"helpful": helpful_vec, "toxic": toxic_vec}
)
```

#### Methods

##### `score_text()`

```python
score_text(text: str) -> Dict[str, float]
```

Scores a single text against all persona vectors.

**Parameters:**
- `text` (str): Text to score

**Returns:**
- `Dict[str, float]`: Dictionary mapping trait names to projection scores

**Example:**
```python
scores = screener.score_text("This is a helpful response")
# Returns: {"helpful": 0.85, "toxic": -0.23}
```

##### `screen_dataset()`

```python
screen_dataset(
    dataset: pd.DataFrame,
    text_column: str = "text"
) -> pd.DataFrame
```

Screens an entire dataset.

**Parameters:**
- `dataset` (pd.DataFrame): DataFrame containing texts
- `text_column` (str): Column name containing text. Default: "text"

**Returns:**
- `pd.DataFrame`: Original DataFrame with added score columns

**Example:**
```python
import pandas as pd

df = pd.DataFrame({"text": ["Sample 1", "Sample 2"]})
screened_df = screener.screen_dataset(df)
# Returns df with columns: text, helpful_score, toxic_score
```

##### `generate_report()`

```python
generate_report(
    screened_df: pd.DataFrame,
    risk_threshold: float = 0.5
) -> Dict[str, Any]
```

Generates a summary report from screened data.

**Parameters:**
- `screened_df` (pd.DataFrame): DataFrame from `screen_dataset()`
- `risk_threshold` (float): Threshold for high-risk classification. Default: 0.5

**Returns:**
- `Dict[str, Any]`: Keys include `total_samples`, `risk_threshold`,
  `high_risk_counts` (alias: `high_risk_samples`), `high_risk_indices`, `average_scores`.

---

## Steering Module (`personasafe.steering`)

### ActivationSteerer

**Location:** `personasafe/steering/activation_steerer.py`

Apply steering vectors during text generation.

#### Constructor

```python
ActivationSteerer(model, tokenizer)
```

**Parameters:**
- `model`: HuggingFace model instance
- `tokenizer`: HuggingFace tokenizer instance

**Example:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from personasafe import ActivationSteerer

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b")
steerer = ActivationSteerer(model, tokenizer)
```

#### Methods

##### `steer()`

```python
steer(
    prompt: str,
    persona_vector: torch.Tensor,
    multiplier: float = 1.0,
    layer: int = 20,
    max_new_tokens: int = 50
) -> List[str]
```

Generates text with and without steering for comparison.

**Parameters:**
- `prompt` (str): Input prompt
- `persona_vector` (torch.Tensor): Steering vector
- `multiplier` (float): Steering strength. Default: 1.0
- `layer` (int): Layer to apply steering. Default: 20
- `max_new_tokens` (int): Maximum tokens to generate. Default: 50

**Returns:**
- `List[str]`: [original_output, steered_output]

**Example:**
```python
outputs = steerer.steer(
    prompt="Write a story about",
    persona_vector=helpful_vector,
    multiplier=2.0
)
print("Original:", outputs[0])
print("Steered:", outputs[1])
```

---

## Type Hints

PersonaSafe uses Python type hints throughout. Import types:

```python
from typing import List, Dict, Optional, Any
import torch
import pandas as pd
```

---

## Error Handling

All PersonaSafe components raise standard Python exceptions:

- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing cache files
- `RuntimeError`: Model loading failures
- `KeyError`: Missing required keys in data structures

**Example error handling:**
```python
try:
    extractor = PersonaExtractor("invalid/model")
except Exception as e:
    print(f"Model loading failed: {e}")
```

---

## Configuration

### Environment Variables

PersonaSafe respects this environment variable:

- `HUGGINGFACE_TOKEN`: HuggingFace API token (required for gated models)

**Example:**
```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

---

## Examples

### Complete Pipeline

```python
from personasafe import PersonaExtractor, VectorCache, DataScreener
import pandas as pd

# 1. Setup
extractor = PersonaExtractor("google/gemma-3-4b")
cache = VectorCache()

# 2. Extract or load vectors
helpful_vec = cache.get("google/gemma-3-4b", "helpful")
if helpful_vec is None:
    helpful_vec = extractor.compute_persona_vector(
        positive_prompts=["Be very helpful"],
        negative_prompts=["Be unhelpful"],
        trait_name="helpful"
    )

# 3. Screen dataset
screener = DataScreener(
    extractor=extractor,
    persona_vectors={"helpful": helpful_vec}
)

df = pd.read_json("dataset.jsonl", lines=True)
screened_df = screener.screen_dataset(df, text_column="text")

# 4. Generate report
report = screener.generate_report(screened_df, risk_threshold=0.6)
print(report)
```

---

## Version Information

```python
import personasafe
print(personasafe.__version__)  # "0.1.0"
```

---

**Last Updated:** October 25, 2025
**API Version:** 0.1.0
**Python Compatibility:** 3.10+
