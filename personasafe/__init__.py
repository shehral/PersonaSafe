"""
PersonaSafe: Safety Monitoring Toolkit for Google Gemma 3 Models

PersonaSafe implements the Persona Vectors methodology to detect personality drift
in language models before expensive fine-tuning runs.

Core Components:
    - PersonaExtractor: Extract persona vectors from models using contrastive prompts
    - VectorCache: Efficient disk-based caching for computed vectors
    - DataScreener: Screen datasets for personality drift
    - ActivationSteerer: Apply steering vectors during generation

Example:
    ```python
    from personasafe import PersonaExtractor, DataScreener
    import pandas as pd

    # Extract persona vectors
    extractor = PersonaExtractor(model_name="google/gemma-3-4b")
    helpful_vector = extractor.compute_persona_vector(
        positive_prompts=["Be very helpful and kind"],
        negative_prompts=["Be unhelpful and rude"],
        trait_name="helpful"
    )

    # Screen a dataset
    screener = DataScreener(
        extractor=extractor,
        persona_vectors={"helpful": helpful_vector}
    )
    df = pd.DataFrame({"text": ["Sample text 1", "Sample text 2"]})
    screened_df = screener.screen_dataset(df, text_column="text")
    report = screener.generate_report(screened_df)
    ```

See Also:
    - Documentation: docs/README.md
    - GitHub: https://github.com/shehral/PersonaSafe
    - Research Paper: https://arxiv.org/abs/2407.08338
"""

__version__ = "0.1.0"
__author__ = "PersonaSafe Development Team"
__license__ = "MIT"

# Core components
from personasafe.core.persona_extractor import PersonaExtractor
from personasafe.core.vector_cache import VectorCache

# Screening
from personasafe.screening.data_screener import DataScreener

# Steering
from personasafe.steering.activation_steerer import ActivationSteerer

# Public API
__all__ = [
    # Core
    "PersonaExtractor",
    "VectorCache",
    # Screening
    "DataScreener",
    # Steering
    "ActivationSteerer",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
