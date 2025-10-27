from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from .vector_cache import VectorCache

logger = logging.getLogger(__name__)


class PersonaExtractor:
    """Extracts persona vectors from language models.
    
    A persona vector represents a personality trait (e.g., helpfulness)
    as a direction in the model's activation space.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "auto",
        layer_idx: int = -1,
        cache_dir: str = "vectors"
    ):
        """
        Initialize persona extractor.

        Args:
            model_name: HuggingFace model identifier (e.g., "google/gemma-3-4b-it")
            device: Device to use ("cuda", "cpu", or "auto")
            layer_idx: Which layer to extract from (-1 for last layer)
            cache_dir: Directory to store cached vectors.
        """
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.cache = VectorCache(cache_dir=cache_dir)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing PersonaExtractor for {model_name} on {self.device}")
        
        # Defer model loading until it's actually needed
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Load the model and tokenizer only when needed."""
        if self.model is not None and self.tokenizer is not None:
            logger.info("Model and tokenizer already loaded.")
            return

        try:
            logger.info(f"Loading model {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()  # Evaluation mode (no gradients)
            
            logger.info(f"Model loaded successfully ({self.model.num_parameters():,} parameters)")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_activations(self, text: str, layer: Optional[int] = None) -> torch.Tensor:
        """
        Extract hidden state activations for text.
        
        Args:
            text: Input text to process
            
        Returns:
            torch.Tensor: Activation vector from specified layer
        """
        # Ensure model is loaded before extracting activations
        if self.model is None or self.tokenizer is None:
            self._load_model()
            
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Determine which layer to use
        target_layer_idx = self.layer_idx if layer is None else layer

        # Extract hidden states from specified layer
        hidden_states = outputs.hidden_states[target_layer_idx]
        
        # Average over sequence length to get single vector
        activation = hidden_states.mean(dim=1).squeeze()
        
        return activation
    
    def compute_persona_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        trait_name: str = "unnamed",
        layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute persona vector from contrastive prompts, using cache if available.
        
        Args:
            positive_prompts: Prompts that embody the trait
            negative_prompts: Prompts that oppose the trait
            trait_name: Name of the trait (for logging)
            layer: Optional layer index to override the extractor default
            
        Returns:
            torch.Tensor: Normalized persona vector
        """
        # 1. Check cache first
        cached_vector = self.cache.get(self.model_name, trait_name)
        if cached_vector is not None:
            return cached_vector

        # 2. If not in cache, load model and compute
        logger.info(f"Cache miss for {self.model_name}/{trait_name}. Computing new vector.")
        self._load_model()

        logger.info(f"Computing persona vector for '{trait_name}'...")
        logger.info(f"  Using {len(positive_prompts)} positive and {len(negative_prompts)} negative prompts")
        
        # Extract activations for positive prompts
        positive_activations = []
        for i, prompt in enumerate(positive_prompts, 1):
            logger.debug(f"  Processing positive prompt {i}/{len(positive_prompts)}")
            act = self.extract_activations(prompt, layer=layer)
            positive_activations.append(act)
        positive_mean = torch.stack(positive_activations).mean(dim=0)
        
        # Extract activations for negative prompts
        negative_activations = []
        for i, prompt in enumerate(negative_prompts, 1):
            logger.debug(f"  Processing negative prompt {i}/{len(negative_prompts)}")
            act = self.extract_activations(prompt, layer=layer)
            negative_activations.append(act)
        negative_mean = torch.stack(negative_activations).mean(dim=0)
        
        # Compute difference vector (this is the persona vector!)
        persona_vector = positive_mean - negative_mean
        
        # Normalize to unit length
        persona_vector = persona_vector / torch.norm(persona_vector)
        
        logger.info(f"✓ Persona vector computed for '{trait_name}'")
        
        # 3. Save the newly computed vector to the cache
        self.cache.set(self.model_name, trait_name, persona_vector)
        
        return persona_vector
