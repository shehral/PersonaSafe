#!/usr/bin/env python
"""
Applies steering vectors to a model during generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector, addition_operator
from typing import List

class ActivationSteerer:
    """A wrapper class to apply steering vectors to a model."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Initializes the ActivationSteerer.

        Args:
            model: The HuggingFace model to steer.
            tokenizer: The tokenizer for the model.
        """
        self.model = model
        self.tokenizer = tokenizer

    def steer(
        self, 
        prompt: str,
        persona_vector: torch.Tensor,
        multiplier: float,
        layer: int,
        max_new_tokens: int = 100,
    ) -> List[str]:
        """
        Generates text with and without applying a steering vector.

        Args:
            prompt: The input prompt for the model.
            persona_vector: The persona vector to apply.
            multiplier: The strength of the steering vector.
            layer: The model layer to apply the steering vector to.
            max_new_tokens: The maximum number of tokens to generate.

        Returns:
            A list containing two strings: [original_output, steered_output]
        """
        # 1. Generate original, unsteered output
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        original_output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )
        original_text = self.tokenizer.decode(original_output_tokens[0], skip_special_tokens=True)

        # 2. Create the SteeringVector object
        steering_vector = SteeringVector(layer_activations={layer: persona_vector.to(self.model.device)})

        # 3. Generate steered output using the context manager
        with steering_vector.apply(self.model, multiplier=multiplier, operator=addition_operator()):
            steered_output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
        steered_text = self.tokenizer.decode(steered_output_tokens[0], skip_special_tokens=True)

        return [original_text, steered_text]
