#!/usr/bin/env python
"""Tests for the ActivationSteerer class."""

import pytest
import torch
from unittest.mock import patch, MagicMock, ANY

# Add project root to path to allow imports
import sys

sys.path.insert(0, ".")

from personasafe.steering.activation_steerer import ActivationSteerer
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture
def mock_model_and_tokenizer():
    """Pytest fixture to create a mock model and tokenizer."""
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_tokenizer = MagicMock()

    # Mock the tokenizer encoding by setting the return value of the mock itself
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    # Let's have decode return different things to distinguish the calls
    mock_tokenizer.decode.side_effect = ["original output", "steered output"]

    # Make model.generate return different values for each call
    mock_model.generate.side_effect = [
        torch.tensor([[1, 2, 3, 4, 5]]),  # First call (original)
        torch.tensor([[1, 2, 3, 6, 7]]),  # Second call (steered)
    ]

    return mock_model, mock_tokenizer


@patch("personasafe.steering.activation_steerer.SteeringVector")
def test_steer_method_calls(MockSteeringVector, mock_model_and_tokenizer):
    """Test that the steer method correctly uses the SteeringVector context manager."""
    # Arrange
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    # The mock for the SteeringVector class itself
    mock_sv_instance = MockSteeringVector.return_value

    steerer = ActivationSteerer(model=mock_model, tokenizer=mock_tokenizer)

    prompt = "test prompt"
    persona_vector = torch.randn(1024)
    multiplier = 1.5
    layer = 15

    # Act
    original_text, steered_text = steerer.steer(
        prompt=prompt,
        persona_vector=persona_vector,
        multiplier=multiplier,
        layer=layer,
        max_new_tokens=50,
    )

    # Assert
    # 1. Check that SteeringVector was initialized correctly
    MockSteeringVector.assert_called_once_with(layer_activations={layer: ANY})

    # 2. Check that the .apply() context manager was used
    mock_sv_instance.apply.assert_called_once_with(
        mock_model, multiplier=multiplier, operator=ANY
    )

    # 3. Check that model.generate was called twice (once outside, once inside the context)
    assert mock_model.generate.call_count == 2

    # 4. Check that the outputs are what we expect from the mocks
    assert original_text == "original output"
    assert steered_text == "steered output"

    print("\n✅ Unit test for ActivationSteerer passed!")
