#!/usr/bin/env python
"""Tests for the PersonaExtractor class."""

import pytest
import torch
from unittest.mock import patch, MagicMock

# Add project root to path to allow imports
import sys

sys.path.insert(0, ".")

from personasafe.core.persona_extractor import PersonaExtractor


@pytest.fixture
def mock_model_and_tokenizer():
    """Pytest fixture to create mock model and tokenizer."""
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    # The tokenizer returns a dictionary of tensors
    mock_tokenizer.return_value = {
        "input_ids": torch.randint(0, 100, (1, 5)),
        "attention_mask": torch.ones(1, 5),
    }

    # Mock the model
    mock_model = MagicMock()
    mock_model.device = "cpu"
    # The model's forward pass returns an object with a `hidden_states` attribute
    mock_hidden_states = [
        torch.randn(1, 5, 1024) for _ in range(12)
    ]  # 12 layers of hidden states
    mock_outputs = MagicMock()
    mock_outputs.hidden_states = mock_hidden_states
    mock_model.return_value = mock_outputs
    mock_model.config.hidden_size = 1024
    mock_model.num_parameters.return_value = 1000  # Add this line to fix the TypeError

    return mock_model, mock_tokenizer


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_persona_extractor_logic(
    mock_tokenizer_from_pretrained, mock_model_from_pretrained, mock_model_and_tokenizer
):
    """Tests the core vector computation logic without loading a real model."""
    # Arrange
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    mock_model_from_pretrained.return_value = mock_model
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    # Instantiate the extractor. This will use the mocked loader.
    extractor = PersonaExtractor(model_name="mock-model")
    extractor._load_model()  # Explicitly load the mock model

    # Define a predictable set of activations for the mock model to return
    # The first call to extract_activations will return this
    positive_activation_1 = torch.ones(1, 5, 1024) * 2.0
    # The second call will return this
    positive_activation_2 = torch.ones(1, 5, 1024) * 4.0
    # The third call
    negative_activation_1 = torch.ones(1, 5, 1024) * 0.5
    # The fourth call
    negative_activation_2 = torch.ones(1, 5, 1024) * 1.5

    # The hook inside extract_activations will see these values in order
    extractor.model.return_value.hidden_states = [
        positive_activation_1,
        positive_activation_2,
        negative_activation_1,
        negative_activation_2,
    ]

    # Act
    # We need to mock the `extract_activations` method to control its output directly for this test
    activations_to_return = [
        positive_activation_1.mean(dim=1).squeeze(),
        positive_activation_2.mean(dim=1).squeeze(),
        negative_activation_1.mean(dim=1).squeeze(),
        negative_activation_2.mean(dim=1).squeeze(),
    ]
    extractor.extract_activations = MagicMock(side_effect=activations_to_return)

    persona_vector = extractor.compute_persona_vector(
        positive_prompts=["be good", "be nice"],
        negative_prompts=["be bad", "be mean"],
        trait_name="test_trait",
    )

    # Assert
    # Expected positive_mean = (2.0 + 4.0) / 2 = 3.0
    # Expected negative_mean = (0.5 + 1.5) / 2 = 1.0
    # Expected difference = 3.0 - 1.0 = 2.0
    expected_diff_vector = torch.ones(1024) * 2.0
    expected_normalized_vector = expected_diff_vector / torch.norm(expected_diff_vector)

    assert persona_vector is not None
    assert isinstance(persona_vector, torch.Tensor)
    assert torch.allclose(persona_vector, expected_normalized_vector, atol=1e-6)
    assert abs(persona_vector.norm().item() - 1.0) < 1e-6
    print("\n✅ Unit test for PersonaExtractor logic passed!")


@patch("personasafe.core.persona_extractor.VectorCache")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_extractor_uses_cache(
    mock_tokenizer_loader, mock_model_loader, mock_cache_class
):
    """Tests that the extractor uses the cache to avoid re-computation."""
    # Arrange
    # Mock the loaders so we don't load real models
    mock_model_loader.return_value = MagicMock()
    mock_tokenizer_loader.return_value = MagicMock()

    # Mock the VectorCache instance and its methods
    mock_cache_instance = mock_cache_class.return_value
    mock_vector = torch.randn(1024)  # A dummy vector

    # 1. First call: cache miss
    mock_cache_instance.get.return_value = None

    # 2. Second call: cache hit
    # We use side_effect to change the return value on subsequent calls
    mock_cache_instance.get.side_effect = [None, mock_vector]

    extractor = PersonaExtractor(model_name="mock-model-cache-test")
    # Mock the expensive computation part to track its calls
    extractor._load_model = MagicMock()
    positive_activation = torch.ones(1024)
    negative_activation = torch.zeros(1024)
    extractor.extract_activations = MagicMock(
        side_effect=[positive_activation, negative_activation]
    )

    # Act
    # First call should compute the vector
    vec1 = extractor.compute_persona_vector(["pos"], ["neg"], "caching_trait")
    # Second call should hit the cache
    vec2 = extractor.compute_persona_vector(["pos"], ["neg"], "caching_trait")

    # Assert
    # 1. Check that cache.get was called twice
    assert mock_cache_instance.get.call_count == 2
    mock_cache_instance.get.assert_any_call("mock-model-cache-test", "caching_trait")

    # 2. Check that the expensive _load_model was only called ONCE (on cache miss)
    assert extractor._load_model.call_count == 1

    # 3. Check that cache.set was only called ONCE (on cache miss)
    assert mock_cache_instance.set.call_count == 1
    mock_cache_instance.set.assert_called_once_with(
        "mock-model-cache-test", "caching_trait", vec1
    )

    # 4. Check that the second call returned the cached vector
    assert torch.allclose(vec2, mock_vector)

    print("\n✅ Unit test for PersonaExtractor caching passed!")


def test_compute_persona_vector_requires_non_empty_prompts(monkeypatch):
    """Ensure empty prompt lists raise a helpful error when cache misses."""
    extractor = PersonaExtractor(model_name="mock-model-empty")
    extractor.cache.get = MagicMock(return_value=None)

    # Prevent actual model loading in this unit test
    monkeypatch.setattr(extractor, "_load_model", MagicMock())

    with pytest.raises(ValueError) as excinfo_pos:
        extractor.compute_persona_vector([], ["neg"], "empty-positive")
    assert "positive_prompts must contain at least one prompt" in str(excinfo_pos.value)

    with pytest.raises(ValueError) as excinfo_neg:
        extractor.compute_persona_vector(["pos"], [], "empty-negative")
    assert "negative_prompts must contain at least one prompt" in str(excinfo_neg.value)


def test_compute_persona_vector_raises_on_zero_norm(monkeypatch):
    """Guard against zero-norm persona vectors."""
    extractor = PersonaExtractor(model_name="mock-model-zero")
    extractor.cache.get = MagicMock(return_value=None)
    extractor.cache.set = MagicMock()

    monkeypatch.setattr(extractor, "_load_model", MagicMock())

    identical_activation = torch.ones(16)
    extractor.extract_activations = MagicMock(return_value=identical_activation)

    with pytest.raises(ValueError) as excinfo:
        extractor.compute_persona_vector(["pos"], ["neg"], "degenerate-trait")

    assert "zero-norm persona vector" in str(excinfo.value)
    extractor.cache.set.assert_not_called()
