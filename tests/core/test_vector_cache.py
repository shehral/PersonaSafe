#!/usr/bin/env python
"""Tests for the VectorCache class."""

import json
import pytest
import torch
from pathlib import Path

from personasafe.core.vector_cache import VectorCache


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> str:
    """Create a temporary directory for the cache for testing."""
    return str(tmp_path / "test_cache")


def test_cache_initialization_nested_dirs(tmp_path: Path):
    """Cache init should create nested directories when requested."""
    nested_dir = tmp_path / "nested" / "persona" / "cache"
    VectorCache(cache_dir=str(nested_dir))
    assert nested_dir.is_dir()
    assert (nested_dir / "metadata.json").exists()


def test_cache_initialization(temp_cache_dir: str):
    """Test that the cache directory and metadata are created on init."""
    # Arrange & Act
    VectorCache(cache_dir=temp_cache_dir)

    # Assert
    assert Path(temp_cache_dir).is_dir()
    assert (Path(temp_cache_dir) / "metadata.json").exists()


def test_get_non_existent_item(temp_cache_dir: str):
    """Test that getting a non-existent item returns None."""
    # Arrange
    cache = VectorCache(cache_dir=temp_cache_dir)

    # Act
    result = cache.get("test-model", "test-trait")

    # Assert
    assert result is None


def test_set_and_get_item(temp_cache_dir: str):
    """Test that setting an item allows it to be retrieved."""
    # Arrange
    cache = VectorCache(cache_dir=temp_cache_dir)
    model_name = "test-model"
    trait_name = "test-trait"
    vector_to_cache = torch.randn(128)

    # Act
    cache.set(model_name, trait_name, vector_to_cache)
    retrieved_vector = cache.get(model_name, trait_name)

    # Assert
    assert retrieved_vector is not None
    assert torch.allclose(vector_to_cache.cpu(), retrieved_vector)


def test_metadata_is_updated(temp_cache_dir: str):
    """Test that the metadata.json file is correctly updated after a set operation."""
    # Arrange
    cache = VectorCache(cache_dir=temp_cache_dir)
    model_name = "test-model-2"
    trait_name = "test-trait-2"
    vector_to_cache = torch.randn(256)

    # Act
    cache.set(model_name, trait_name, vector_to_cache)

    # Assert
    metadata_path = Path(temp_cache_dir) / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    cache_key = cache._get_cache_key(model_name, trait_name)
    assert cache_key in metadata
    assert metadata[cache_key]["model_name"] == model_name
    assert metadata[cache_key]["trait_name"] == trait_name
    assert Path(metadata[cache_key]["path"]).exists()
    assert metadata[cache_key]["source_device"] == str(vector_to_cache.device)


def test_get_uses_cpu_map_location(temp_cache_dir: str, monkeypatch):
    """Verify that torch.load is invoked with map_location='cpu'."""
    cache = VectorCache(cache_dir=temp_cache_dir)
    model_name = "test-model-3"
    trait_name = "test-trait-3"
    vector_to_cache = torch.randn(64)
    cache.set(model_name, trait_name, vector_to_cache)

    original_load = torch.load
    call_args = {}

    def fake_load(path, map_location=None):
        call_args["map_location"] = map_location
        return original_load(path, map_location=map_location)

    monkeypatch.setattr(torch, "load", fake_load)

    retrieved = cache.get(model_name, trait_name)
    assert retrieved is not None
    assert call_args["map_location"] == "cpu"
