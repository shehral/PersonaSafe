#!/usr/bin/env python
"""Tests for the DataScreener class."""

import pytest
import torch
import pandas as pd
from unittest.mock import patch, MagicMock
from typing import Dict

# Add project root to path to allow imports
import sys

sys.path.insert(0, ".")

from personasafe.core.persona_extractor import PersonaExtractor
from personasafe.screening.data_screener import DataScreener


@pytest.fixture
def mock_persona_vectors() -> Dict[str, torch.Tensor]:
    """Fixture for creating mock persona vectors."""
    # Create two orthogonal vectors for predictable projections
    vec1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    vec2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    return {"trait1": vec1, "trait2": vec2}


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Fixture for a mock PersonaExtractor."""

    extractor = MagicMock(spec=PersonaExtractor)

    extractor.model_name = "mock-model"

    return extractor


def test_score_text(
    mock_extractor: MagicMock, mock_persona_vectors: Dict[str, torch.Tensor]
):
    """Test that score_text correctly calculates projection scores."""

    # Arrange

    activation_vector = torch.tensor([0.5, -0.5, 0.0, 0.0])

    mock_extractor.extract_activations.return_value = activation_vector

    screener = DataScreener(
        extractor=mock_extractor, persona_vectors=mock_persona_vectors
    )

    # Act

    scores = screener.score_text("some text")

    # Assert

    assert "trait1" in scores

    assert "trait2" in scores

    assert pytest.approx(scores["trait1"]) == 0.5

    assert pytest.approx(scores["trait2"]) == -0.5

    mock_extractor.extract_activations.assert_called_once_with("some text")

    print("\n✅ Unit test for score_text passed!")


def test_screen_dataset(
    mock_extractor: MagicMock, mock_persona_vectors: Dict[str, torch.Tensor]
):
    """Test that screen_dataset correctly adds score columns to a DataFrame."""

    # Arrange

    mock_extractor.extract_activations.side_effect = [
        torch.tensor([0.8, 0.1, 0, 0]),  # for "text 1"
        torch.tensor([-0.2, 0.9, 0, 0]),  # for "text 2"
    ]

    screener = DataScreener(
        extractor=mock_extractor, persona_vectors=mock_persona_vectors
    )

    dataset = pd.DataFrame({"text": ["text 1", "text 2"]})

    # Act

    screened_df = screener.screen_dataset(dataset, text_column="text")

    # Assert

    assert "trait1_score" in screened_df.columns

    assert "trait2_score" in screened_df.columns

    assert pytest.approx(screened_df.loc[0, "trait1_score"]) == 0.8

    assert pytest.approx(screened_df.loc[1, "trait1_score"]) == -0.2

    assert pytest.approx(screened_df.loc[0, "trait2_score"]) == 0.1

    assert pytest.approx(screened_df.loc[1, "trait2_score"]) == 0.9

    print("\n✅ Unit test for screen_dataset passed!")


def test_generate_report(
    mock_extractor: MagicMock, mock_persona_vectors: Dict[str, torch.Tensor]
):
    """Test that the report generation correctly summarizes the screened data."""

    # Arrange

    screener = DataScreener(
        extractor=mock_extractor, persona_vectors=mock_persona_vectors
    )

    screened_data = {
        "text": ["a", "b", "c", "d"],
        "trait1_score": [0.1, 0.8, -0.2, 0.9],
        "trait2_score": [0.9, 0.2, 0.6, -0.4],
    }

    screened_df = pd.DataFrame(screened_data)

    # Act

    report = screener.generate_report(screened_df, risk_threshold=0.7)

    # Assert

    assert report["total_samples"] == 4

    assert report["risk_threshold"] == 0.7

    # For trait1, samples at index 1 and 3 are high risk (0.8, 0.9)

    assert report["high_risk_counts"]["trait1"] == 2

    assert report["high_risk_indices"]["trait1"] == [1, 3]

    # For trait2, sample at index 0 is high risk (0.9)

    assert report["high_risk_counts"]["trait2"] == 1

    assert report["high_risk_indices"]["trait2"] == [0]

    print("\n✅ Unit test for generate_report passed!")
