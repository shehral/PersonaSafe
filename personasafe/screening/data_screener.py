#!/usr/bin/env python
"""
Screens datasets for personality drift risks using persona vectors.
"""

from typing import List, Dict, Any
import torch
import pandas as pd
from pathlib import Path
import logging

# Add project root to path to allow imports
import sys

sys.path.insert(0, ".")

from personasafe.core.persona_extractor import PersonaExtractor

logger = logging.getLogger(__name__)


class DataScreener:
    """Screens text data against pre-computed persona vectors."""

    def __init__(
        self, extractor: PersonaExtractor, persona_vectors: Dict[str, torch.Tensor]
    ):
        """
        Initializes the DataScreener.

        Args:
            extractor: An initialized PersonaExtractor instance.
            persona_vectors: A dictionary mapping trait names to their corresponding persona vectors.
        """
        self.extractor = extractor
        self.persona_vectors = persona_vectors
        logger.info(
            f"DataScreener initialized for model {extractor.model_name} with {len(persona_vectors)} persona vectors."
        )

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Scores a single piece of text against all loaded persona vectors.

        Args:
            text: The text to score.

        Returns:
            A dictionary mapping trait names to their projection scores.
        """
        # 1. Get the activation vector for the input text.
        activation_vector = self.extractor.extract_activations(text)

        scores: Dict[str, float] = {}
        # 2. Calculate the projection score for each trait.
        for trait, persona_vector in self.persona_vectors.items():
            # Ensure vectors are on the same device
            persona_vector_device = persona_vector.to(activation_vector.device)

            # Calculate the dot product (projection)
            score = torch.dot(activation_vector, persona_vector_device).item()
            scores[trait] = score

        return scores

    def screen_dataset(
        self,
        dataset: pd.DataFrame,
        text_column: str = "text",
    ) -> pd.DataFrame:
        """
        Screens an entire dataset, adding a score for each persona vector.

        Args:
            dataset: A pandas DataFrame containing the text data.
            text_column: The name of the column in the DataFrame that contains the text to screen.

        Returns:
            The original DataFrame with added columns for each trait's score.
        """
        if text_column not in dataset.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Prepare a list to hold the new score data
        scores_data = []

        # Iterate through the DataFrame with a progress bar
        from tqdm import tqdm

        tqdm.pandas(desc="Screening Dataset")

        for text in dataset[text_column].progress_apply(self.score_text):
            scores_data.append(text)

        # Create a DataFrame from the scores
        scores_df = pd.DataFrame(scores_data, index=dataset.index)
        scores_df = scores_df.add_suffix("_score")

        # Join the scores DataFrame with the original dataset
        return dataset.join(scores_df)

    def generate_report(
        self, screened_df: pd.DataFrame, risk_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generates a summary report from a screened dataset.

        Args:
            screened_df: A DataFrame that has been processed by `screen_dataset`.
            risk_threshold: The score threshold to flag a sample as high-risk.

        Returns:
            A dictionary containing the report summary.
        """
        report: Dict[str, Any] = {
            "total_samples": len(screened_df),
            "risk_threshold": risk_threshold,
            "high_risk_counts": {},
            "high_risk_indices": {},
            "average_scores": {},
        }

        score_columns = [col for col in screened_df.columns if col.endswith("_score")]

        for col in score_columns:
            trait_name = col.replace("_score", "")
            high_risk_mask = screened_df[col] > risk_threshold

            report["high_risk_counts"][trait_name] = int(high_risk_mask.sum())
            report["high_risk_indices"][trait_name] = screened_df.index[
                high_risk_mask
            ].tolist()
            report["average_scores"][trait_name] = screened_df[col].mean()

        # Back-compat alias for docs that expect 'high_risk_samples'
        report["high_risk_samples"] = dict(report["high_risk_counts"])  # alias

        logger.info(f"Generated report for {len(screened_df)} samples.")
        return report
