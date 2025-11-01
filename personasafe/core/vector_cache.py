#!/usr/bin/env python
"""
Efficient caching for persona vectors.
"""
from typing import Dict, Optional, List
from pathlib import Path
import torch
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class VectorCache:
    """Cache for persona vectors to avoid recomputation."""

    def __init__(self, cache_dir: str = "vectors"):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cached vectors
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / "metadata.json"

        # Load metadata
        self.metadata = self._load_metadata()
        self._save_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        if self.metadata_path.exists():
            with self.metadata_path.open("r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Metadata file is corrupted. Starting fresh.")
                    return {}
        return {}

    def _save_metadata(self):
        """Save the current metadata to disk."""
        with self.metadata_path.open("w") as f:
            json.dump(self.metadata, f, indent=2)

    def _get_cache_key(self, model_name: str, trait_name: str) -> str:
        """Generate a unique and consistent cache key."""
        # Sanitize model name for filesystem friendliness
        sanitized_model_name = model_name.replace("/", "__")
        key_string = f"{sanitized_model_name}_{trait_name}"
        # Use hashlib for a short, unique, and filesystem-safe key
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, model_name: str, trait_name: str) -> Optional[torch.Tensor]:
        """
        Get a cached vector if it exists.

        Args:
            model_name: The model identifier.
            trait_name: The name of the trait.

        Returns:
            The cached torch.Tensor, or None if not found.
        """
        cache_key = self._get_cache_key(model_name, trait_name)

        if cache_key in self.metadata:
            vector_path = Path(self.metadata[cache_key]["path"])
            if vector_path.exists():
                logger.info(f"Loading cached vector for {model_name}/{trait_name}")
                return torch.load(vector_path, map_location="cpu")
            else:
                logger.warning(f"Metadata points to a non-existent file: {vector_path}")
                # Clean up stale metadata entry
                del self.metadata[cache_key]
                self._save_metadata()

        logger.info(f"No cache hit for {model_name}/{trait_name}")
        return None

    def set(self, model_name: str, trait_name: str, vector: torch.Tensor):
        """
        Cache a persona vector to disk.

        Args:
            model_name: The model identifier.
            trait_name: The name of the trait.
            vector: The persona vector tensor to cache.
        """
        cache_key = self._get_cache_key(model_name, trait_name)
        # Use the cache key for the filename to ensure uniqueness
        vector_path = self.cache_dir / f"{cache_key}.pt"

        # Persist a CPU copy to keep cache portable across devices
        vector_to_save = vector.detach().cpu()
        torch.save(vector_to_save, vector_path)

        # Update and save metadata
        self.metadata[cache_key] = {
            "model_name": model_name,
            "trait_name": trait_name,
            "path": str(vector_path),
            "shape": list(vector_to_save.shape),
            "norm": vector_to_save.norm().item(),
            "source_device": str(vector.device),
        }
        self._save_metadata()
        logger.info(f"Successfully cached vector for {model_name}/{trait_name}")

    def list_cached(self) -> Dict[str, Dict]:
        """List all cached vectors."""
        return self.metadata

    def list_cached_as_list(self) -> List[Dict]:
        """Return cached vectors as a list of metadata dicts for convenience."""
        return [
            {
                **meta,
                "cache_key": key,
            }
            for key, meta in self.metadata.items()
        ]
