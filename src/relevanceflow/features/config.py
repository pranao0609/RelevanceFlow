from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Controls which feature groups are generated."""

    lexical_enabled: bool = True
    product_text_enabled: bool = True
    metadata_enabled: bool = True
    compatibility_enabled: bool = True
    semantic_enabled: bool = False
