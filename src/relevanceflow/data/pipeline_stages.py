from __future__ import annotations

from pathlib import Path

import pandas as pd

from relevanceflow.data.clean import (
    build_processed_dataset,
    load_wands_raw,
    save_processed_dataset,
)
from relevanceflow.data.validate import (
    validate_wands,
    validation_summary,
)


def validate_raw_wands(
    data_dir: str | Path,
) -> dict:
    """Validate the raw WANDS dataset and return a JSON-safe summary."""

    data_dir = Path(data_dir)

    products, queries, labels = load_wands_raw(data_dir)

    result = validate_wands(
        queries=queries,
        products=products,
        labels=labels,
    )

    return validation_summary(result)


def clean_wands_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
) -> dict:
    """Clean raw WANDS data and write processed datasets."""

    products, queries, labels = load_wands_raw(raw_dir)

    (
        cleaned_products,
        cleaned_queries,
        cleaned_labels,
        conflict_audit,
    ) = build_processed_dataset(
        products=products,
        queries=queries,
        labels=labels,
    )

    save_processed_dataset(
        products=cleaned_products,
        queries=cleaned_queries,
        labels=cleaned_labels,
        conflict_audit=conflict_audit,
        output_dir=processed_dir,
    )

    return {
        "products": len(cleaned_products),
        "queries": len(cleaned_queries),
        "judgments": len(cleaned_labels),
        "conflicts": len(conflict_audit),
    }


def load_processed_judgments(
    processed_dir: str | Path,
) -> pd.DataFrame:
    """Load cleaned judgments."""

    processed_dir = Path(processed_dir)

    return pd.read_parquet(processed_dir / "judgments.parquet")
