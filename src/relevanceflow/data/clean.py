from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

LABEL_TO_SCORE = {
    "Irrelevant": 0,
    "Partial": 1,
    "Exact": 2,
}

SCORE_TO_LABEL = {
    0: "Irrelevant",
    1: "Partial",
    2: "Exact",
}


def load_wands_raw(
    data_dir: str | Path = "data/raw/wands",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw WANDS TSV files."""

    data_dir = Path(data_dir)

    products = pd.read_csv(
        data_dir / "product.csv",
        sep="\t",
    )

    queries = pd.read_csv(
        data_dir / "query.csv",
        sep="\t",
    )

    labels = pd.read_csv(
        data_dir / "label.csv",
        sep="\t",
    )

    return products, queries, labels


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize text without changing its semantic content.

    Operations:
    - convert null values to empty strings
    - cast to string
    - strip leading/trailing whitespace
    - collapse repeated whitespace
    """

    return (
        series.fillna("")
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean_queries(
    queries: pd.DataFrame,
) -> pd.DataFrame:
    """Clean WANDS query records."""

    cleaned = queries.copy()

    cleaned["query"] = normalize_text(cleaned["query"])

    if "query_class" in cleaned.columns:
        cleaned["query_class"] = normalize_text(cleaned["query_class"])

        # Preserve missing query_class semantics.
        cleaned.loc[
            cleaned["query_class"].eq(""),
            "query_class",
        ] = pd.NA

    return cleaned


def clean_products(
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Clean WANDS product records."""

    cleaned = products.copy()

    text_columns = [
        "product_name",
        "product_class",
        "category hierarchy",
        "product_description",
        "product_features",
    ]

    for column in text_columns:
        if column in cleaned.columns:
            cleaned[column] = normalize_text(cleaned[column])

    # Missing optional text fields remain empty strings.
    # This makes downstream vectorization deterministic.
    optional_text_columns = [
        "product_class",
        "category hierarchy",
        "product_description",
        "product_features",
    ]

    for column in optional_text_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("")

    return cleaned


def resolve_label_group(
    group: pd.DataFrame,
) -> tuple[str, str]:
    """
    Resolve all annotations belonging to one query-product pair.

    Returns:
        resolved_label
        resolution_method
    """

    labels = group["label"].tolist()

    if len(set(labels)) == 1:
        return labels[0], "agreement"

    score_counts = group["label"].map(LABEL_TO_SCORE).value_counts().sort_index()

    max_count = score_counts.max()

    majority_scores = score_counts[score_counts == max_count].index.tolist()

    # Unique majority label.
    if len(majority_scores) == 1:
        resolved_score = int(majority_scores[0])

        return (
            SCORE_TO_LABEL[resolved_score],
            "majority_vote",
        )

    # Tie: use the lower relevance score.
    resolved_score = int(min(majority_scores))

    return (
        SCORE_TO_LABEL[resolved_score],
        "majority_tie_conservative",
    )


def clean_labels(
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate and resolve WANDS query-product judgments.

    Returns:
        cleaned_labels
        conflict_audit
    """

    rows: list[dict[str, Any]] = []

    for (query_id, product_id), group in labels.groupby(
        ["query_id", "product_id"],
        sort=True,
    ):
        resolved_label, method = resolve_label_group(group)

        original_labels = sorted(group["label"].unique().tolist())

        rows.append(
            {
                "query_id": query_id,
                "product_id": product_id,
                "label": resolved_label,
                "annotation_count": len(group),
                "original_labels": "|".join(original_labels),
                "resolution_method": method,
            }
        )

    cleaned = pd.DataFrame(rows)

    cleaned["relevance_score"] = cleaned["label"].map(LABEL_TO_SCORE).astype("int8")

    conflict_audit = cleaned[
        cleaned["original_labels"].str.contains(
            r"\|",
            regex=True,
        )
    ].copy()

    return cleaned, conflict_audit


def build_processed_dataset(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run the complete WANDS cleaning pipeline.
    """

    cleaned_products = clean_products(products)

    cleaned_queries = clean_queries(queries)

    cleaned_labels, conflict_audit = clean_labels(labels)

    return (
        cleaned_products,
        cleaned_queries,
        cleaned_labels,
        conflict_audit,
    )


def save_processed_dataset(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    conflict_audit: pd.DataFrame,
    output_dir: str | Path = "data/processed/wands",
) -> None:
    """Save cleaned datasets as Parquet."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    products.to_parquet(
        output_dir / "products.parquet",
        index=False,
    )

    queries.to_parquet(
        output_dir / "queries.parquet",
        index=False,
    )

    labels.to_parquet(
        output_dir / "judgments.parquet",
        index=False,
    )

    conflict_audit.to_parquet(
        output_dir / "conflict_audit.parquet",
        index=False,
    )


def main() -> int:
    """CLI entry point."""

    print("=" * 60)
    print("RelevanceFlow — WANDS Data Cleaning")
    print("=" * 60)

    products, queries, labels = load_wands_raw()

    print(f"Raw products:   {len(products):,}")
    print(f"Raw queries:    {len(queries):,}")
    print(f"Raw judgments:  {len(labels):,}")

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
    )

    print()
    print("Processed:")
    print(f"Products:       {len(cleaned_products):,}")
    print(f"Queries:        {len(cleaned_queries):,}")
    print(f"Judgments:      {len(cleaned_labels):,}")
    print(f"Conflicts:      {len(conflict_audit):,}")

    print()
    print("Resolution methods:")

    print(cleaned_labels["resolution_method"].value_counts().to_string())

    print()
    print("Saved to:")
    print("  data/processed/wands/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
