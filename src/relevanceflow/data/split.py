from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class SplitError(ValueError):
    """Raised when query-level dataset splitting fails."""


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for query-level train/validation/test splitting."""

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    random_seed: int = 42

    def validate(self) -> None:
        """Validate split configuration."""

        sizes = [
            self.train_size,
            self.validation_size,
            self.test_size,
        ]

        if any(size <= 0 or size >= 1 for size in sizes):
            raise SplitError("All split sizes must be between 0 and 1.")

        total = sum(sizes)

        if abs(total - 1.0) > 1e-9:
            raise SplitError(f"Split sizes must sum to 1.0, got {total:.6f}.")


def validate_judgments(judgments: pd.DataFrame) -> None:
    """
    Validate the input judgments dataframe.

    Required column:
        query_id
    """

    if "query_id" not in judgments.columns:
        raise SplitError("Input judgments must contain 'query_id'.")

    if judgments.empty:
        raise SplitError("Input judgments dataframe is empty.")

    if judgments["query_id"].isna().any():
        raise SplitError("Input judgments contain missing query_id values.")


def get_unique_query_ids(judgments: pd.DataFrame) -> pd.Series:
    """Return deterministic unique query IDs."""

    validate_judgments(judgments)

    return judgments["query_id"].drop_duplicates().sort_values().reset_index(drop=True)


def split_query_ids(
    query_ids: pd.Series,
    config: SplitConfig,
) -> dict[str, pd.Series]:
    """
    Split unique query IDs into train, validation, and test groups.

    The split is performed ONLY on unique query IDs.
    """

    config.validate()

    if query_ids.empty:
        raise SplitError("No query IDs available for splitting.")

    if query_ids.duplicated().any():
        raise SplitError("query_ids must contain unique values.")

    if len(query_ids) < 3:
        raise SplitError(
            "At least 3 unique queries are required for train/validation/test."
        )

    train_ids, temp_ids = train_test_split(
        query_ids,
        test_size=config.validation_size + config.test_size,
        random_state=config.random_seed,
        shuffle=True,
    )

    validation_fraction_of_temp = config.validation_size / (
        config.validation_size + config.test_size
    )

    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1.0 - validation_fraction_of_temp,
        random_state=config.random_seed,
        shuffle=True,
    )

    train_ids = pd.Series(train_ids).reset_index(drop=True)
    validation_ids = pd.Series(validation_ids).reset_index(drop=True)
    test_ids = pd.Series(test_ids).reset_index(drop=True)

    return {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }


def verify_query_disjointness(
    split_query_ids_dict: dict[str, pd.Series],
) -> None:
    """Verify that no query occurs in more than one split."""

    train = set(split_query_ids_dict["train"])
    validation = set(split_query_ids_dict["validation"])
    test = set(split_query_ids_dict["test"])

    train_validation_overlap = train.intersection(validation)
    train_test_overlap = train.intersection(test)
    validation_test_overlap = validation.intersection(test)

    if train_validation_overlap:
        raise SplitError(
            "Train/validation query overlap detected: "
            f"{sorted(train_validation_overlap)}"
        )

    if train_test_overlap:
        raise SplitError(
            f"Train/test query overlap detected: {sorted(train_test_overlap)}"
        )

    if validation_test_overlap:
        raise SplitError(
            f"Validation/test query overlap detected: {sorted(validation_test_overlap)}"
        )


def assign_split(
    judgments: pd.DataFrame,
    split_query_ids_dict: dict[str, pd.Series],
) -> dict[str, pd.DataFrame]:
    """
    Assign every judgment row to the split belonging to its query_id.
    """

    validate_judgments(judgments)
    verify_query_disjointness(split_query_ids_dict)

    query_to_split = {}

    for split_name, query_ids in split_query_ids_dict.items():
        for query_id in query_ids:
            query_to_split[query_id] = split_name

    split_labels = judgments["query_id"].map(query_to_split)

    if split_labels.isna().any():
        missing_queries = judgments.loc[split_labels.isna(), "query_id"].unique()

        raise SplitError(
            "Some judgments could not be assigned to a split. "
            f"Missing query IDs: {missing_queries.tolist()}"
        )

    result = {}

    for split_name in ("train", "validation", "test"):
        split_df = judgments.loc[split_labels == split_name].copy()

        # Stable ordering makes generated parquet files reproducible.
        split_df = split_df.sort_values(by=["query_id", "product_id"]).reset_index(
            drop=True
        )

        result[split_name] = split_df

    return result


def split_judgments(
    judgments: pd.DataFrame,
    config: SplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Perform a complete query-level split.

    Returns:
        {
            "train": train dataframe,
            "validation": validation dataframe,
            "test": test dataframe,
        }
    """

    config = config or SplitConfig()

    query_ids = get_unique_query_ids(judgments)

    split_ids = split_query_ids(
        query_ids=query_ids,
        config=config,
    )

    return assign_split(
        judgments=judgments,
        split_query_ids_dict=split_ids,
    )


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save split datasets as parquet files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_splits = {"train", "validation", "test"}

    if set(splits) != required_splits:
        raise SplitError(
            "Splits must contain exactly: train, validation, validation, test."
        )

    output_paths = {
        "train": output_dir / "train.parquet",
        "validation": output_dir / "validation.parquet",
        "test": output_dir / "test.parquet",
    }

    for split_name, path in output_paths.items():
        splits[split_name].to_parquet(
            path,
            index=False,
        )


def create_query_level_split(
    input_path: str | Path,
    output_dir: str | Path,
    config: SplitConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load cleaned judgments, perform query-level splitting,
    validate the result, and save parquet files.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Judgments file not found: {input_path}")

    judgments = pd.read_parquet(input_path)

    splits = split_judgments(
        judgments=judgments,
        config=config,
    )

    save_splits(
        splits=splits,
        output_dir=output_dir,
    )

    return splits
