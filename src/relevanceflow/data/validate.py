from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

VALID_LABELS = {"Exact", "Partial", "Irrelevant"}

REQUIRED_QUERY_COLUMNS = {
    "query_id",
    "query",
    "query_class",
}

REQUIRED_PRODUCT_COLUMNS = {
    "product_id",
    "product_name",
    "product_class",
    "category hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
}

REQUIRED_LABEL_COLUMNS = {
    "id",
    "query_id",
    "product_id",
    "label",
}

# Fields that must exist and contain values.
QUERY_REQUIRED_VALUE_COLUMNS = {
    "query_id",
    "query",
}

PRODUCT_REQUIRED_VALUE_COLUMNS = {
    "product_id",
    "product_name",
}

LABEL_REQUIRED_VALUE_COLUMNS = {
    "id",
    "query_id",
    "product_id",
    "label",
}


class DataValidationError(ValueError):
    """Raised when WANDS data validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors

        message = "WANDS data validation failed:\n" + "\n".join(
            f"- {error}" for error in errors
        )

        super().__init__(message)


@dataclass
class ValidationResult:
    """Container for validation results."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, check: str, message: str) -> None:
        self.errors.append(message)

        self.checks.append(
            {
                "check": check,
                "status": "FAIL",
                "message": message,
            }
        )

    def add_warning(self, check: str, message: str) -> None:
        self.warnings.append(message)

        self.checks.append(
            {
                "check": check,
                "status": "WARN",
                "message": message,
            }
        )

    def add_pass(self, check: str, message: str = "") -> None:
        self.checks.append(
            {
                "check": check,
                "status": "PASS",
                "message": message,
            }
        )


def _check_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
    result: ValidationResult,
) -> bool:
    """Check whether all required columns exist."""

    missing = sorted(required_columns - set(df.columns))

    if missing:
        result.add_error(
            f"{dataset_name}_required_columns",
            f"{dataset_name} is missing required columns: {missing}.",
        )
        return False

    result.add_pass(
        f"{dataset_name}_required_columns",
        "All required columns are present.",
    )

    return True


def _check_empty_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    result: ValidationResult,
) -> bool:
    """Check that a dataframe is not empty."""

    if df.empty:
        result.add_error(
            f"{dataset_name}_not_empty",
            f"{dataset_name} dataframe is empty.",
        )
        return False

    result.add_pass(
        f"{dataset_name}_not_empty",
        f"{len(df):,} rows found.",
    )

    return True


def _is_empty_string_series(series: pd.Series) -> pd.Series:
    """Return mask for null or whitespace-only values."""

    return series.isna() | series.astype("string").str.strip().eq("")


def _check_unique_ids(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
    result: ValidationResult,
) -> None:
    """Check uniqueness of an identifier column."""

    duplicate_count = int(df[column].duplicated(keep=False).sum())

    if duplicate_count > 0:
        result.add_error(
            f"{dataset_name}_{column}_unique",
            (
                f"{dataset_name} contains {duplicate_count:,} rows "
                f"with duplicate {column} values."
            ),
        )
        return

    result.add_pass(
        f"{dataset_name}_{column}_unique",
        f"{column} values are unique.",
    )


def _check_required_values(
    df: pd.DataFrame,
    columns: set[str],
    dataset_name: str,
    result: ValidationResult,
) -> None:
    """Check fields that must not be null or empty."""

    failures: list[str] = []

    for column in sorted(columns):
        if column not in df.columns:
            continue

        missing_count = int(_is_empty_string_series(df[column]).sum())

        if missing_count > 0:
            failures.append(f"{column}={missing_count:,}")

    if failures:
        result.add_error(
            f"{dataset_name}_required_values",
            (
                f"{dataset_name} contains missing/empty values "
                f"in required fields: {', '.join(failures)}."
            ),
        )
        return

    result.add_pass(
        f"{dataset_name}_required_values",
        "Required fields contain no missing or empty values.",
    )


def _check_optional_missing_values(
    df: pd.DataFrame,
    columns: set[str],
    dataset_name: str,
    result: ValidationResult,
) -> None:
    """Report missing values in optional/descriptive fields as warnings."""

    missing_fields: list[str] = []

    for column in sorted(columns):
        if column not in df.columns:
            continue

        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            missing_fields.append(f"{column}={missing_count:,}")

    if missing_fields:
        result.add_warning(
            f"{dataset_name}_optional_missing_values",
            (
                f"{dataset_name} contains missing values in "
                f"optional/descriptive fields: "
                f"{', '.join(missing_fields)}."
            ),
        )
    else:
        result.add_pass(
            f"{dataset_name}_optional_missing_values",
            "No missing values found in optional fields.",
        )


def _check_empty_text(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
    result: ValidationResult,
) -> None:
    """Check that a text column contains no empty values."""

    if column not in df.columns:
        return

    empty_count = int(_is_empty_string_series(df[column]).sum())

    if empty_count > 0:
        result.add_error(
            f"{dataset_name}_{column}_not_empty",
            (f"{dataset_name}.{column} contains {empty_count:,} empty or null values."),
        )
        return

    result.add_pass(
        f"{dataset_name}_{column}_not_empty",
        f"{column} contains no empty values.",
    )


def _check_foreign_key(
    child_df: pd.DataFrame,
    child_column: str,
    parent_df: pd.DataFrame,
    parent_column: str,
    check_name: str,
    result: ValidationResult,
) -> None:
    """Check that child values exist in the parent dataframe."""

    child_values = set(child_df[child_column].dropna().unique())
    parent_values = set(parent_df[parent_column].dropna().unique())

    missing_values = child_values - parent_values

    if missing_values:
        result.add_error(
            check_name,
            (
                f"{len(missing_values):,} {child_column} values "
                f"do not exist in {parent_column}."
            ),
        )
        return

    result.add_pass(
        check_name,
        "All foreign-key values exist in the parent dataset.",
    )


def _check_valid_labels(
    labels: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """Check that all relevance labels are valid."""

    invalid_mask = ~labels["label"].isin(VALID_LABELS)
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        invalid_values = sorted(
            labels.loc[invalid_mask, "label"]
            .astype("string")
            .dropna()
            .unique()
            .tolist()
        )

        result.add_error(
            "labels_valid",
            (
                f"Found {invalid_count:,} invalid labels. "
                f"Invalid values: {invalid_values}."
            ),
        )
        return

    result.add_pass(
        "labels_valid",
        f"All labels belong to {sorted(VALID_LABELS)}.",
    )


def _check_duplicate_judgment_ids(
    labels: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """Check uniqueness of annotation IDs."""

    duplicate_count = int(labels["id"].duplicated(keep=False).sum())

    if duplicate_count > 0:
        result.add_error(
            "judgment_id_unique",
            (f"Found {duplicate_count:,} rows with duplicate judgment IDs."),
        )
        return

    result.add_pass(
        "judgment_id_unique",
        "Judgment IDs are unique.",
    )


def _check_duplicate_query_product_pairs(
    labels: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Detect duplicate query-product judgments.

    Duplicate pairs are reported as warnings because the raw WANDS
    annotation file contains repeated query-product pairs. They must
    be investigated and resolved during data cleaning rather than
    silently removed during validation.
    """

    duplicate_mask = labels.duplicated(
        subset=["query_id", "product_id"],
        keep=False,
    )

    duplicate_rows = int(duplicate_mask.sum())

    if duplicate_rows == 0:
        result.add_pass(
            "query_product_pair_unique",
            "No duplicate query-product pairs found.",
        )
        return

    duplicate_pairs = (
        labels.loc[duplicate_mask, ["query_id", "product_id"]]
        .drop_duplicates()
        .shape[0]
    )

    grouped = (
        labels.loc[duplicate_mask]
        .groupby(["query_id", "product_id"])["label"]
        .nunique()
    )

    conflicting_pairs = int((grouped > 1).sum())

    result.add_warning(
        "query_product_pair_unique",
        (
            f"Found {duplicate_rows:,} rows belonging to "
            f"{duplicate_pairs:,} duplicate query-product pairs."
        ),
    )

    if conflicting_pairs > 0:
        result.add_error(
            "query_product_pair_label_consistency",
            (
                f"Found {conflicting_pairs:,} query-product pairs "
                "with conflicting relevance labels."
            ),
        )
    else:
        result.add_pass(
            "query_product_pair_label_consistency",
            "All duplicate query-product pairs have consistent labels.",
        )


def validate_queries(
    queries: pd.DataFrame,
) -> ValidationResult:
    """Validate WANDS query data."""

    result = ValidationResult()

    columns_ok = _check_required_columns(
        queries,
        REQUIRED_QUERY_COLUMNS,
        "query",
        result,
    )

    if not columns_ok:
        return result

    _check_empty_dataframe(
        queries,
        "query",
        result,
    )

    _check_unique_ids(
        queries,
        "query_id",
        "query",
        result,
    )

    _check_required_values(
        queries,
        QUERY_REQUIRED_VALUE_COLUMNS,
        "query",
        result,
    )

    _check_empty_text(
        queries,
        "query",
        "query",
        result,
    )

    # query_class is metadata and is intentionally not required
    # for the initial ranking pipeline.
    if "query_class" in queries.columns:
        missing_query_class = int(queries["query_class"].isna().sum())

        if missing_query_class > 0:
            result.add_warning(
                "query_query_class_missing",
                (
                    f"query_class contains {missing_query_class:,} "
                    "missing values. This field is optional and is "
                    "not used by the initial ranking model."
                ),
            )
        else:
            result.add_pass(
                "query_query_class_missing",
                "query_class contains no missing values.",
            )

    return result


def validate_products(
    products: pd.DataFrame,
) -> ValidationResult:
    """Validate WANDS product data."""

    result = ValidationResult()

    columns_ok = _check_required_columns(
        products,
        REQUIRED_PRODUCT_COLUMNS,
        "product",
        result,
    )

    if not columns_ok:
        return result

    _check_empty_dataframe(
        products,
        "product",
        result,
    )

    _check_unique_ids(
        products,
        "product_id",
        "product",
        result,
    )

    _check_required_values(
        products,
        PRODUCT_REQUIRED_VALUE_COLUMNS,
        "product",
        result,
    )

    _check_empty_text(
        products,
        "product_name",
        "product",
        result,
    )

    optional_columns = set(products.columns) - PRODUCT_REQUIRED_VALUE_COLUMNS

    _check_optional_missing_values(
        products,
        optional_columns,
        "product",
        result,
    )

    return result


def validate_labels(
    labels: pd.DataFrame,
) -> ValidationResult:
    """Validate WANDS relevance judgments."""

    result = ValidationResult()

    columns_ok = _check_required_columns(
        labels,
        REQUIRED_LABEL_COLUMNS,
        "label",
        result,
    )

    if not columns_ok:
        return result

    _check_empty_dataframe(
        labels,
        "label",
        result,
    )

    _check_required_values(
        labels,
        LABEL_REQUIRED_VALUE_COLUMNS,
        "label",
        result,
    )

    _check_valid_labels(
        labels,
        result,
    )

    _check_duplicate_judgment_ids(
        labels,
        result,
    )

    _check_duplicate_query_product_pairs(
        labels,
        result,
    )

    return result


def validate_wands(
    queries: pd.DataFrame,
    products: pd.DataFrame,
    labels: pd.DataFrame,
) -> ValidationResult:
    """Run all WANDS validation checks."""

    result = ValidationResult()

    query_result = validate_queries(queries)
    product_result = validate_products(products)
    label_result = validate_labels(labels)

    result.errors.extend(query_result.errors)
    result.errors.extend(product_result.errors)
    result.errors.extend(label_result.errors)

    result.warnings.extend(query_result.warnings)
    result.warnings.extend(product_result.warnings)
    result.warnings.extend(label_result.warnings)

    result.checks.extend(query_result.checks)
    result.checks.extend(product_result.checks)
    result.checks.extend(label_result.checks)

    # Foreign-key validation is only meaningful if the relevant
    # identifier columns exist.
    if "query_id" in labels.columns and "query_id" in queries.columns:
        _check_foreign_key(
            labels,
            "query_id",
            queries,
            "query_id",
            "foreign_key_label_query_id",
            result,
        )

    if "product_id" in labels.columns and "product_id" in products.columns:
        _check_foreign_key(
            labels,
            "product_id",
            products,
            "product_id",
            "foreign_key_label_product_id",
            result,
        )

    return result


def raise_if_invalid(
    result: ValidationResult,
) -> None:
    """Raise DataValidationError if validation failed."""

    if not result.is_valid:
        raise DataValidationError(result.errors)


def validation_summary(
    result: ValidationResult,
) -> dict[str, Any]:
    """Return a JSON-serializable validation summary."""

    passed = sum(1 for check in result.checks if check["status"] == "PASS")

    failed = sum(1 for check in result.checks if check["status"] == "FAIL")

    warnings = sum(1 for check in result.checks if check["status"] == "WARN")

    return {
        "valid": result.is_valid,
        "total_checks": len(result.checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "warning_checks": warnings,
        "errors": result.errors,
        "warnings": result.warnings,
        "checks": result.checks,
    }


def validate_wands_files(
    data_dir: str | Path = "data/raw/wands",
) -> ValidationResult:
    """Load and validate the three raw WANDS files."""

    data_dir = Path(data_dir)

    product_path = data_dir / "product.csv"
    query_path = data_dir / "query.csv"
    label_path = data_dir / "label.csv"

    products = pd.read_csv(product_path, sep="\t")
    queries = pd.read_csv(query_path, sep="\t")
    labels = pd.read_csv(label_path, sep="\t")

    return validate_wands(
        queries=queries,
        products=products,
        labels=labels,
    )


def main() -> int:
    """CLI entry point."""

    print("=" * 60)
    print("RelevanceFlow — WANDS Data Validation")
    print("=" * 60)

    result = validate_wands_files()

    summary = validation_summary(result)

    print(f"Valid: {summary['valid']}")
    print(f"Total checks: {summary['total_checks']}")
    print(f"Passed checks: {summary['passed_checks']}")
    print(f"Failed checks: {summary['failed_checks']}")
    print(f"Warning checks: {summary['warning_checks']}")

    print("\nChecks:")

    for check in result.checks:
        status = check["status"]
        name = check["check"]

        print(f"  [{status}] {name}")

    if result.warnings:
        print("\nWarnings:")

        for warning in result.warnings:
            print(f"  - {warning}")

    if result.errors:
        print("\nErrors:")

        for error in result.errors:
            print(f"  - {error}")

    print()

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
