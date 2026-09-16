from pathlib import Path

from relevanceflow.data.split import (
    SplitConfig,
    create_query_level_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "wands" / "judgments.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "wands"


def main() -> None:
    config = SplitConfig(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    splits = create_query_level_split(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        config=config,
    )

    print("=" * 60)
    print("QUERY-LEVEL DATASET SPLIT")
    print("=" * 60)

    for split_name, dataframe in splits.items():
        query_count = dataframe["query_id"].nunique()
        row_count = len(dataframe)

        print(
            f"{split_name:12s} | "
            f"queries={query_count:3d} | "
            f"judgments={row_count:6d}"
        )

    print("=" * 60)

    train_queries = set(splits["train"]["query_id"])
    validation_queries = set(splits["validation"]["query_id"])
    test_queries = set(splits["test"]["query_id"])

    print(
        f"Total unique queries: {len(train_queries | validation_queries | test_queries)}"
    )
    print(f"Train/Validation overlap: {len(train_queries & validation_queries)}")
    print(f"Train/Test overlap:        {len(train_queries & test_queries)}")
    print(f"Validation/Test overlap:   {len(validation_queries & test_queries)}")


if __name__ == "__main__":
    main()
