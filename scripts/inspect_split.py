from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "wands"


def main() -> None:
    for split_name in ("train", "validation", "test"):
        path = DATA_DIR / f"{split_name}.parquet"

        dataframe = pd.read_parquet(path)

        queries = sorted(dataframe["query_id"].unique())

        print(f"\n{split_name.upper()}")
        print("-" * 40)
        print(f"Unique queries: {len(queries)}")
        print(f"Judgments: {len(dataframe)}")
        print(f"First query IDs: {queries[:10]}")
        print(f"Last query IDs: {queries[-10:]}")


if __name__ == "__main__":
    main()
