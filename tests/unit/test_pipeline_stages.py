from pathlib import Path

import pandas as pd

from relevanceflow.data.pipeline_stages import (
    load_processed_judgments,
)


def test_load_processed_judgments(tmp_path: Path) -> None:
    processed_dir = tmp_path / "wands"
    processed_dir.mkdir()

    dataframe = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": [10, 20],
            "relevance_score": [2, 0],
        }
    )

    dataframe.to_parquet(
        processed_dir / "judgments.parquet",
        index=False,
    )

    loaded = load_processed_judgments(processed_dir)

    assert len(loaded) == 2
    assert list(loaded.columns) == [
        "query_id",
        "product_id",
        "relevance_score",
    ]
