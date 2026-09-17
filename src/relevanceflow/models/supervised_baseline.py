from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


@dataclass(frozen=True)
class SupervisedBaselineConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42


class SupervisedBaseline:
    """LightGBM multiclass relevance classifier used as a ranking baseline."""

    TARGET_COLUMN: str = "relevance_score"

    EXCLUDED_COLUMNS: ClassVar[set[str]] = {
        "query_id",
        "product_id",
        "relevance_score",
        "label",
        "query_embedding_norm",
        "product_embedding_norm",
    }

    def __init__(
        self,
        config: SupervisedBaselineConfig | None = None,
    ) -> None:
        self.config = config or SupervisedBaselineConfig()
        self.model: LGBMClassifier | None = None
        self.feature_columns: list[str] = []

    def _prepare_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in self.EXCLUDED_COLUMNS
        ]

        if not feature_columns:
            raise ValueError("No model features were found.")

        features = dataframe[feature_columns].copy()

        if features.isnull().any().any():
            raise ValueError("Feature matrix contains missing values.")

        non_numeric = features.select_dtypes(exclude=np.number).columns.tolist()

        if non_numeric:
            raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

        return features

    def fit(
        self,
        train: pd.DataFrame,
    ) -> SupervisedBaseline:
        if self.TARGET_COLUMN not in train.columns:
            raise ValueError(f"Missing target column: {self.TARGET_COLUMN}")

        if "query_id" not in train.columns:
            raise ValueError("Missing query_id column.")

        X = self._prepare_features(train)
        y = train[self.TARGET_COLUMN].astype(int)

        if not set(y.unique()).issubset({0, 1, 2}):
            raise ValueError(f"Unexpected relevance labels: {sorted(y.unique())}")

        self.feature_columns = X.columns.tolist()

        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            verbosity=-1,
        )

        self.model.fit(X, y)

        return self

    def predict_scores(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        X = dataframe[self.feature_columns]

        probabilities = self.model.predict_proba(X)

        # Probability of higher relevance.
        # Class 0 = Irrelevant
        # Class 1 = Partial
        # Class 2 = Exact
        scores = probabilities[:, 1] + probabilities[:, 2]

        return scores

    def rank_candidates(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        if "query_id" not in dataframe.columns:
            raise ValueError("Missing query_id column.")

        if "product_id" not in dataframe.columns:
            raise ValueError("Missing product_id column.")

        result = dataframe[["query_id", "product_id"]].copy()

        result["score"] = self.predict_scores(dataframe)

        result = result.sort_values(
            ["query_id", "score", "product_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        result["rank"] = result.groupby("query_id").cumcount() + 1

        return result
