from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


@dataclass(frozen=True)
class LTRConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42


class LightGBMLTR:
    """LightGBM LambdaRank model for query-product relevance ranking."""

    TARGET_COLUMN: str = "relevance_score"

    EXCLUDED_COLUMNS: ClassVar[set[str]] = {
        "query_id",
        "product_id",
        "relevance_score",
        "label",
    }

    def __init__(
        self,
        config: LTRConfig | None = None,
    ) -> None:
        self.config = config or LTRConfig()
        self.model: LGBMRanker | None = None
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

    @staticmethod
    def _prepare_group(
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        if "query_id" not in dataframe.columns:
            raise ValueError("Missing query_id column.")

        query_counts = dataframe.groupby("query_id", sort=False).size().to_numpy()

        if len(query_counts) == 0:
            raise ValueError("No query groups found.")

        if np.sum(query_counts) != len(dataframe):
            raise ValueError("Invalid query group sizes.")

        return query_counts.astype(np.int32)

    def fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame | None = None,
    ) -> LightGBMLTR:
        if self.TARGET_COLUMN not in train.columns:
            raise ValueError(f"Missing target column: {self.TARGET_COLUMN}")

        if "query_id" not in train.columns:
            raise ValueError("Missing query_id column.")

        if "product_id" not in train.columns:
            raise ValueError("Missing product_id column.")

        train = train.sort_values(["query_id", "product_id"]).reset_index(drop=True)

        X_train = self._prepare_features(train)
        y_train = train[self.TARGET_COLUMN].astype(float)
        group_train = self._prepare_group(train)

        self.feature_columns = X_train.columns.tolist()

        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
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
            eval_at=[5, 10, 20],
        )

        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "group": group_train,
        }

        if validation is not None:
            if "query_id" not in validation.columns:
                raise ValueError("Missing query_id column.")

            if "product_id" not in validation.columns:
                raise ValueError("Missing product_id column.")

            if self.TARGET_COLUMN not in validation.columns:
                raise ValueError(f"Missing target column: {self.TARGET_COLUMN}")

            validation = validation.sort_values(["query_id", "product_id"]).reset_index(
                drop=True
            )

            X_validation = validation[self.feature_columns]

            y_validation = validation[self.TARGET_COLUMN].astype(float)

            group_validation = self._prepare_group(validation)

            fit_kwargs["eval_set"] = [(X_validation, y_validation)]

            fit_kwargs["eval_group"] = [group_validation]

        self.model.fit(**fit_kwargs)

        return self

    def predict_scores(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        if not self.feature_columns:
            raise RuntimeError("Feature columns are not available.")

        missing_features = [
            column for column in self.feature_columns if column not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        X = dataframe[self.feature_columns]

        return self.model.predict(X)

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
