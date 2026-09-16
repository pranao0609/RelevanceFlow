from __future__ import annotations

import json
import time
from pathlib import Path

import optuna
import pandas as pd
from lightgbm import LGBMRanker

from relevanceflow.evaluation.ranking_metrics import ndcg_at_k

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ltr_optimization"

TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_PATH = FEATURE_DIR / "validation_features.parquet"


TARGET_COLUMN = "relevance_score"

EXCLUDED_COLUMNS = {
    "query_id",
    "product_id",
    "relevance_score",
    "label",
}


def prepare_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = [
        column for column in dataframe.columns if column not in EXCLUDED_COLUMNS
    ]

    return dataframe[feature_columns]


def prepare_groups(
    dataframe: pd.DataFrame,
):
    return dataframe.groupby("query_id", sort=False).size().to_numpy()


def evaluate_ndcg_at_10(
    predictions: pd.Series,
    dataframe: pd.DataFrame,
) -> float:
    evaluation = dataframe[["query_id", "product_id", TARGET_COLUMN]].copy()

    evaluation["score"] = predictions.to_numpy()

    evaluation = evaluation.sort_values(
        [
            "query_id",
            "score",
            "product_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    scores = []

    for _, group in evaluation.groupby(
        "query_id",
        sort=False,
    ):
        relevance = group[TARGET_COLUMN].to_numpy()

        scores.append(
            ndcg_at_k(
                relevance,
                10,
            )
        )

    return sum(scores) / len(scores)


def create_model(
    trial: optuna.Trial,
) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=trial.suggest_int(
            "n_estimators",
            200,
            800,
            step=100,
        ),
        learning_rate=trial.suggest_float(
            "learning_rate",
            0.01,
            0.10,
            log=True,
        ),
        num_leaves=trial.suggest_int(
            "num_leaves",
            15,
            63,
        ),
        max_depth=trial.suggest_int(
            "max_depth",
            3,
            10,
        ),
        min_child_samples=trial.suggest_int(
            "min_child_samples",
            10,
            100,
        ),
        subsample=trial.suggest_float(
            "subsample",
            0.7,
            1.0,
        ),
        colsample_bytree=trial.suggest_float(
            "colsample_bytree",
            0.7,
            1.0,
        ),
        reg_alpha=trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),
        reg_lambda=trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        ),
        random_state=42,
        verbosity=-1,
        eval_at=[5, 10, 20],
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading feature matrices...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    train = train.sort_values(["query_id", "product_id"]).reset_index(drop=True)

    validation = validation.sort_values(["query_id", "product_id"]).reset_index(
        drop=True
    )

    X_train = prepare_features(train)
    y_train = train[TARGET_COLUMN].astype(float)
    group_train = prepare_groups(train)

    X_validation = prepare_features(validation)
    validation[TARGET_COLUMN].astype(float)
    group_validation = prepare_groups(validation)

    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")
    print(f"Training query groups: " f"{len(group_train)}")
    print(f"Validation query groups: " f"{len(group_validation)}")

    def objective(
        trial: optuna.Trial,
    ) -> float:
        {
            "n_estimators": trial.params.get("n_estimators"),
        }

        print(f"\nTrial {trial.number} " f"starting...")

        start_time = time.perf_counter()

        model = create_model(trial)

        model.fit(
            X_train,
            y_train,
            group=group_train,
        )

        predictions = model.predict(X_validation)

        validation_ndcg = evaluate_ndcg_at_10(
            pd.Series(predictions),
            validation,
        )

        elapsed = time.perf_counter() - start_time

        trial.set_user_attr(
            "training_time_seconds",
            elapsed,
        )

        print(
            f"Trial {trial.number}: "
            f"NDCG@10="
            f"{validation_ndcg:.6f}, "
            f"time="
            f"{elapsed:.2f}s"
        )

        return validation_ndcg

    study = optuna.create_study(
        direction="maximize",
        study_name="relevanceflow_ltr",
    )

    print("\nStarting Optuna optimization...")
    print("Trials: 30")
    print("Objective: validation NDCG@10")

    optimization_start = time.perf_counter()

    study.optimize(
        objective,
        n_trials=30,
    )

    optimization_time = time.perf_counter() - optimization_start

    print("\nOptimization complete.")

    print(f"Best validation NDCG@10: " f"{study.best_value:.6f}")

    print("\nBest parameters:")

    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_trial = study.best_trial

    best_params_path = OUTPUT_DIR / "best_params.json"

    with open(
        best_params_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "objective": "validation_ndcg@10",
                "best_validation_ndcg@10": (study.best_value),
                "best_trial": best_trial.number,
                "parameters": study.best_params,
            },
            file,
            indent=2,
        )

    summary_path = OUTPUT_DIR / "optimization_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "study_name": ("relevanceflow_ltr"),
                "n_trials": len(study.trials),
                "best_trial": (best_trial.number),
                "best_validation_ndcg@10": (study.best_value),
                "optimization_time_seconds": (optimization_time),
                "train_queries": (len(group_train)),
                "validation_queries": (len(group_validation)),
            },
            file,
            indent=2,
        )

    trials_path = OUTPUT_DIR / "trials.csv"

    study.trials_dataframe().to_csv(
        trials_path,
        index=False,
    )

    print(
        f"\nSaved:" f"\n  {best_params_path}" f"\n  {summary_path}" f"\n  {trials_path}"
    )


if __name__ == "__main__":
    main()
