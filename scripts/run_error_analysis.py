from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker

from relevanceflow.evaluation.ranking_metrics import ndcg_at_k

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "error_analysis"

TRAIN_FEATURES = FEATURE_DIR / "train_features.parquet"

VALIDATION_FEATURES = FEATURE_DIR / "validation_features.parquet"

TEST_FEATURES = FEATURE_DIR / "test_features.parquet"

QUERIES_PATH = DATA_DIR / "processed" / "wands" / "queries.parquet"

PRODUCTS_PATH = DATA_DIR / "processed" / "wands" / "products.parquet"

BEST_PARAMS_PATH = (
    PROJECT_ROOT / "experiments" / "ltr_optimization" / "best_params.json"
)

TARGET_COLUMN = "relevance_score"

EXCLUDED_FEATURE_COLUMNS = {
    "query_id",
    "product_id",
    "relevance_score",
    "label",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    print("Loading feature matrices...")

    train = pd.read_parquet(TRAIN_FEATURES)

    validation = pd.read_parquet(VALIDATION_FEATURES)

    test = pd.read_parquet(TEST_FEATURES)

    queries = pd.read_parquet(QUERIES_PATH)

    products = pd.read_parquet(PRODUCTS_PATH)

    if "query" not in test.columns:
        test = test.merge(
            queries[["query_id", "query"]],
            on="query_id",
            how="left",
            validate="many_to_one",
        )

    print(f"Train:       {train.shape}")
    print(f"Validation:  {validation.shape}")
    print(f"Test:        {test.shape}")
    print(f"Queries:     {queries.shape}")

    return (
        train,
        validation,
        test,
        queries,
        products,
    )


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_features(
    dataframe: pd.DataFrame,
):
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in EXCLUDED_FEATURE_COLUMNS and column not in {"query"}
    ]

    if not feature_columns:
        raise ValueError("No model features found.")

    return dataframe[feature_columns].astype(float)


# ---------------------------------------------------------------------------
# Group preparation
# ---------------------------------------------------------------------------


def build_groups(
    dataframe: pd.DataFrame,
):
    if "query_id" not in dataframe.columns:
        raise ValueError("query_id is required.")

    return (
        dataframe.groupby(
            "query_id",
            sort=False,
        )
        .size()
        .tolist()
    )


# ---------------------------------------------------------------------------
# LTR parameters
# ---------------------------------------------------------------------------


def load_ltr_parameters():
    with open(
        BEST_PARAMS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if "parameters" not in config:
        raise ValueError("best_params.json does not contain " "'parameters'.")

    return config["parameters"]


# ---------------------------------------------------------------------------
# Train LTR
# ---------------------------------------------------------------------------


def train_ltr(
    train: pd.DataFrame,
):
    print()
    print("Training LightGBM LTR...")

    parameters = load_ltr_parameters()

    feature_columns = [
        column for column in train.columns if column not in EXCLUDED_FEATURE_COLUMNS
    ]

    X_train = train[feature_columns].astype(float)

    y_train = train[TARGET_COLUMN].astype(int)

    sorted_train = train.sort_values(["query_id", "product_id"]).reset_index(drop=True)

    X_train = sorted_train[feature_columns].astype(float)

    y_train = sorted_train[TARGET_COLUMN].astype(int)

    groups = build_groups(sorted_train)

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5, 10, 20],
        verbosity=-1,
        n_jobs=-1,
        **parameters,
    )

    model.fit(
        X_train,
        y_train,
        group=groups,
    )

    return model, feature_columns


# ---------------------------------------------------------------------------
# Train supervised baseline
# ---------------------------------------------------------------------------


def train_supervised_baseline(
    train: pd.DataFrame,
):
    print()
    print("Training supervised baseline...")

    feature_columns = [
        column for column in train.columns if column not in EXCLUDED_FEATURE_COLUMNS
    ]

    X_train = train[feature_columns].astype(float)

    y_train = train[TARGET_COLUMN].astype(int)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model, feature_columns


# ---------------------------------------------------------------------------
# Generate model scores
# ---------------------------------------------------------------------------


def generate_scores(
    test: pd.DataFrame,
    ltr_model,
    ltr_features: list[str],
    supervised_model,
    supervised_features: list[str],
):
    print()
    print("Generating model scores...")

    scores = {}

    # Existing retrieval scores.
    scores["TF-IDF"] = test["tfidf_similarity"].to_numpy(dtype=float)

    scores["BM25"] = test["bm25_score"].to_numpy(dtype=float)

    # Supervised baseline.
    X_supervised = test[supervised_features].astype(float)

    probabilities = supervised_model.predict_proba(X_supervised)

    # P(Partial) + P(Exact)
    scores["Supervised"] = probabilities[:, 1:].sum(axis=1)

    # LTR.
    X_ltr = test[ltr_features].astype(float)

    scores["LTR"] = ltr_model.predict(X_ltr)

    return scores


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------


def create_rankings(
    test: pd.DataFrame,
    scores: dict[str, np.ndarray],
):
    print()
    print("Creating query-level rankings...")

    required_columns = [
        "query_id",
        "product_id",
        "query",
        TARGET_COLUMN,
        "exact_match",
        "bm25_score",
        "tfidf_similarity",
    ]

    missing = [column for column in required_columns if column not in test.columns]

    if missing:
        raise ValueError("Missing columns required for " f"error analysis: {missing}")

    rankings = test[required_columns].copy()

    # Preserve the original retrieval scores.
    rankings["tfidf_score"] = rankings["tfidf_similarity"]

    # Add generated model scores.
    rankings["bm25_model_score"] = scores["BM25"]

    rankings["supervised_score"] = scores["Supervised"]

    rankings["ltr_score"] = scores["LTR"]

    # Model-specific ranking columns.
    model_score_columns = {
        "tfidf": "tfidf_score",
        "bm25": "bm25_model_score",
        "supervised": "supervised_score",
        "ltr": "ltr_score",
    }

    for model_name, score_column in model_score_columns.items():
        rank_column = f"{model_name}_rank"

        rankings[rank_column] = rankings.groupby("query_id")[score_column].rank(
            ascending=False,
            method="first",
        )

    return rankings


# ---------------------------------------------------------------------------
# Canonical NDCG
# ---------------------------------------------------------------------------


def ndcg_from_relevance(
    relevance,
    k: int = 10,
) -> float:
    """
    Use the project's canonical NDCG implementation.
    """

    if len(relevance) == 0:
        return 0.0

    return float(
        ndcg_at_k(
            relevance=relevance,
            k=k,
        )
    )


# ---------------------------------------------------------------------------
# Query-level analysis
# ---------------------------------------------------------------------------


def analyze_queries(
    rankings: pd.DataFrame,
):
    rows = []

    model_names = [
        "tfidf",
        "bm25",
        "supervised",
        "ltr",
    ]

    for query_id, group in rankings.groupby("query_id"):
        query = group["query"].iloc[0]

        query_length = len(str(query).split())

        if query_length <= 2:
            length_bucket = "short"
        elif query_length <= 5:
            length_bucket = "medium"
        else:
            length_bucket = "long"

        exact_count = int(group["exact_match"].sum())

        if exact_count > 0:
            query_type = "exact_product_match"
        elif query_length <= 2:
            query_type = "short"
        elif query_length >= 6:
            query_type = "long"
        else:
            query_type = "general"

        model_ndcg = {}

        for model_name in model_names:
            score_column = f"{model_name}_score"

            ordered = group.sort_values(
                [
                    score_column,
                    "product_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )

            relevance = ordered[TARGET_COLUMN].tolist()

            model_ndcg[model_name] = ndcg_from_relevance(
                relevance,
                k=10,
            )

        ltr_minus_bm25 = model_ndcg["ltr"] - model_ndcg["bm25"]

        ltr_minus_tfidf = model_ndcg["ltr"] - model_ndcg["tfidf"]

        ltr_minus_supervised = model_ndcg["ltr"] - model_ndcg["supervised"]

        if ltr_minus_bm25 > 1e-12:
            relation = "LTR_better"
        elif ltr_minus_bm25 < -1e-12:
            relation = "BM25_better"
        else:
            relation = "tie"

        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "query_length_tokens": query_length,
                "query_length_bucket": length_bucket,
                "query_type": query_type,
                "candidate_count": len(group),
                "relevant_count": int((group[TARGET_COLUMN] > 0).sum()),
                "exact_count": int((group[TARGET_COLUMN] == 2).sum()),
                "partial_count": int((group[TARGET_COLUMN] == 1).sum()),
                "tfidf_ndcg@10": model_ndcg["tfidf"],
                "bm25_ndcg@10": model_ndcg["bm25"],
                "supervised_ndcg@10": model_ndcg["supervised"],
                "ltr_ndcg@10": model_ndcg["ltr"],
                "ltr_minus_bm25": ltr_minus_bm25,
                "ltr_minus_tfidf": ltr_minus_tfidf,
                "ltr_minus_supervised": ltr_minus_supervised,
                "bm25_relation": relation,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top success cases
# ---------------------------------------------------------------------------


def create_success_report(
    query_report: pd.DataFrame,
):
    success = query_report[query_report["ltr_minus_bm25"] > 0].copy()

    success = success.sort_values(
        "ltr_minus_bm25",
        ascending=False,
    )

    columns = [
        "query_id",
        "query",
        "query_length_tokens",
        "query_length_bucket",
        "query_type",
        "candidate_count",
        "relevant_count",
        "exact_count",
        "partial_count",
        "bm25_ndcg@10",
        "ltr_ndcg@10",
        "ltr_minus_bm25",
    ]

    return success[columns].head(25)


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------


def create_failure_report(
    rankings: pd.DataFrame,
):
    rows = []

    for query_id, group in rankings.groupby("query_id"):
        ordered = group.sort_values(
            [
                "ltr_score",
                "product_id",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(drop=True)

        top10 = ordered.head(10)

        # Irrelevant products incorrectly ranked
        # inside LTR top 10.
        for position, (_, row) in enumerate(
            top10.iterrows(),
            start=1,
        ):
            if row[TARGET_COLUMN] == 0:
                rows.append(
                    {
                        "failure_type": ("irrelevant_in_top10"),
                        "query_id": query_id,
                        "query": row["query"],
                        "product_id": row["product_id"],
                        "relevance_score": row[TARGET_COLUMN],
                        "ltr_rank": position,
                        "ltr_score": row["ltr_score"],
                        "bm25_rank": row["bm25_rank"],
                        "tfidf_rank": row["tfidf_rank"],
                    }
                )

        # Relevant products below top 10.
        below_top10 = ordered.iloc[10:]

        for _, row in below_top10.iterrows():
            if row[TARGET_COLUMN] > 0:
                rows.append(
                    {
                        "failure_type": ("relevant_below_top10"),
                        "query_id": query_id,
                        "query": row["query"],
                        "product_id": row["product_id"],
                        "relevance_score": row[TARGET_COLUMN],
                        "ltr_rank": row["ltr_rank"],
                        "ltr_score": row["ltr_score"],
                        "bm25_rank": row["bm25_rank"],
                        "tfidf_rank": row["tfidf_rank"],
                    }
                )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "failure_type",
            "query_id",
            "ltr_rank",
        ]
    )


# ---------------------------------------------------------------------------
# BM25 vs LTR
# ---------------------------------------------------------------------------


def create_bm25_vs_ltr_failures(
    query_report: pd.DataFrame,
):
    failures = query_report[query_report["ltr_minus_bm25"] < 0].copy()

    return failures.sort_values(
        "ltr_minus_bm25",
        ascending=True,
    )


# ---------------------------------------------------------------------------
# Query length analysis
# ---------------------------------------------------------------------------


def create_query_length_analysis(
    query_report: pd.DataFrame,
):
    rows = []

    for bucket, group in query_report.groupby("query_length_bucket"):
        rows.append(
            {
                "query_length_bucket": bucket,
                "queries": len(group),
                "tfidf_ndcg@10": group["tfidf_ndcg@10"].mean(),
                "bm25_ndcg@10": group["bm25_ndcg@10"].mean(),
                "supervised_ndcg@10": group["supervised_ndcg@10"].mean(),
                "ltr_ndcg@10": group["ltr_ndcg@10"].mean(),
                "ltr_minus_bm25": group["ltr_minus_bm25"].mean(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Query type analysis
# ---------------------------------------------------------------------------


def create_query_type_analysis(
    query_report: pd.DataFrame,
):
    rows = []

    for query_type, group in query_report.groupby("query_type"):
        rows.append(
            {
                "query_type": query_type,
                "queries": len(group),
                "tfidf_ndcg@10": group["tfidf_ndcg@10"].mean(),
                "bm25_ndcg@10": group["bm25_ndcg@10"].mean(),
                "supervised_ndcg@10": group["supervised_ndcg@10"].mean(),
                "ltr_ndcg@10": group["ltr_ndcg@10"].mean(),
                "ltr_minus_bm25": group["ltr_minus_bm25"].mean(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Category analysis
# ---------------------------------------------------------------------------


def extract_top_level_category(
    value,
):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if not value:
        return "Unknown"

    # WANDS category hierarchy is hierarchical.
    # Handle common delimiters conservatively.
    for delimiter in [
        " > ",
        " / ",
        "/",
        ">",
        "|",
    ]:
        if delimiter in value:
            return value.split(delimiter)[0].strip() or "Unknown"

    return value


def create_category_analysis(
    rankings: pd.DataFrame,
    products: pd.DataFrame,
):
    category_column = "category hierarchy"

    if category_column not in products.columns:
        raise ValueError("Missing product category column: " f"{category_column}")

    product_categories = products[
        [
            "product_id",
            category_column,
        ]
    ].copy()

    product_categories["category"] = product_categories[category_column].apply(
        extract_top_level_category
    )

    product_categories = product_categories[
        [
            "product_id",
            "category",
        ]
    ].drop_duplicates("product_id")

    data = rankings.merge(
        product_categories,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    data["category"] = data["category"].fillna("Unknown")

    rows = []

    for category, group in data.groupby("category"):
        query_scores_ltr = []
        query_scores_bm25 = []

        for _, query_group in group.groupby("query_id"):
            ltr_ordered = query_group.sort_values(
                [
                    "ltr_score",
                    "product_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )

            bm25_ordered = query_group.sort_values(
                [
                    "bm25_score",
                    "product_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )

            query_scores_ltr.append(
                ndcg_from_relevance(
                    ltr_ordered[TARGET_COLUMN].tolist(),
                    k=10,
                )
            )

            query_scores_bm25.append(
                ndcg_from_relevance(
                    bm25_ordered[TARGET_COLUMN].tolist(),
                    k=10,
                )
            )

        if not query_scores_ltr:
            continue

        rows.append(
            {
                "category": category,
                "queries": group["query_id"].nunique(),
                "candidates": len(group),
                "relevant_candidates": int((group[TARGET_COLUMN] > 0).sum()),
                "ltr_ndcg@10": float(np.mean(query_scores_ltr)),
                "bm25_ndcg@10": float(np.mean(query_scores_bm25)),
                "ltr_minus_bm25": float(
                    np.mean(np.array(query_scores_ltr) - np.array(query_scores_bm25))
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "queries",
            "ltr_minus_bm25",
        ],
        ascending=[
            False,
            False,
        ],
    )


# ---------------------------------------------------------------------------
# Ambiguity analysis
# ---------------------------------------------------------------------------


def create_ambiguous_query_analysis(
    rankings: pd.DataFrame,
    products: pd.DataFrame,
):
    """
    Operational ambiguity heuristic:

    A query is considered ambiguous when its relevant
    products span at least two top-level product categories.

    This is NOT a WANDS ground-truth ambiguity label.
    """

    category_column = "category hierarchy"

    product_categories = products[
        [
            "product_id",
            category_column,
        ]
    ].copy()

    product_categories["category"] = product_categories[category_column].apply(
        extract_top_level_category
    )

    product_categories = product_categories[
        [
            "product_id",
            "category",
        ]
    ].drop_duplicates("product_id")

    data = rankings.merge(
        product_categories,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    data["category"] = data["category"].fillna("Unknown")

    rows = []

    for query_id, group in data.groupby("query_id"):
        relevant = group[group[TARGET_COLUMN] > 0]

        categories = sorted(relevant["category"].dropna().unique().tolist())

        if len(categories) < 2:
            continue

        ltr_ordered = group.sort_values(
            [
                "ltr_score",
                "product_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        bm25_ordered = group.sort_values(
            [
                "bm25_score",
                "product_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        ltr_ndcg = ndcg_from_relevance(
            ltr_ordered[TARGET_COLUMN].tolist(),
            k=10,
        )

        bm25_ndcg = ndcg_from_relevance(
            bm25_ordered[TARGET_COLUMN].tolist(),
            k=10,
        )

        rows.append(
            {
                "query_id": query_id,
                "query": group["query"].iloc[0],
                "relevant_category_count": len(categories),
                "relevant_categories": (" | ".join(categories)),
                "candidate_count": len(group),
                "relevant_count": int((group[TARGET_COLUMN] > 0).sum()),
                "bm25_ndcg@10": bm25_ndcg,
                "ltr_ndcg@10": ltr_ndcg,
                "ltr_minus_bm25": (ltr_ndcg - bm25_ndcg),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values("ltr_minus_bm25")


# ---------------------------------------------------------------------------
# Confusion-style relevance analysis
# ---------------------------------------------------------------------------


def create_confusion_analysis(
    rankings: pd.DataFrame,
):
    rows = []

    for model_name in [
        "tfidf",
        "bm25",
        "supervised",
        "ltr",
    ]:
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for _, group in rankings.groupby("query_id"):
            ordered = group.sort_values(
                [
                    f"{model_name}_score",
                    "product_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )

            predicted_relevant = set(ordered.head(10)["product_id"])

            actual_relevant = set(group[group[TARGET_COLUMN] > 0]["product_id"])

            all_products = set(group["product_id"])

            tp += len(predicted_relevant & actual_relevant)

            fp += len(predicted_relevant - actual_relevant)

            fn += len(actual_relevant - predicted_relevant)

            tn += len(all_products - predicted_relevant - actual_relevant)

        rows.append(
            {
                "model": model_name,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start_time = time.perf_counter()

    (
        train,
        _validation,
        test,
        _queries,
        products,
    ) = load_data()

    # Train LTR.
    ltr_model, ltr_features = train_ltr(train)

    # Train supervised baseline.
    (
        supervised_model,
        supervised_features,
    ) = train_supervised_baseline(train)

    # Generate scores.
    scores = generate_scores(
        test=test,
        ltr_model=ltr_model,
        ltr_features=ltr_features,
        supervised_model=supervised_model,
        supervised_features=supervised_features,
    )

    # Create rankings.
    rankings = create_rankings(
        test,
        scores,
    )

    print()
    print("=" * 75)
    print("RelevanceFlow — Stage 18 " "Model Evaluation & Error Analysis")
    print("=" * 75)

    test_queries = rankings["query_id"].nunique()

    test_candidates = len(rankings)

    print(f"\nTest queries: {test_queries}")
    print(f"Test candidates: {test_candidates}")

    # Query report.
    query_report = analyze_queries(rankings)

    print("\nAverage query-level NDCG@10:")

    print("  TF-IDF:       " f"{query_report['tfidf_ndcg@10'].mean():.6f}")

    print("  BM25:         " f"{query_report['bm25_ndcg@10'].mean():.6f}")

    print("  Supervised:   " f"{query_report['supervised_ndcg@10'].mean():.6f}")

    print("  LightGBM LTR: " f"{query_report['ltr_ndcg@10'].mean():.6f}")

    ltr_better = int((query_report["ltr_minus_bm25"] > 0).sum())

    bm25_better = int((query_report["ltr_minus_bm25"] < 0).sum())

    ties = int((query_report["ltr_minus_bm25"] == 0).sum())

    print()
    print("LTR vs BM25:")
    print(f"  LTR better: {ltr_better}")
    print(f"  BM25 better: {bm25_better}")
    print(f"  Ties: {ties}")

    # Reports.
    failure_report = create_failure_report(rankings)

    success_report = create_success_report(query_report)

    bm25_failure_report = create_bm25_vs_ltr_failures(query_report)

    length_report = create_query_length_analysis(query_report)

    type_report = create_query_type_analysis(query_report)

    category_report = create_category_analysis(
        rankings,
        products,
    )

    ambiguous_report = create_ambiguous_query_analysis(
        rankings,
        products,
    )

    confusion_report = create_confusion_analysis(rankings)

    # Output directory.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save artifacts.
    rankings.to_parquet(
        OUTPUT_DIR / "test_rankings.parquet",
        index=False,
    )

    query_report.to_csv(
        OUTPUT_DIR / "query_level_error_report.csv",
        index=False,
    )

    failure_report.to_csv(
        OUTPUT_DIR / "top_failure_cases.csv",
        index=False,
    )

    success_report.to_csv(
        OUTPUT_DIR / "top_success_cases.csv",
        index=False,
    )

    bm25_failure_report.to_csv(
        OUTPUT_DIR / "bm25_vs_ltr_failures.csv",
        index=False,
    )

    length_report.to_csv(
        OUTPUT_DIR / "query_length_error_analysis.csv",
        index=False,
    )

    type_report.to_csv(
        OUTPUT_DIR / "query_type_error_analysis.csv",
        index=False,
    )

    category_report.to_csv(
        OUTPUT_DIR / "category_error_analysis.csv",
        index=False,
    )

    ambiguous_report.to_csv(
        OUTPUT_DIR / "ambiguous_query_analysis.csv",
        index=False,
    )

    confusion_report.to_csv(
        OUTPUT_DIR / "relevance_confusion_analysis.csv",
        index=False,
    )

    elapsed = time.perf_counter() - start_time

    # Summary.
    summary = {
        "evaluation_scope": ("Held-out WANDS test query groups"),
        "test_queries": test_queries,
        "test_candidates": test_candidates,
        "models": [
            "TF-IDF",
            "BM25",
            "Supervised baseline",
            "LightGBM LTR",
        ],
        "primary_metric": "NDCG@10",
        "average_ndcg@10": {
            "tfidf": float(query_report["tfidf_ndcg@10"].mean()),
            "bm25": float(query_report["bm25_ndcg@10"].mean()),
            "supervised": float(query_report["supervised_ndcg@10"].mean()),
            "ltr": float(query_report["ltr_ndcg@10"].mean()),
        },
        "bm25_better_query_count": bm25_better,
        "ltr_better_query_count": ltr_better,
        "tie_query_count": ties,
        "failure_case_count": len(failure_report),
        "success_case_count": len(success_report),
        "ambiguous_query_count": len(ambiguous_report),
        "analysis_runtime_seconds": float(elapsed),
        "ambiguity_definition": (
            "Heuristic: a query is considered "
            "operationally ambiguous when its "
            "relevant products span at least "
            "two top-level product categories. "
            "This is not a WANDS ground-truth label."
        ),
        "evaluation_note": (
            "All NDCG@10 calculations use the "
            "project's canonical ndcg_at_k "
            "implementation."
        ),
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("Failure cases extracted: " f"{len(failure_report)}")

    print("Success cases extracted: " f"{len(success_report)}")

    print("Ambiguous queries identified: " f"{len(ambiguous_report)}")

    print()
    print("Artifacts saved to:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("Stage 18 analysis completed.")


if __name__ == "__main__":
    main()
