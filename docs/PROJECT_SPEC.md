# RelevanceFlow — Project Specification

**Stage:** 01 — Project Definition
**Status:** FROZEN FOR V1
**Dataset:** WANDS (Wayfair ANNotation Dataset)
**Primary task:** E-commerce product-search relevance ranking

## 1. Problem Statement

E-commerce search systems must rank candidate products so that products most relevant to a customer's search query appear first.

RelevanceFlow will build a reproducible, production-oriented learning-to-rank system that takes a search query and candidate products, estimates query-product relevance, and produces an ordered list of products.

The project uses the public WANDS dataset from Wayfair, which contains 42,994 candidate products, 480 search queries, and 233,448 human-annotated query-product relevance judgments. The official dataset defines three relevance labels: `Exact`, `Partial`, and `Irrelevant`.

Source: https://github.com/wayfair/WANDS

## 2. Business Objective

Improve the quality of e-commerce search result ordering by placing products that better satisfy the user's search intent higher in the result list.

The business-facing objective is not merely to classify a query-product pair. It is to improve the ordering of candidate products, especially at the top of the search results where user attention is concentrated.

Potential business impact:
- more relevant first-page search results
- reduced search effort
- better product discovery
- improved search experience
- a measurable framework for comparing ranking models

No business-conversion or revenue improvement will be claimed because WANDS does not provide the necessary production conversion/click outcome data.

## 3. ML Objective

Given a query `q` and a candidate product `p`, learn a relevance scoring function:

    score(q, p) -> R

where a larger score indicates greater predicted relevance.

For a candidate set:

    P(q) = {p1, p2, ..., pn}

the system produces:

    rank(P(q)) = [p(1), p(2), ..., p(k)]

such that products with higher predicted relevance are ranked earlier.

The primary ML formulation is **learning-to-rank / query-product relevance ranking**.

A secondary diagnostic formulation may treat the three labels as an ordinal classification problem, but the production objective remains ranking.

## 4. Users

### Primary users
- ML engineers developing and evaluating search-ranking models
- Search/relevance engineers
- Data scientists analyzing search quality

### Secondary users
- Recruiters/interviewers reviewing the engineering and ML workflow
- Developers consuming the ranking API

This is a portfolio system, not a production Wayfair system.

## 5. Input

### Online inference input

A query and a candidate product set.

Example conceptual request:

```json
{
  "query": "white leather chair",
  "products": [
    {
      "product_id": "123",
      "product_name": "...",
      "product_class": "...",
      "category_hierarchy": "...",
      "product_description": "...",
      "product_features": "...",
      "rating_count": 10,
      "average_rating": 4.5,
      "review_count": 8
    }
  ]
}
```

The exact API schema will be finalized during the serving stage.

### Dataset fields

WANDS provides:

**Query**
- `query_id`
- `query`
- `query_class`

**Product**
- `product_id`
- `product_name`
- `product_class`
- `category_hierarchy`
- `product_description`
- `product_features`
- `rating_count`
- `average_rating`
- `review_count`

**Judgment**
- `id`
- `query_id`
- `product_id`
- `label`

## 6. Output

For each query and candidate set:

```text
product_id
predicted_relevance_score
rank
```

Optionally, V2 may expose:
- model/version identifier
- component feature scores
- explanation/debug information

The API will return products ordered from highest to lowest predicted relevance.

## 7. Dataset

### WANDS — Wayfair ANNotation Dataset

WANDS was released by Wayfair as a dataset for product-search relevance assessment and accompanied the ECIR 2022 paper "WANDS: Dataset for Product Search Relevance Assessment."

Verified dataset scale:
- 42,994 candidate products
- 480 queries
- 233,448 query-product relevance judgments

The official repository describes the dataset as containing product metadata, search queries, and human relevance judgments.

Official repository:
https://github.com/wayfair/WANDS

Paper:
Chen, Yan; Liu, Shujian; Liu, Zheng; Sun, Weiyi; Baltrunas, Linas; Schroeder, Benjamin.
"WANDS: Dataset for Product Search Relevance Assessment."
ECIR 2022.

### Dataset suitability

WANDS is small enough for iterative local development while still providing a meaningful ranking problem with rich text and product metadata.

The 42,994 products and 233,448 judgments are substantial enough for classical ML/NLP experiments without requiring the multi-million-row infrastructure of a large industrial search dataset.

## 8. Relevance Labels

Official WANDS labels:

| Label | Meaning | Ranking relevance |
|---|---|---:|
| `Exact` | Product precisely matches the query intent | 2 |
| `Partial` | Product is related/relevant but not an exact match | 1 |
| `Irrelevant` | Product does not satisfy the query intent | 0 |

For ranking metrics, the canonical relevance grades will therefore be:

```text
Exact      -> 2
Partial    -> 1
Irrelevant -> 0
```

The mapping is an engineering representation of the ordered human relevance judgments. It does not mean the numerical distance between labels has a literal business interpretation.

## 9. Ranking Formulation

For query `q` and candidate product `p`:

```text
q = search query
p = candidate product

score(q, p) = f(features(q, p))
```

The model learns:

```text
f(q, p) -> predicted relevance score
```

For a candidate set:

```text
P(q) = {p1, p2, ..., pn}
```

the ranking is:

```text
sort(P(q), key=score(q,p), descending=True)
```

producing:

```text
p1 > p2 > p3 > ... > pk
```

where `p1` is the highest-ranked predicted result.

### Pairwise/listwise considerations

V1 will begin with pointwise/classification-style and feature-based ranking baselines where appropriate, then move toward a genuine learning-to-rank objective if the data representation and implementation justify it.

We will not force a neural ranker into the project simply for complexity.

## 10. Data Splitting Strategy

The primary unit of generalization is the **query**, not an individual query-product row.

Therefore, random row-level splitting is prohibited for the final evaluation because products from the same query would otherwise appear across train/validation/test and can make evaluation less representative of unseen-query performance.

V1 will use **query-grouped splitting**:

```text
train:      ~70% of unique queries
validation: ~15% of unique queries
test:       ~15% of unique queries
```

The exact query counts will be determined by the implemented deterministic group split and recorded in the dataset manifest.

A fixed random seed will be used.

Before training, we will also check:
- duplicate query-product judgments
- duplicate products
- repeated query text under different IDs
- product overlap across query groups
- preprocessing leakage
- target leakage
- feature availability at inference time

### Important distinction

A product appearing in multiple query groups is not automatically leakage. Products are reusable candidates in an e-commerce catalog. The critical boundary is that relevance judgments for the held-out queries must not be used to train the model.

## 11. Features

Candidate feature families:

### Text matching
- query/product token overlap
- TF-IDF similarity
- BM25-style lexical features if justified
- character/word n-gram similarity
- query length
- product title length
- description length

### Product/query semantic metadata
- product class compatibility
- category overlap
- query/product token containment
- feature/attribute overlap

### Numeric product features
- average rating
- rating count
- review count

Numeric popularity/quality features will be handled carefully because they are not direct relevance labels.

### Features excluded initially

`query_class` will not be treated as an automatically trusted production feature in V1. The official WANDS schema describes it as a predicted product class for the query, so using it without reproducing its upstream generation process could introduce an uncontrolled dependency or leakage risk.

It can be investigated later as an explicit experiment.

## 12. Evaluation Metrics

### Primary metrics

**NDCG@10**
- Measures ranking quality while giving higher weight to highly relevant products and higher positions.
- Appropriate for graded relevance (`2/1/0`).

**MRR@10**
- Measures how early the first highly relevant result appears.
- Useful when users need at least one strong result quickly.

### Secondary metrics

**Recall@10**
- Measures whether relevant products appear within the top 10.
- Useful for top-k retrieval/ranking diagnostics.

**MAP@10**
- Measures precision across positions for binary relevance variants.
- Useful as a supplementary metric, not the sole objective.

### Additional reporting

- Exact-label precision/recall where useful for diagnostics
- per-query metric distributions
- worst-performing query examples
- latency at inference time once serving exists

The primary ranking metric will remain NDCG@10 unless later analysis shows a strong reason to change it.

No metric value will be reported as achieved until it is measured by an actual experiment.

## 13. Model Progression

The progression will be evidence-driven.

### Phase A — Data and sanity baselines
1. Random / simple ranking sanity check
2. Popularity-style baseline where valid

### Phase B — Classical information retrieval
3. TF-IDF lexical similarity
4. BM25 lexical retrieval/ranking

### Phase C — Feature-based ML
5. Query-product feature engineering
6. Logistic Regression / ordinal-style diagnostic baseline
7. Gradient-boosted learning-to-rank or ranking-compatible tree model

### Phase D — Semantic NLP
8. Sentence-transformer/embedding similarity
9. Hybrid lexical + semantic ranking

A transformer cross-encoder will only be added if the dataset, compute budget, and measured baseline results justify it.

### Phase E — Productionization
10. MLflow experiment tracking
11. model/version management
12. reproducible inference pipeline
13. FastAPI serving
14. Docker
15. CI tests
16. monitoring/drift checks

The final architecture will be selected from measured results rather than predetermined complexity.

## 14. System Architecture

High-level system:

```text
                 ┌────────────────────┐
                 │   WANDS Dataset    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Data Ingestion     │
                 │ + Versioning       │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Data Validation    │
                 │ + Leakage Checks   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Preprocessing &    │
                 │ Feature Engineering│
                 └─────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌──────────────┐          ┌──────────────┐
      │ IR Baselines │          │ ML Rankers   │
      │ TF-IDF/BM25  │          │ LTR/ML       │
      └──────┬───────┘          └──────┬───────┘
             │                         │
             └────────────┬────────────┘
                          ▼
                 ┌────────────────────┐
                 │ Evaluation         │
                 │ NDCG/MRR/Recall    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ MLflow             │
                 │ Experiments/Models │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ FastAPI Inference  │
                 │ Ranking Service    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Docker / CI        │
                 │ Monitoring         │
                 └────────────────────┘
```

## 15. Functional Requirements

### FR-01 Dataset ingestion
The system shall ingest the official WANDS query, product, and label data.

### FR-02 Validation
The pipeline shall validate schema, required columns, nulls, duplicates, label values, and referential integrity.

### FR-03 Leakage-safe splitting
The pipeline shall split data by query group rather than random query-product rows.

### FR-04 Preprocessing
The system shall normalize and transform query/product text without fitting preprocessing components on test data.

### FR-05 Feature engineering
The system shall generate reproducible query-product relevance features.

### FR-06 Baselines
The system shall provide deterministic baseline ranking implementations.

### FR-07 Model training
The system shall train at least one feature-based ML ranking model after the IR baselines.

### FR-08 Evaluation
The system shall calculate ranking metrics on held-out queries.

### FR-09 Experiment tracking
MLflow shall track model parameters, dataset/feature versions, training sizes, metrics, and artifacts once experimentation reaches the model-comparison stage.

### FR-10 Model versioning
The selected production candidate shall have a reproducible model artifact and version metadata.

### FR-11 Inference API
The system shall expose a FastAPI endpoint that accepts a query and candidate products and returns ranked products.

### FR-12 Testing
Core preprocessing, feature engineering, ranking, API validation, and utility logic shall have automated tests.

### FR-13 Containerization
The inference service shall be runnable through Docker.

### FR-14 Monitoring
The deployed service shall expose useful operational signals and data-quality/drift checks appropriate to the offline/portfolio setting.

## 16. Non-Functional Requirements

### Reproducibility
- fixed random seeds
- pinned dependencies
- versioned configuration
- dataset/version manifest
- reproducible training command

### Maintainability
- modular `src/` package
- type hints
- structured configuration
- logging
- separation of training and inference code

### Testability
- pytest-based unit tests
- deterministic tests for critical transformations

### Performance
The service should provide practical local inference latency for a reasonable candidate set. Exact latency targets will be established only after measurement.

### Resource efficiency
The project should remain runnable on a normal development laptop for core experiments. GPU use is optional and justified only for models that benefit from it.

### Observability
Training runs and production inference should expose enough information to diagnose failures and model/data changes.

## 17. V1 Scope

V1 is the minimum complete end-to-end ranking system:

1. WANDS ingestion
2. schema/data validation
3. query-grouped train/validation/test split
4. leakage checks
5. text preprocessing
6. TF-IDF baseline
7. BM25 baseline if implementation remains practical
8. query-product feature engineering
9. feature-based ML ranking model
10. NDCG@10 / MRR@10 / Recall@10 evaluation
11. MLflow experiment tracking
12. model artifact/versioning
13. FastAPI inference endpoint
14. pytest tests
15. Dockerized inference
16. basic CI
17. basic data/model monitoring

## 18. V2 Scope

V2 is an improvement layer added only after V1 is stable:

- semantic embeddings
- hybrid lexical + semantic ranking
- optional transformer/cross-encoder experiment
- model comparison dashboard/report
- richer model explanations
- stronger drift analysis
- load testing
- cloud deployment if it provides meaningful engineering value
- advanced retrieval/reranking architecture if supported by measured evidence

## 19. Non-Goals

The following are explicitly outside the initial project:

- reproducing Wayfair's proprietary production search system
- claiming production business impact such as revenue uplift
- building a full web storefront
- collecting real customer data
- building a real-time distributed search engine
- processing millions/billions of products
- adding Kubernetes solely for resume decoration
- adding LLMs solely because the project is NLP-related
- treating a classification accuracy score as the primary search-quality metric
- using random row splits for final ranking evaluation
- claiming unmeasured performance
- creating an artificially complex architecture before the core ranking problem works

## 20. Stage-01 Freeze Criteria

Stage 01 is complete when the following are agreed and frozen:

- [x] Problem definition
- [x] Business objective
- [x] ML objective
- [x] Users
- [x] Input/output
- [x] WANDS dataset
- [x] Relevance labels
- [x] Ranking formulation
- [x] Evaluation metric policy
- [x] Model progression
- [x] High-level architecture
- [x] Functional requirements
- [x] Non-functional requirements
- [x] V1/V2 scope
- [x] Non-goals

**No production ML code should be written until Stage 01 is approved.**
