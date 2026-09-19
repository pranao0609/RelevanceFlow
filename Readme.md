# RelevanceFlow

## Multilingual Product Search Relevance & Ranking Platform

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-LambdaRank-GREEN?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)

RelevanceFlow is a production-oriented machine learning system for **e-commerce product search relevance and ranking**.

Given a search query and a set of candidate products, RelevanceFlow computes relevance features and uses a learning-to-rank model to order products by predicted relevance.

The project focuses on practical **Information Retrieval, NLP, classical machine learning, Learning-to-Rank, model management, API serving, database persistence, caching, testing, reproducibility, and performance evaluation**.

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Project Objectives](#2-project-objectives)
- [3. System Architecture](#3-system-architecture)
- [4. Dataset](#4-dataset)
- [5. Data Processing](#5-data-processing)
- [6. Data Versioning](#6-data-versioning)
- [7. Train / Validation / Test Split](#7-train--validation--test-split)
- [8. Retrieval Baselines](#8-retrieval-baselines)
- [9. Feature Engineering](#9-feature-engineering)
- [10. Learning-to-Rank Model](#10-learning-to-rank-model)
- [11. Evaluation](#11-evaluation)
- [12. MLflow Experiment Tracking & Model Registry](#12-mlflow)
- [13. Model Reproducibility](#13-model-reproducibility)
- [14. FastAPI Serving](#14-fastapi-serving)
- [15. PostgreSQL Request Audit](#15-postgresql)
- [16. Redis Caching](#16-redis-caching)
- [17. Performance Testing](#17-performance-testing)
- [18. Testing & Code Quality](#18-testing)
- [19. Project Structure](#19-project-structure)
- [20. Local Setup](#20-local-setup)
- [21. Data Setup](#21-data-setup)
- [22. Running the API](#22-running-the-api)
- [23. Redis for Local Development](#23-redis-for-local-development)
- [24. Running Tests](#24-running-tests)
- [25. Design Decisions](#25-design-decisions)
- [26. Limitations](#26-limitations)
- [27. Future Extensions](#27-future-extensions)
- [28. Technical Stack](#28-technical-stack)
- [29. Engineering Focus](#29-engineering-focus)
- [30. Project Status](#30-project-status)
- [License](#license)

---

## 1. Problem Statement

In an e-commerce search system, retrieving candidate products is only part of the problem.

The system must also determine:

> **Which candidate products should appear first for a given query?**

For a query `q` and candidate product `p`, RelevanceFlow learns a ranking function:

```text
score(q, p) -> relevance score
```

Products are then sorted by the predicted score.

The project uses the **WANDS (Wayfair ANnotation Dataset)** dataset, which provides query-product relevance judgments.

---

## 2. Project Objectives

The primary objectives are:

- Build an end-to-end product-search relevance pipeline.
- Establish classical Information Retrieval baselines.
- Compare lexical retrieval approaches.
- Engineer query-product relevance features.
- Train a supervised Learning-to-Rank model.
- Evaluate ranking quality using standard IR metrics.
- Track experiments and models using MLflow.
- Serve the trained model through FastAPI.
- Persist ranking-request metadata using PostgreSQL.
- Add Redis caching for repeated ranking requests.
- Test the system using automated unit and integration tests.
- Benchmark API performance under controlled workloads.
- Maintain reproducibility using configuration, DVC, caching, and provenance tracking.

> [!NOTE]
> This project does not claim production business impact such as increased conversion or revenue because the WANDS dataset does not contain production click or transaction outcomes.

---

## 3. System Architecture

```text
                         ┌─────────────────────┐
                         │      WANDS          │
                         │ Products / Queries  │
                         │ Relevance Judgments │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Validation &        │
                         │ Data Cleaning       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Query-Grouped       │
                         │ Train / Val / Test  │
                         │ Split               │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
        ┌─────────────────┐                   ┌─────────────────┐
        │ TF-IDF          │                   │ BM25            │
        │ Retrieval       │                   │ Retrieval       │
        └────────┬────────┘                   └────────┬────────┘
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │ Lexical             │
                         │ Metadata            │
                         │ Semantic            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ LightGBM            │
                         │ LambdaRank          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ MLflow              │
                         │ Experiment Tracking │
                         │ Model Registry      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FastAPI Ranking API │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                ┌─────────────────┐   ┌─────────────────┐
                │ PostgreSQL      │   │ Redis           │
                │ Request Audit   │   │ Response Cache  │
                └─────────────────┘   └─────────────────┘
```

---

## 4. Dataset

RelevanceFlow uses the **WANDS** dataset with three primary files:

- `product.csv`
- `query.csv`
- `label.csv`

### Key Fields

- **Product data**: `product_id`, `product_name`, `product_class`, category hierarchy, `product_description`, `product_features`, `rating_count`, `average_rating`, `review_count`
- **Query data**: `query_id`, `query`, `query_class`
- **Relevance labels**: Mapped relevance levels:
  - **Exact** $\rightarrow$ `2`
  - **Partial** $\rightarrow$ `1`
  - **Irrelevant** $\rightarrow$ `0`

---

## 5. Data Processing

The raw WANDS data is validated before entering the modeling pipeline.

```text
Raw WANDS ──► Schema Validation ──► Missing Checks ──► Duplicate Detection ──► Conflict Resolution ──► Cleaned Dataset
```

### Dataset Statistics

| Dataset Version | Products | Queries | Judgments |
| :--- | :---: | :---: | :---: |
| **Raw Dataset** | - | - | 233,448 |
| **Processed Dataset** | 42,994 | 480 | 231,873 |

The cleaning stage also generates a conflict audit artifact for inspection.

---

## 6. Data Versioning

**DVC (Data Version Control)** is used to version dataset artifacts separately from Git-tracked source code.

- **Tracked by DVC**:
  - `data/raw/wands/`
  - `data/processed/wands/`
  - Large binary data artifacts
- **Tracked by Git**:
  - Source code
  - Configuration files (`configs/`)
  - Automated tests
  - Documentation & pipelines

---

## 7. Train / Validation / Test Split

The project uses a **query-grouped split** where a query belongs entirely to one split. This prevents judgments for the same query from leaking across training and evaluation sets.

| Split | Queries | Judgments |
| :--- | :---: | :---: |
| **Train** | 336 | 156,714 |
| **Validation** | 72 | 28,531 |
| **Test** | 72 | 46,628 |
| **Total** | **480** | **231,873** |

*Note: Products can appear across different queries because the grouping constraint is applied strictly at the query level.*

---

## 8. Retrieval Baselines

Before training supervised ranking models, classical Information Retrieval baselines are established:

1. **TF-IDF**: Provides a lexical similarity baseline measuring similarity between query and product text based on term frequency.
2. **BM25**: Provides a probabilistic lexical retrieval baseline incorporating term frequency, inverse document frequency, and document length normalization.

---

## 9. Feature Engineering

The supervised ranking model extracts comprehensive query-product features across multiple categories:

| Category | Features Included |
| :--- | :--- |
| **Lexical Features** | `bm25_score`, `tfidf_similarity`, `exact_match`, `token_overlap`, `character_overlap`, `query_title_overlap`, `phrase_match` |
| **Text / Length Features** | `title_length`, `description_length`, `feature_count` |
| **Product Metadata** | `average_rating`, `rating_count`, `review_count`, `category_depth`, `category_match`, `product_class_match` |
| **Semantic Features** | `semantic_similarity` |

Semantic similarity is generated using the pre-trained `all-MiniLM-L6-v2` SentenceTransformer. The transformer is used strictly as a feature extractor.

---

## 10. Learning-to-Rank Model

The primary supervised model is **LightGBM LambdaRank**.

- **Why LambdaRank?** The objective is query-level ranking rather than independent binary classification.
- **Grouping**: Ranking groups are defined by `query_id` to learn relative within-query ranking relationships.

---

## 11. Evaluation

The primary ranking metric is **NDCG@10**, supplemented by **MRR@10**, **Recall@10**, and **MAP@10**.

### Test Set Reference Benchmark Results

| Model / Baseline | NDCG@10 | MRR@10 | MAP@10 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 0.689050 | 0.961806 | 0.843746 | 0.059700 |
| **BM25 Baseline** | 0.694206 | 0.899074 | 0.823091 | 0.059845 |
| **Transformer Semantic Baseline** | **0.751518** | **0.969907** | **0.899680** | **0.063666** |
| **Hybrid Learning-to-Rank System** | 0.742351 | 0.933697 | 0.877130 | 0.062343 |

---

## 12. MLflow

MLflow handles experiment tracking and model registry management via a SQLite backend.

- **Tracked Artifacts**: Experiment runs, hyperparameters, metrics, model artifacts, versions, and aliases.
- **Aliases**: Models are managed using aliases such as `candidate` and `champion`.
- **Integration**: The FastAPI serving layer dynamically loads the `champion` model from the registry.

---

## 13. Model Reproducibility

Reproducibility is maintained via:

```text
Configuration Files + DVC + Query-Grouped Split + Feature Caching + Feature Fingerprints + MLflow + Pipeline Provenance
```

### Key Configuration Files

- `configs/base.yaml`
- `configs/data.yaml`
- `configs/retrieval.yaml`
- `configs/ranking.yaml`
- `configs/transformer.yaml`
- `configs/training.yaml`
- `configs/production.yaml`
- `configs/features_cache.yaml`

---

## 14. FastAPI Serving

The trained model is exposed through a production-ready **FastAPI** service.

### Core Endpoints

- `POST /rank` — Rank candidate products for a query.
- `GET /health` — Check service health and active model metadata.

### Request Payload Example

```json
{
  "query": "wireless headphones",
  "products": [
    {
      "product_id": 1,
      "product_name": "Wireless Headphones",
      "product_class": "Electronics"
    },
    {
      "product_id": 2,
      "product_name": "Bluetooth Speaker",
      "product_class": "Electronics"
    }
  ]
}
```

---

## 15. PostgreSQL

PostgreSQL stores request-level audit metadata (request ID, query, candidate count, top-k, model name/alias, latency, timestamp).

> [!TIP]
> Database audit persistence is completely decoupled from the ranking path. If persistence fails, the API still successfully returns ranking predictions.

---

## 16. Redis Caching

Redis implements a performance cache-aside pattern:

```text
Request ──► Deterministic Cache Key ──► Redis GET
                                           ├── HIT  ──► Return Cached Response
                                           └── MISS ──► ML Inference ──► Redis SET ──► Return Response
```

Redis is disabled by default in `configs/production.yaml` and can be enabled for performance benchmarking:

```yaml
redis:
  enabled: true
  url: "redis://localhost:6379/0"
  ttl_seconds: 300
```

---

## 17. Performance Testing

Load testing is performed using `scripts/load_test.py`.

### Warm-Cache Benchmark (100 Requests, Concurrency: 10)

| Metric | Value |
| :--- | :--- |
| **Successful / Failed** | 100 / 0 |
| **Error Rate** | 0.00% |
| **Throughput** | 7.54 req/s |
| **Mean Latency** | 1,317.64 ms |
| **P95 / P99 Latency** | 1,477.25 ms / 1,503.25 ms |

### Unique-Request Benchmark (100 Requests, Concurrency: 10)

| Metric | Value |
| :--- | :--- |
| **Successful / Failed** | 50 / 50 |
| **Error Rate** | 50.00% |
| **Throughput** | 0.48 req/s |
| **Mean / Max Latency** | 7,975.70 ms / 20,200.64 ms |

### Controlled Lower-Concurrency Benchmarks

- **10 req / Concurrency 1**: 100% success (1.57 req/s)
- **10 req / Concurrency 2**: 100% success (3.26 req/s)

---

## 18. Testing & Code Quality

```text
tests/
├── unit/          # Configuration, retrieval, features, models, database, caching
└── integration/   # FastAPI endpoints and full-pipeline integration
```

Run test suite and quality checks:

```bash
# Run pytest test suite
pytest -q

# Code formatting and linting checks
ruff check .
black --check .
```

---

## 19. Project Structure

```text
RelevanceFlow/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── src/
│   └── relevanceflow/
│       ├── data/
│       ├── preprocessing/
│       ├── retrieval/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       ├── serving/
│       ├── monitoring/
│       ├── database/
│       ├── cache/
│       └── utils/
├── configs/
├── notebooks/
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   └── load_test.py
├── pipelines/
├── docs/
│   ├── PROJECT_SPEC.md
│   └── ARCHITECTURE.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 20. Local Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd RelevanceFlow

# 2. Create virtual environment (Python 3.11 recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install project dependencies in editable mode
pip install -e .
```

---

## 21. Data Setup

1. Place raw WANDS files in `data/raw/wands/`:
   - `product.csv`
   - `query.csv`
   - `label.csv`
2. Run data validation & preprocessing scripts.
3. Track dataset versions using DVC: `dvc pull` / `dvc repro`.

---

## 22. Running the API

Start the FastAPI application:

```bash
uvicorn relevanceflow.serving.app:app --reload
```

- API Base URL: `http://127.0.0.1:8000`
- Interactive API Docs: `http://127.0.0.1:8000/docs`

---

## 23. Redis for Local Development

Start Redis via Docker for local response caching tests:

```bash
docker start relevanceflow-redis
docker exec relevanceflow-redis redis-cli ping
# Expected response: PONG
```

---

## 24. Running Tests

```bash
# Run unit tests
pytest tests/unit -q

# Run integration tests
pytest tests/integration -q

# Run full test suite
pytest -q
```

---

## 25. Design Decisions

- **Why Query-Grouped Splitting?** Evaluation is query-centric. Query-level grouping prevents data leakage between splits.
- **Why Learning-to-Rank?** Search requires ordering candidate products relative to a query, which classification cannot model directly.
- **Why LightGBM LambdaRank?** Gradient boosting handles heterogeneous features cleanly while directly optimizing ranking metrics.
- **Why Decoupled PostgreSQL & Redis?** Ensures ranking performance is resilient to storage component downtime.

---

## 26. Limitations

- WANDS is an offline relevance dataset without click-through, session, or revenue data.
- Offline metrics do not directly predict business conversion rates.
- Local benchmark measurements are subject to machine-specific constraints.
- Semantic transformer features are extracted fixed-embeddings (not fine-tuned).

---

## 27. Future Extensions

- Hybrid lexical + dense vector retrieval.
- Cross-encoder reranking models.
- Hyperparameter tuning via Optuna.
- Model drift monitoring and alert automation.
- Cloud-native containerized deployment (K8s / ECS).

---

## 28. Technical Stack

- **Core & Runtime**: Python 3.11
- **Machine Learning & NLP**: LightGBM, scikit-learn, PyTorch, Hugging Face, Sentence-Transformers
- **Information Retrieval**: TF-IDF, BM25, LambdaRank
- **MLOps & Data Versioning**: MLflow, DVC
- **API & Backend**: FastAPI, Uvicorn, Pydantic
- **Database & Cache**: PostgreSQL, SQLAlchemy, Redis
- **Testing & Quality**: pytest, Ruff, Black

---

## 29. Engineering Focus

```text
Problem Definition ──► Data Validation ──► Query Split ──► Baseline IR ──► LTR Training ──► MLflow Tracking ──► FastAPI Serving ──► Redis/Postgres Audit ──► Load Testing
```

---

## 30. Project Status

- [x] WANDS dataset ingestion & validation
- [x] Conflict resolution & cleaning
- [x] Query-grouped dataset splitting
- [x] TF-IDF & BM25 baselines
- [x] Feature engineering & SentenceTransformers integration
- [x] LightGBM LambdaRank model implementation
- [x] MLflow experiment tracking & model registry
- [x] FastAPI production endpoint (`/rank`, `/health`)
- [x] PostgreSQL request audit logging
- [x] Redis cache-aside implementation
- [x] Unit & integration test suites
- [x] Load testing & benchmarking scripts

---

## License

This project utilizes the WANDS dataset subject to its official licensing and terms of use.
