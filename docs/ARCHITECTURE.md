# RelevanceFlow — Architecture

**Stage:** 01 — Project Definition
**Status:** FROZEN FOR V1

## 1. Architecture Goal

RelevanceFlow is designed as a modular query-product ranking platform rather than a single notebook.

The architecture separates:

- data
- validation
- preprocessing
- feature generation
- model training
- evaluation
- experiment tracking
- inference
- serving
- monitoring

## 2. Logical Architecture

```text
                         OFFLINE / TRAINING
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  WANDS                                                           │
│   │                                                              │
│   ▼                                                              │
│  Ingestion ──► Validation ──► Leakage Checks ──► Split          │
│                                              │                   │
│                                              ▼                   │
│                                  Preprocessing / Features        │
│                                              │                   │
│                         ┌────────────────────┼────────────────┐  │
│                         ▼                    ▼                ▼  │
│                      TF-IDF                 BM25          ML/LTR │
│                         │                    │                │  │
│                         └────────────────────┼────────────────┘  │
│                                              ▼                   │
│                                         Evaluation               │
│                                   NDCG / MRR / Recall            │
│                                              │                   │
│                                              ▼                   │
│                                            MLflow                 │
│                                   experiments / artifacts         │
│                                              │                   │
│                                              ▼                   │
│                                      Model Candidate              │
└──────────────────────────────────────────────┬───────────────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ Model Artifact  │
                                      │ + Metadata      │
                                      └────────┬────────┘
                                               │
                         ONLINE / INFERENCE     │
                                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI Ranking Service                    │
│                                                                  │
│   Query + Candidate Products                                     │
│              │                                                   │
│              ▼                                                   │
│      Request Validation                                          │
│              │                                                   │
│              ▼                                                   │
│      Feature Generation                                          │
│              │                                                   │
│              ▼                                                   │
│      Loaded Ranking Model                                        │
│              │                                                   │
│              ▼                                                   │
│      Ranked Products + Scores                                    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
                     Monitoring / Logging
```

## 3. Data Flow

```text
product.csv ─────┐
                 │
query.csv ───────┼──► validated relational data
                 │
label.csv ───────┘
                         │
                         ▼
              query-product training examples
                         │
                         ▼
                query-grouped split
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       training queries       held-out queries
             │                       │
             ▼                       ▼
       feature pipeline        same frozen
             │                 transformations
             ▼                       │
        model training              │
             │                       │
             └───────────┬───────────┘
                         ▼
                    evaluation
```

## 4. Repository Structure

The initial target structure is:

```text
relevanceflow/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   └── model.yaml
│
├── docs/
│   ├── PROJECT_SPEC.md
│   └── ARCHITECTURE.md
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   └── relevanceflow/
│       ├── __init__.py
│       ├── data/
│       ├── validation/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       ├── tracking/
│       ├── serving/
│       └── utils/
│
├── pipelines/
│   ├── ingest.py
│   ├── validate.py
│   ├── train.py
│   └── evaluate.py
│
├── scripts/
│
├── api/
│
├── monitoring/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── reports/
│
├── models/
│
├── infra/
│   ├── docker/
│   └── ci/
│
├── .github/
│   └── workflows/
│
├── .gitignore
├── README.md
├── pyproject.toml
└── requirements/ or equivalent dependency configuration
```

The exact files inside these directories will be introduced only when their corresponding stage begins.

## 5. Component Responsibilities

### Ingestion

Responsible for acquiring/loading the WANDS source files.

It must not contain model logic.

### Validation

Responsible for:
- schema checks
- required columns
- data types
- missing values
- duplicate checks
- valid labels
- query/product referential integrity
- basic leakage checks

### Preprocessing

Responsible for deterministic transformations of text and structured data.

Any fitted transformation must be learned only from the training partition.

### Feature Engineering

Responsible for creating query-product features such as:
- lexical overlap
- TF-IDF similarity
- length statistics
- category compatibility
- numeric product signals

### Models

Contains independent model implementations.

No notebook-specific training code should be required for inference.

### Evaluation

Contains ranking metric implementations and evaluation orchestration.

### Tracking

Handles MLflow logging once experiment tracking is introduced.

### Serving

Loads a selected model artifact and exposes ranking functionality through FastAPI.

### Monitoring

Handles:
- request/inference logging
- input data quality
- distribution/drift checks where meaningful
- basic service health

## 6. Training/Serving Contract

Training must produce a versioned artifact containing enough metadata to reproduce inference:

```text
model
feature configuration
preprocessing configuration
model version
training dataset version
feature version
random seed
training timestamp
evaluation metrics
```

The serving layer must not retrain or modify the model.

## 7. Leakage Controls

The architecture treats query grouping as a first-class boundary.

```text
Unique queries
      │
      ├── train queries
      ├── validation queries
      └── test queries
```

No relevance judgment belonging to a test query may be used during training.

Preprocessing components such as vocabulary construction, normalization statistics, or learned encoders must be fitted using training data only.

Repeated products across query groups are allowed because a product catalog naturally participates in many searches.

## 8. Model Selection Contract

Model selection is based on validation ranking metrics, not test-set optimization.

```text
Train → fit models
Validation → compare/tune
Test → final one-time evaluation
```

The test set should remain untouched until the model-selection process is frozen.

Primary metric:

```text
NDCG@10
```

Secondary metrics:

```text
MRR@10
Recall@10
MAP@10
```

## 9. Deployment Architecture

For V1:

```text
Client
  │
  ▼
FastAPI
  │
  ├── request validation
  ├── preprocessing
  ├── feature generation
  ├── model inference
  └── ranking response
  │
  ▼
Docker container
```

A database is not required for the initial V1 serving path because the client supplies candidate products.

A catalog/index-backed retrieval layer may be considered in V2 if the project evolves toward a two-stage search system.

## 10. CI/CD

Initial CI should focus on:

```text
push / pull request
      │
      ├── install dependencies
      ├── lint/type checks where configured
      ├── unit tests
      └── integration smoke tests
```

Deployment automation should be added only after the local service is stable.

## 11. Monitoring

Because WANDS is an offline benchmark rather than a live customer system, monitoring will focus on engineering and data-quality signals rather than fabricated business KPIs.

Potential signals:
- request count
- error rate
- inference latency
- input schema failures
- query length distribution
- candidate-set size
- feature distribution drift
- model/version identifier
- score distribution

Ground-truth relevance drift cannot be monitored online unless new labeled data becomes available.

## 12. Architecture Decisions

### Decision: query-grouped evaluation

Chosen because ranking performance should measure generalization to unseen search queries.

### Decision: NDCG@10 as primary metric

Chosen because WANDS has graded relevance and the system is intended to optimize top-ranked results.

### Decision: classical baselines before neural models

Chosen to establish interpretable and computationally efficient baselines before introducing additional model complexity.

### Decision: no forced vector database

A vector database is not inherently required for the benchmark. It will be introduced only if the retrieval architecture later demonstrates a genuine need.

### Decision: no Kubernetes in V1

Containerization and CI/CD provide meaningful deployment engineering value first. Kubernetes will only be considered if the deployment problem actually requires orchestration.

## 13. Architecture Freeze

The architecture is frozen at the logical level for Stage 01.

Implementation details may change when measured constraints appear, but changes must be documented as architecture decisions rather than silently introducing new components.
