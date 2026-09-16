# RelevanceFlow — WANDS Dataset Exploration Report

**Stage:** 06 — Dataset Exploration
**Dataset:** WANDS
**Status:** Awaiting local notebook execution

## Purpose

This report records descriptive statistics and visual analysis of the actual WANDS files used by RelevanceFlow.

The report must be generated from:

```text
data/raw/wands/
├── product.csv
├── query.csv
└── label.csv
```

No dataset statistics are fabricated in this document. Run:

```text
notebooks/06_dataset_exploration.ipynb
```

to measure the values locally.

## 1. Dataset Overview

| Dataset component | Rows | Columns |
|---|---:|---:|
| Products | Not measured yet | Not measured yet |
| Queries | Not measured yet | Not measured yet |
| Judgments | Not measured yet | Not measured yet |

## 2. Query Statistics

Record:

- number of queries
- unique query IDs
- duplicate query IDs
- duplicate query text
- query word-count mean/median
- query word-count distribution
- query classes

**Measured values:** Not measured yet.

## 3. Product Statistics

Record:

- number of products
- unique product IDs
- duplicate product IDs
- product classes
- missing descriptions
- missing product features
- category distribution
- average rating distribution
- rating-count distribution
- review-count distribution
- product text-length distribution

**Measured values:** Not measured yet.

## 4. Judgment Statistics

Record:

- total judgments
- Exact count and percentage
- Partial count and percentage
- Irrelevant count and percentage
- judgments per query
- unique products per query
- duplicate query-product pairs

**Measured values:** Not measured yet.

## 5. Data Integrity Checks

The notebook checks:

- query/product referential integrity
- duplicate identifiers
- duplicate query-product judgments
- null values
- label values

**Measured values:** Not measured yet.

## 6. Visualizations

The notebook generates:

1. relevance-label distribution
2. query-length distribution
3. product-name length distribution
4. judgments-per-query distribution
5. top product-category distribution
6. rating distributions

## 7. Initial Observations

Do not write conclusions until the notebook has been executed.

**Current status:** Not measured yet.

## 8. ML Implications

The exploration should inform later decisions about:

- preprocessing
- feature engineering
- class/relevance imbalance
- query-grouped splitting
- ranking evaluation
- text representation
- computational constraints
- potential leakage

These decisions must be based on the measured dataset rather than assumptions.

## 9. Reproducibility

The notebook saves:

```text
reports/stage06_exploration_summary.csv
```

The raw WANDS files remain immutable under:

```text
data/raw/wands/
```

Generated statistics belong in `reports/`, not in the raw dataset directory.
