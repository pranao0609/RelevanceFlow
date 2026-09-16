# WANDS Dataset Provenance

## Dataset

**WANDS — Wayfair ANNotation Dataset**

WANDS is a public e-commerce product-search relevance dataset released by Wayfair as a companion dataset to the ECIR 2022 paper:

> Chen, Yan; Liu, Shujian; Liu, Zheng; Sun, Weiyi; Baltrunas, Linas; Schroeder, Benjamin.
> WANDS: Dataset for Product Search Relevance Assessment.
> Proceedings of the 44th European Conference on Information Retrieval (ECIR), 2022.

Official repository:
https://github.com/wayfair/WANDS

Official dataset location:
https://github.com/wayfair/WANDS/tree/main/dataset

## License

The official WANDS repository states that the project is distributed under the **MIT License**.

Before redistributing the raw dataset with this project, review the official repository and license terms. For the portfolio repository, the recommended approach is to keep raw WANDS data out of Git and document how to obtain it.

## Dataset Scale

According to the official WANDS repository:

- 42,994 candidate products
- 480 search queries
- 233,448 query-product relevance judgments

These figures describe the official dataset release and are not measurements produced by RelevanceFlow.

## Files

The official `dataset/` directory contains:

### `product.csv`

Product catalog metadata:

- `product_id`
- `product_name`
- `product_class`
- `category_hierarchy`
- `product_description`
- `product_features`
- `rating_count`
- `average_rating`
- `review_count`

### `query.csv`

Search-query metadata:

- `query_id`
- `query`
- `query_class`

### `label.csv`

Human relevance annotations:

- `id`
- `query_id`
- `product_id`
- `label`

The official labels are:

- `Exact`
- `Partial`
- `Irrelevant`

## Local Storage

RelevanceFlow stores the downloaded files at:

```text
data/raw/wands/
├── product.csv
├── query.csv
└── label.csv
```

Raw data is intentionally excluded from Git through `.gitignore`.

## Acquisition

The official repository can be obtained with:

```powershell
git clone https://github.com/wayfair/WANDS.git
```

Then copy:

```text
WANDS/dataset/product.csv
WANDS/dataset/query.csv
WANDS/dataset/label.csv
```

into:

```text
RelevanceFlow/data/raw/wands/
```

Alternatively, the raw files can be obtained directly from the official repository's `dataset/` directory.

## Integrity / Reproducibility

At the acquisition stage, RelevanceFlow should record:

- source repository
- source commit/tag when available
- acquisition date
- file names
- row counts
- column names
- file checksums

The checksums and measured local statistics will be generated during dataset validation rather than fabricated in this document.

## Important

Do not edit files inside `data/raw/wands/`.

Raw data should be treated as immutable source data.

All cleaning, joining, normalization, filtering, and feature engineering should happen downstream in:

```text
data/interim/
data/processed/
```

## Citation

If WANDS is used in RelevanceFlow, cite the original WANDS ECIR 2022 paper as requested by the official repository.
