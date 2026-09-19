from __future__ import annotations

import hashlib
import json


def build_ranking_cache_key(
    *,
    model_name: str,
    model_alias: str,
    query: str,
    products: list[dict],
    top_k: int,
) -> str:
    """Create a deterministic Redis cache key."""

    payload = {
        "query": query,
        "products": sorted(
            products,
            key=lambda product: product["product_id"],
        ),
        "top_k": top_k,
    }

    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    return f"ranking:{model_name}:{model_alias}:{digest}"
