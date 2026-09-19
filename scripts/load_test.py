from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx


@dataclass
class RequestResult:
    success: bool
    status_code: int
    latency_ms: float


BASE_PRODUCTS = [
    {
        "product_id": 1,
        "product_name": "Wireless Headphones",
        "product_class": "Electronics",
    },
    {
        "product_id": 2,
        "product_name": "Bluetooth Speaker",
        "product_class": "Electronics",
    },
]


def build_payload(
    request_index: int,
    unique: bool,
) -> dict:
    """Build the ranking request payload."""

    query = "wireless headphones"

    if unique:
        query = f"wireless headphones {request_index}"

    return {
        "query": query,
        "products": BASE_PRODUCTS,
    }


def send_request(
    url: str,
    timeout: float,
    request_index: int,
    unique: bool,
) -> RequestResult:
    """Send one ranking request."""

    start = time.perf_counter()

    payload = build_payload(
        request_index=request_index,
        unique=unique,
    )

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=timeout,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RequestResult(
            success=response.status_code == 200,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
        )

    except httpx.HTTPError:
        elapsed_ms = (time.perf_counter() - start) * 1000

        return RequestResult(
            success=False,
            status_code=0,
            latency_ms=elapsed_ms,
        )


def percentile(
    values: list[float],
    percentage: float,
) -> float:
    """Calculate a percentile without external dependencies."""

    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * percentage / 100

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = index - lower

    return values[lower] + (values[upper] - values[lower]) * weight


def run_load_test(
    url: str,
    requests: int,
    concurrency: int,
    timeout: float,
    unique: bool,
) -> list[RequestResult]:
    """Execute concurrent API requests."""

    if requests <= 0:
        raise ValueError("requests must be greater than zero.")

    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero.")

    concurrency = min(concurrency, requests)

    start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=concurrency,
    ) as executor:
        results = list(
            executor.map(
                lambda index: send_request(
                    url=url,
                    timeout=timeout,
                    request_index=index,
                    unique=unique,
                ),
                range(requests),
            )
        )

    total_seconds = time.perf_counter() - start

    print()
    print("RelevanceFlow Load Test")
    print("=" * 40)
    print(f"URL:           {url}")
    print(f"Requests:      {requests}")
    print(f"Concurrency:   {concurrency}")
    print("Request Mode:  " f"{'unique/cold' if unique else 'repeated/warm-cache'}")
    print(f"Duration:      {total_seconds:.3f}s")

    successful = [result for result in results if result.success]

    failed = [result for result in results if not result.success]

    successful_latencies = [result.latency_ms for result in successful]

    throughput = requests / total_seconds if total_seconds > 0 else 0.0

    error_rate = len(failed) / requests * 100 if requests > 0 else 0.0

    print()
    print("Results")
    print("-" * 40)
    print(f"Successful:    {len(successful)}")
    print(f"Failed:        {len(failed)}")
    print(f"Error rate:    {error_rate:.2f}%")
    print(f"Throughput:    {throughput:.2f} req/s")

    if successful_latencies:
        print()
        print("Latency")
        print("-" * 40)

        print(f"Mean:          " f"{statistics.mean(successful_latencies):.2f} ms")

        print(f"Median:        " f"{statistics.median(successful_latencies):.2f} ms")

        print(f"P50:           " f"{percentile(successful_latencies, 50):.2f} ms")

        print(f"P95:           " f"{percentile(successful_latencies, 95):.2f} ms")

        print(f"P99:           " f"{percentile(successful_latencies, 99):.2f} ms")

        print(f"Min:           " f"{min(successful_latencies):.2f} ms")

        print(f"Max:           " f"{max(successful_latencies):.2f} ms")

    print()

    status_counts: dict[int, int] = {}

    for result in results:
        status_counts[result.status_code] = (
            status_counts.get(
                result.status_code,
                0,
            )
            + 1
        )

    print("HTTP Status Codes")
    print("-" * 40)

    for status_code, count in sorted(status_counts.items()):
        print(f"{status_code}: {count}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Load test the RelevanceFlow ranking API.")
    )

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/rank",
        help="Ranking API endpoint.",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )

    parser.add_argument(
        "--unique",
        action="store_true",
        help=("Use a unique query for every request " "to bypass the Redis cache."),
    )

    args = parser.parse_args()

    run_load_test(
        url=args.url,
        requests=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
        unique=args.unique,
    )


if __name__ == "__main__":
    main()
