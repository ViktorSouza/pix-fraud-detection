#!/usr/bin/env python3
"""Run a small, reproducible HTTP benchmark against POST /pix/transaction."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import math
import os
import platform
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def make_payload(source_document_number: str, device_id: str) -> dict:
    now = datetime.now(timezone.utc)
    opening_date = (now - timedelta(days=365)).isoformat()
    return {
        "id": f"benchmark-{uuid4()}",
        "amount": 25_000,
        "transaction_date": now.isoformat(),
        "source_account": {
            "branch": "0001",
            "account_number": "88213",
            "opening_date": opening_date,
            "owner": {"document_number": source_document_number},
        },
        "destination_account": {
            "participant": "60701190",
            "branch": "3675",
            "account_number": "123456789012",
            "opening_date": opening_date,
            "owner": {"document_number": "987.654.321-00"},
        },
        "source": {
            "device_id": device_id,
            "channel": "app",
            "ip": "198.51.100.23",
            "geolocation": {"latitude": -23.56, "longitude": -46.65},
        },
    }


def send_request(
    url: str,
    token: str,
    timeout: float,
    source_document_number: str,
    device_id: str,
) -> tuple[float, int | None, str | None]:
    payload = make_payload(source_document_number, device_id)
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "INTERNAL-TOKEN": token,
        },
        method="POST",
    )

    started_at = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            return (perf_counter() - started_at) * 1000, response.status, None
    except HTTPError as error:
        error.read()
        return (perf_counter() - started_at) * 1000, error.code, str(error)
    except (TimeoutError, URLError) as error:
        return (perf_counter() - started_at) * 1000, None, str(error)


def percentile(values: list[float], fraction: float) -> float:
    ordered_values = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered_values)) - 1)
    return ordered_values[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure latency and throughput for Pix transaction analysis."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:3000/pix/transaction",
        help="Target endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("INTERNAL_TOKEN", "default_token"),
        help="INTERNAL-TOKEN (default: INTERNAL_TOKEN env or local default)",
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--source-document", default="123.456.789-00")
    parser.add_argument("--device-id", default="benchmark-device")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests precisa ser pelo menos 1")
    if args.concurrency < 1:
        parser.error("--concurrency precisa ser pelo menos 1")
    if args.timeout <= 0:
        parser.error("--timeout precisa ser maior que zero")
    return args


def main() -> int:
    args = parse_args()
    request_url = args.url.rstrip("/")
    latencies_ms = []
    status_counts: dict[str, int] = {}
    transport_errors = []

    started_at = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                send_request,
                request_url,
                args.token,
                args.timeout,
                args.source_document,
                args.device_id,
            )
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            latency_ms, status_code, error = future.result()
            latencies_ms.append(latency_ms)
            if status_code is None:
                status_counts["transport_error"] = (
                    status_counts.get("transport_error", 0) + 1
                )
                if len(transport_errors) < 3 and error is not None:
                    transport_errors.append(error)
            else:
                status_key = str(status_code)
                status_counts[status_key] = status_counts.get(status_key, 0) + 1

    elapsed_seconds = perf_counter() - started_at
    successful_requests = status_counts.get("200", 0)
    failed_requests = args.requests - successful_requests

    report = {
        "endpoint": request_url,
        "method": "POST",
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_requests_per_second": round(
            args.requests / elapsed_seconds, 2
        ),
        "success_rate_percent": round(
            successful_requests / args.requests * 100, 2
        ),
        "latency_ms": {
            "p50": round(percentile(latencies_ms, 0.50), 2),
            "p95": round(percentile(latencies_ms, 0.95), 2),
            "p99": round(percentile(latencies_ms, 0.99), 2),
            "max": round(max(latencies_ms), 2),
        },
        "http_status_counts": status_counts,
        "machine": {
            "system": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        "notes": [
            "Benchmark local, no Docker Compose com --reload.",
            "Cada requisição cria uma transação persistida no banco.",
            "Os números dependem da máquina, do banco e do estado do modelo.",
        ],
    }
    if transport_errors:
        report["transport_error_examples"] = transport_errors

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if failed_requests == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
