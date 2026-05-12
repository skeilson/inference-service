"""
Benchmark — V1 Latency and Throughput
======================================
Three experiments:
  1. Baseline latency    — single requests, no concurrency
  2. Concurrency sweep   — watch latency degrade as load increases
  3. Throughput ceiling  — find the requests/sec saturation point

Run with:
    python benchmark/benchmark.py

Make sure the service is running first:
    uvicorn v1.service:app --port 8000
"""

import time
import statistics
import asyncio
import aiohttp


# ── Config ────────────────────────────────────────────────────────────────────
URL = "http://localhost:8000/embed"

# A handful of different sentences so we're not always sending the same input.
# In a real benchmark you'd use a representative sample of your actual traffic.
SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning models require significant computational resources",
    "inference latency is a critical metric for production systems",
    "transformers have revolutionized natural language processing",
    "the capital of France is Paris",
    "neural networks learn by adjusting weights during training",
    "embeddings represent semantic meaning as vectors in high dimensional space",
    "batch processing improves throughput by amortizing fixed costs",
]


# ── Core request function ─────────────────────────────────────────────────────
# This is an async function — it sends one request and returns how long it took.
# `async` means it can pause and let other requests run while waiting for
# the server to respond. This is how we simulate concurrent clients.
async def send_request(session: aiohttp.ClientSession, sentence: str) -> dict:
    """Send one inference request. Return timing breakdown."""
    payload = {"text": sentence}

    # wall_start is the total time including network overhead
    wall_start = time.perf_counter()

    async with session.post(URL, json=payload) as response:
        data = await response.json()

    wall_ms = (time.perf_counter() - wall_start) * 1000

    return {
        "wall_ms": wall_ms,               # total time client experienced
        "inference_ms": data["inference_ms"],  # time the model took server-side
        "overhead_ms": wall_ms - data["inference_ms"],  # everything else
    }


# ── Experiment 1: Baseline ────────────────────────────────────────────────────
async def experiment_baseline(session: aiohttp.ClientSession, n: int = 10):
    """
    Send requests one at a time, sequentially.
    No concurrency. This is the best case — zero contention.
    """
    print("\n" + "═" * 60)
    print("EXPERIMENT 1 — Baseline (sequential, no concurrency)")
    print("═" * 60)
    print(f"Sending {n} requests one at a time...\n")

    results = []
    for i in range(n):
        # We pick sentences in rotation using modulo
        sentence = SENTENCES[i % len(SENTENCES)]
        result = await send_request(session, sentence)
        results.append(result)
        print(f"  Request {i+1:2d}: {result['wall_ms']:6.1f}ms total  "
              f"({result['inference_ms']:6.1f}ms model  "
              f"{result['overhead_ms']:5.1f}ms overhead)")

    wall_times = [r["wall_ms"] for r in results]
    print(f"\n  p50: {statistics.median(wall_times):.1f}ms")
    print(f"  p95: {sorted(wall_times)[int(len(wall_times) * 0.95)]:.1f}ms")
    print(f"  avg: {statistics.mean(wall_times):.1f}ms")


# ── Experiment 2: Concurrency Sweep ──────────────────────────────────────────
async def experiment_concurrency(session: aiohttp.ClientSession):
    """
    Send requests at increasing concurrency levels.
    Watch what happens to latency as more requests compete for the model.

    This is the key experiment for V1. The model can only handle one
    request at a time, so concurrent requests stack up and wait.
    """
    print("\n" + "═" * 60)
    print("EXPERIMENT 2 — Concurrency Sweep")
    print("═" * 60)
    print("Sending batches of concurrent requests at increasing load...\n")
    print(f"  {'Concurrency':>12}  {'p50 (ms)':>10}  {'p95 (ms)':>10}  {'avg (ms)':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    # These are the concurrency levels we'll test
    # Each level means N requests sent simultaneously
    for concurrency in [1, 5, 10, 25, 50]:

        # Build a list of N coroutines — one per simulated client
        # Each picks a sentence in rotation
        tasks = [
            send_request(session, SENTENCES[i % len(SENTENCES)])
            for i in range(concurrency)
        ]

        # asyncio.gather runs all tasks concurrently and waits for all to finish
        # This simulates N clients hitting the service at the exact same moment
        results = await asyncio.gather(*tasks)

        wall_times = sorted([r["wall_ms"] for r in results])
        p50 = statistics.median(wall_times)
        p95 = wall_times[int(len(wall_times) * 0.95)]
        avg = statistics.mean(wall_times)

        print(f"  {concurrency:>12}  {p50:>10.1f}  {p95:>10.1f}  {avg:>10.1f}")


# ── Experiment 3: Throughput Ceiling ─────────────────────────────────────────
async def experiment_throughput(session: aiohttp.ClientSession):
    """
    Find the maximum sustainable requests per second.

    We send a fixed number of requests as fast as possible and measure
    how many completed per second. This is the throughput ceiling —
    the number we'll compare against V2 and V3.
    """
    print("\n" + "═" * 60)
    print("EXPERIMENT 3 — Throughput Ceiling")
    print("═" * 60)

    total_requests = 50
    print(f"Sending {total_requests} requests as fast as possible...\n")

    tasks = [
        send_request(session, SENTENCES[i % len(SENTENCES)])
        for i in range(total_requests)
    ]

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start

    successful = len(results)
    rps = successful / duration
    wall_times = sorted([r["wall_ms"] for r in results])

    print(f"  Completed:    {successful} requests")
    print(f"  Duration:     {duration:.2f}s")
    print(f"  Throughput:   {rps:.1f} requests/sec")
    print(f"  p50 latency:  {statistics.median(wall_times):.1f}ms")
    print(f"  p95 latency:  {wall_times[int(len(wall_times) * 0.95)]:.1f}ms")
    print(f"  p99 latency:  {wall_times[int(len(wall_times) * 0.99)]:.1f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("\n🔬 V1 Benchmark — Naive Synchronous Inference Service")
    print("Make sure the service is running: uvicorn v1.service:app --port 8000")

    # aiohttp.ClientSession is an async HTTP client.
    # We create one session and reuse it for all requests —
    # this avoids the overhead of creating a new connection every time.
    async with aiohttp.ClientSession() as session:

        # Confirm service is up before starting
        try:
            async with session.get("http://localhost:8000/health") as r:
                assert r.status == 200
                print("\n✓ Service is healthy — starting benchmark\n")
        except Exception:
            print("\n✗ Service not reachable. Start it first.")
            return

        await experiment_baseline(session)
        await experiment_concurrency(session)
        await experiment_throughput(session)

    print("\n" + "═" * 60)
    print("Benchmark complete. Save these numbers — we'll compare")
    print("them directly against V2 and V3.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
