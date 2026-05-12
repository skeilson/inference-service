"""
V2 — Async Inference Service with ThreadPoolExecutor
=====================================================
One targeted change from V1: model.encode() is moved off the
event loop and onto a thread pool worker via run_in_executor.

Key characteristics:
  - `async def` endpoint — non-blocking, yields control at await points
  - model.encode() runs on a worker thread, not the event loop
  - event loop stays free to accept and queue incoming requests
  - multiple requests can be in-flight simultaneously

What to watch for when you benchmark this:
  - latency under concurrency should improve significantly vs V1
  - linear scaling pattern from V1 should break
  - throughput ceiling should rise
  - overhead will increase slightly — run_in_executor has a small cost
"""

import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("inference.v2")


# ── Model loading ─────────────────────────────────────────────────────────────
# Same as V1 — loads once at startup, lives in memory.
# All worker threads share this same model object.
#
# This is safe because model.encode() is read-only during inference —
# it reads the weights but never modifies them. Read-only operations
# are safe to share across threads.
log.info("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
log.info("Model ready")


# ── Thread Pool ───────────────────────────────────────────────────────────────
# This is the key addition in V2.
#
# max_workers controls how many threads can run model.encode() simultaneously.
# Setting it to 4 means up to 4 requests can be in-flight at the same time.
#
# Why 4? It's a starting point — roughly matching typical CPU core counts.
# Too few: requests still queue even though threads are available
# Too many: threads compete for CPU, context switching overhead increases
#
# We'll see the effect of this ceiling in the benchmark — at concurrency
# levels above max_workers, requests will queue waiting for a free thread.
executor = ThreadPoolExecutor(max_workers=4)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Inference Service — V2 ThreadPool")


# ── Request / Response shapes ─────────────────────────────────────────────────
# Identical to V1 — the contract with the client hasn't changed.
class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    embedding: list[float]
    inference_ms: float


# ── The endpoint ──────────────────────────────────────────────────────────────
# Notice: `async def` now, not plain `def`.
#
# This matters because we're using `await` inside — specifically at
# run_in_executor. The await tells the event loop: "hand this work to
# a thread and come back to me when it's done. In the meantime, go
# handle other requests."
#
# Without async def, we couldn't use await.
# Without await, the event loop would still block on model.encode().
# Both pieces are required — async def enables await,
# await enables the event loop to stay free.
@app.post("/embed", response_model=InferenceResponse)
async def embed(request: InferenceRequest):

    t0 = time.perf_counter()

    # This is the critical change from V1.
    #
    # Instead of calling model.encode() directly (which would block),
    # we hand it to the thread pool and await the result.
    #
    # What happens at this await:
    #   1. model.encode() is submitted to a worker thread
    #   2. control returns immediately to the event loop
    #   3. event loop handles other incoming requests
    #   4. when the worker thread finishes, this coroutine resumes
    #   5. embedding now contains the result
    #
    # The event loop is never blocked. This is the fix.
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        executor,           # which thread pool to use
        model.encode,       # what function to run
        request.text        # the argument to pass it
    )

    inference_ms = (time.perf_counter() - t0) * 1000

    log.info(f"Embedded {len(request.text)} chars in {inference_ms:.1f}ms")

    return InferenceResponse(
        embedding=embedding.tolist(),
        inference_ms=inference_ms,
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}
