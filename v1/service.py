"""
V1 — Naive Synchronous Inference Service
=========================================
This is our baseline. Nothing is optimized. Every decision here is
intentionally simple so we have a clear starting point to measure from.

Key characteristics:
  - plain `def` endpoint — synchronous and blocking
  - one request is fully handled before the next one starts
  - the model runs inline, on the main thread, for every request

What to watch for when you load test this:
  - latency grows as concurrent requests pile up
  - throughput flatlines quickly
  - one CPU core gets pegged even under heavy load
"""

import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ── Logging ───────────────────────────────────────────────────────────────────
# We're printing a line for every request showing exactly how long
# the model took. This is our manual observability layer for now —
# no Prometheus yet, just logs we can read.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("inference.v1")


# ── Model loading ─────────────────────────────────────────────────────────────
# This runs ONCE when the service starts up, not on every request.
# The model weights (~90MB) are loaded into memory here and stay there.
# First run downloads them. Every run after uses a local cache.
#
# Loading the model is slow (2-5 seconds). Notice that the service
# is completely unavailable during this time — it hasn't started yet.
# This is the cold start problem we'll address in V3.
log.info("Loading model — service unavailable until this completes...")
model = SentenceTransformer("all-MiniLM-L6-v2")
log.info("Model ready")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Inference Service — V1 Naive")


# ── Request / Response shapes ─────────────────────────────────────────────────
# Pydantic models define what we accept and what we return.
# FastAPI uses these to validate incoming JSON and serialize outgoing JSON.
# If a request doesn't match InferenceRequest, FastAPI rejects it automatically.
class InferenceRequest(BaseModel):
    text: str                  # the sentence to embed


class InferenceResponse(BaseModel):
    embedding: list[float]     # 384 numbers representing the meaning of the text
    inference_ms: float        # how long the model took, in milliseconds


# ── The endpoint ──────────────────────────────────────────────────────────────
# Notice: plain `def`, not `async def`.
#
# This is the most natural way to write this — and it's the wrong way
# for an inference service. Here's why:
#
# FastAPI runs plain `def` endpoints in a thread pool (a small set of
# worker threads). When a request comes in, a thread picks it up and
# runs this function. While the model is running, that thread is blocked —
# it can't do anything else. When all threads are busy, new requests
# queue up and wait.
#
# Under concurrent load, this turns into a traffic jam. We'll see
# exactly what that looks like when we run the benchmark.
@app.post("/embed", response_model=InferenceResponse)
def embed(request: InferenceRequest):

    # Time the model specifically — not the whole request.
    # We want to know: is slowness coming from the model, or from somewhere else?
    t0 = time.perf_counter()
    embedding = model.encode(request.text)
    inference_ms = (time.perf_counter() - t0) * 1000

    log.info(f"Embedded {len(request.text)} chars in {inference_ms:.1f}ms")

    # model.encode() returns a numpy array.
    # .tolist() converts it to a plain Python list so JSON serialization works.
    return InferenceResponse(
        embedding=embedding.tolist(),
        inference_ms=inference_ms,
    )


# ── Health check ──────────────────────────────────────────────────────────────
# A simple endpoint that returns 200 if the service is running.
# We'll use this to confirm the service is up before sending real requests.
@app.get("/health")
def health():
    return {"status": "ok"}
