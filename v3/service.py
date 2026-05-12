"""
V3 — Inference Service with Dynamic Batching
=============================================
The service layer is now minimal — its only job is to accept
requests and hand them to the batcher. All the interesting
logic lives in batcher.py.
"""

import time
import logging
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from v3.batcher import DynamicBatcher


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("inference.v3")


# ── Lifespan ──────────────────────────────────────────────────────────────────
# FastAPI's lifespan manages startup and shutdown.
# Everything in the `async with` block runs for the life of the service.
#
# We use this instead of global variables because it gives us clean
# startup and shutdown hooks — the batch loop starts when the service
# starts and stops when it stops.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    log.info("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Model ready")

    executor = ThreadPoolExecutor(max_workers=4)
    batcher = DynamicBatcher(model, executor)

    # Start the batch loop as a background task
    # It runs concurrently with the FastAPI app for the service lifetime
    batch_task = asyncio.create_task(batcher.batch_loop())

    # Make the batcher available to endpoints via app.state
    app.state.batcher = batcher

    log.info("Batch loop running — service ready")

    yield  # Service runs here

    # ── Shutdown ─────────────────────────────────────────────────────────
    log.info("Shutting down...")
    batch_task.cancel()
    executor.shutdown(wait=True)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Inference Service — V3 Dynamic Batching", lifespan=lifespan)


# ── Request / Response shapes ─────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    embedding: list[float]
    inference_ms: float


# ── Endpoint ──────────────────────────────────────────────────────────────────
# Notice how simple this is compared to V1 and V2.
# The endpoint's only job is to hand the request to the batcher
# and wait for the result. All complexity is in batcher.py.
@app.post("/embed", response_model=InferenceResponse)
async def embed(request: InferenceRequest):
    t0 = time.perf_counter()

    try:
        embedding = await app.state.batcher.infer(request.text)
    except asyncio.QueueFull:
        # Queue is full — service is overloaded, reject the request
        # 503 Service Unavailable is the correct HTTP status here
        raise HTTPException(status_code=503, detail="Service overloaded")

    inference_ms = (time.perf_counter() - t0) * 1000

    log.info(f"Embedded {len(request.text)} chars in {inference_ms:.1f}ms")

    return InferenceResponse(
        embedding=embedding,
        inference_ms=inference_ms,
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}
