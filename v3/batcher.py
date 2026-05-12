"""
V3 — Dynamic Batching Engine
=============================
This is the core of V3. It sits between FastAPI and the model,
collecting requests into batches and running one forward pass
per batch instead of one per request.

Key concepts:
  - asyncio.Queue: the staging area where requests wait
  - Future: a placeholder that gets filled with a result
  - Batch loop: continuously drains the queue and runs batches
  - MAX_BATCH_SIZE: upper limit on batch size
  - MAX_BATCH_WAIT_MS: how long to wait for more requests
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("inference.v3.batcher")

# ── Configuration ─────────────────────────────────────────────────────────────
# These two parameters are the core tuning knobs of the batching engine.
#
# MAX_BATCH_SIZE: maximum number of requests in one forward pass.
# Beyond this, the batch sends immediately regardless of wait time.
# Larger = better throughput, more memory usage per batch.
#
# MAX_BATCH_WAIT_MS: how long to wait for more requests before
# sending whatever is in the staging area.
# Larger = bigger batches = better efficiency but higher latency
# Smaller = smaller batches = lower latency but less efficiency
MAX_BATCH_SIZE = 32
MAX_BATCH_WAIT_MS = 20


class DynamicBatcher:
    """
    Collects inference requests into batches and processes them
    together in a single model forward pass.

    Each request is represented as a (text, future) pair:
      - text:   the input sentence to embed
      - future: a placeholder the client is waiting on
                filled with the result when the batch completes
    """

    def __init__(self, model, executor: ThreadPoolExecutor):
        # The model and executor are passed in from service.py
        # The batcher doesn't own them — it just uses them
        self.model = model
        self.executor = executor

        # The staging area. Requests sit here waiting to be batched.
        # maxsize=256 means we'll reject requests if the queue is full
        # — this is backpressure, preventing unbounded memory growth
        self.queue = asyncio.Queue(maxsize=256)

        # Track basic stats for logging
        self._batches_processed = 0
        self._requests_processed = 0

    async def infer(self, text: str) -> list[float]:
        """
        Called by FastAPI for each incoming request.
        Creates a Future, puts it in the queue, and waits for the result.
        This is what each client coroutine calls and awaits.
        """
        # Create the Future — the empty box the client will wait on
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Put the request in the staging area
        # If the queue is full, this raises QueueFull immediately
        # giving the client an error rather than waiting forever
        await self.queue.put((text, future))

        # Wait here until the batch engine fills our future with a result
        # The client coroutine is paused at this line
        return await future

    async def _run_batch(self, batch: list[tuple[str, asyncio.Future]]):
        """
        Takes a list of (text, future) pairs, runs them as a single
        forward pass, and distributes results back to waiting clients.
        """
        texts = [text for text, _ in batch]
        futures = [future for _, future in batch]

        log.info(f"Running batch of {len(batch)} requests")

        try:
            # Run model.encode() with the full batch on a worker thread
            # This is the same run_in_executor pattern from V2 —
            # but now we're encoding multiple texts in one call
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self.model.encode,  # encode accepts a list of strings
                texts               # pass the whole batch at once
            )

            # Fan results back out — each future gets its own embedding
            # embeddings[i] corresponds to texts[i] corresponds to futures[i]
            for future, embedding in zip(futures, embeddings):
                if not future.cancelled():
                    future.set_result(embedding.tolist())

            self._batches_processed += 1
            self._requests_processed += len(batch)

        except Exception as e:
            # If the batch fails, notify all waiting clients with the error
            # rather than leaving them waiting forever
            log.error(f"Batch failed: {e}")
            for future in futures:
                if not future.cancelled():
                    future.set_exception(e)

    async def batch_loop(self):
        """
        The heart of the batching engine. Runs continuously as a
        background task for the lifetime of the service.

        Algorithm:
          1. Wait for the first request (up to MAX_BATCH_WAIT_MS)
          2. If one arrives, drain the queue up to MAX_BATCH_SIZE
          3. Run the batch
          4. Repeat forever
        """
        log.info("Batch loop started")

        while True:
            batch = []

            # ── Wait for first request ────────────────────────────────────
            # We don't busy-wait — we block here until something arrives.
            # asyncio.wait_for adds a timeout so we don't block forever.
            try:
                first_item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=MAX_BATCH_WAIT_MS / 1000  # convert ms to seconds
                )
                batch.append(first_item)
            except asyncio.TimeoutError:
                # Nothing arrived in the wait window — loop and try again
                continue

            # ── Drain remaining requests ──────────────────────────────────
            # We have at least one request. Now grab everything else
            # that's already waiting in the queue, up to MAX_BATCH_SIZE.
            # queue.get_nowait() returns immediately — no waiting.
            deadline = time.perf_counter() + (MAX_BATCH_WAIT_MS / 1000)

            while len(batch) < MAX_BATCH_SIZE:
                # Stop draining if we've hit our time window
                if time.perf_counter() >= deadline:
                    break
                try:
                    item = self.queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    # Nothing left in queue — send what we have
                    break

            # ── Run the batch ─────────────────────────────────────────────
            await self._run_batch(batch)

            log.info(
                f"Batch stats — "
                f"total batches: {self._batches_processed}, "
                f"total requests: {self._requests_processed}, "
                f"avg batch size: "
                f"{self._requests_processed / self._batches_processed:.1f}"
            )
