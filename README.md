This README contains an analysis of this example project and how each of the architectural improvements changed how latency and throughput manifest and can be observed. 

## Table of Contents
- [Executive Summary](#executive-summary)
- [Tech Stack](#tech-stack-applies-to-all-versions)
- [Benchmarking Methodology](#benchmarking-methodology)
- [Version One](#version-one)
  - [Architecture](#architecture)
  - [Results](#version-one-results)
  - [Warm/Cold Distinction](#important-observation---warmcold-distinction)
  - [Analysis](#analyses-for-v1)
- [Version Two](#version-two)
  - [Architecture](#architecture-1)
  - [Results](#version-two-results)
  - [Analysis](#analyses-for-v2)
- [Version Three](#version-three)
  - [Architecture](#architecture-2)
  - [Results](#version-3-results)
  - [Analysis](#analyses-for-v3)
- [Conclusion](#conclusion)
- [How to Run This Yourself](#how-to-run-this-yourself)

# Executive Summary:
Inference services encompass the “doing part” of AI/ML by taking an input and producing an output based on a trained model. These services are able to handle a large volume of requests in a sustainable, scalable way that has broader applications than a single data scientist building and training a model for a singular purpose. Inference companies are able to combat the bottleneck by selling an operational layer in order to make these models usable at scale. 

When we compare and contrast an inference service to a “traditional” or “normal” API request, we can isolate some key differences:
* A normal API request is I/O bound. The CPU spends a large portion of the duration idling, which leaves bandwidth and capacity for additional, other requests. 
* An inference service is CPU bound. The CPU is fully occupied for the duration of the request; there is zero idle time.

This distinction is important and describes the entire business model we’re observing with companies like Replicate, Baseten, and Modal. These organizations are providing the operational layer that model builders and trainers don’t want to manage or are unable to handle at scale. 

This series encompasses several parts and explores how inference at scale truly works. We start with a basic API request and observe how a simple `sentence-transform` model works with a single request, sequential requests, and concurrent requests. The output clearly demonstrates duration, latency, and throughput.

The secondary component introduces a `ThreadPoolExecutor` to move CPU bound inference off the event loop in order to observe how thread pooling impacts latency and throughput under concurrent load.  

Finally, I conclude with a dynamic batching experiment, leveraging an in-service queuing approach to measure and observe the impact. To accomplish this, I group concurrent requests into single forward passes to observe the impact on latency and throughput at scale. 

Upon the conclusion of this experimental series, you’ll (hopefully) understand how to build a simple inference service, but also why the incremental design decisions exist and how they compound and build upon each other. 

## Tech Stack (applies to all versions):

* `FastAPI` - an async framework that allows for acceptance and queuing of inbound requests, even if a service is fully occupied with no available resources. This ensures that all requests are (eventually) processed
* `sentence-transformers` - a lightweight Python library that can run models that convert text into numerical coordinates. Because this provides identical output every time (384 numbers), it’s an ideal library for simple and consistent experimentation
* `all-MiniLM-L6-v2` - the model that sentence-transformers loads and runs. This is a production-grade model that is small enough to run locally on CPU to provide meaningful benchmarking for this series of exercises.
* `uvicorn` - an ASGI server that handles the networking layer for the application logic provided by FastAPI
* `aiohttp` - allows for concurrent requests, in comparison to the requests library, which only handles synchronous/sequential requests

### Benchmarking Methodology:
This first portion comprises three benchmarking experiments. First, I examined ten unique requests, sent sequentially to collect their durations and present observed cold/warm starts. Second, I examined pressure under concurrent load to observe latency increase as concurrency volume increased. Lastly, I examined the throughput to observe how the system would handle sustained load. 

Different sentences were used to best mimic normal traffic and to provide variability to ensure that cached results weren’t returned and to ensure that each request hits the model. Wall time and inference time were separated in order to isolate where the time is distributed to help with profiling. This separation demonstrated that the majority of the time is spent in the model and the revealed overhead is negligible (~1.2ms). This means that any optimizations should be made and applied at the model; the network layer would yield minimal gains.

## Version One

### Architecture:
```
Benchmark script sends request
        ↓
Uvicorn receives raw HTTP bytes (network layer)
        ↓
FastAPI receives and routes to /embed endpoint
        ↓
Pydantic validates input — rejects malformed requests
before they reach the model
        ↓
Request queued for processing
        ↓
model.encode() runs — CPU fully occupied, 
converts sentence to 384 numbers
        ↓
FastAPI serializes response to JSON
        ↓
Uvicorn sends response back over the network
        ↓
Client receives embedding + inference_ms
```

### Version One Results:
```
════════════════════════════════════════════════════════════
EXPERIMENT 1 — Baseline (sequential, no concurrency)
════════════════════════════════════════════════════════════
Sending 10 requests one at a time...
  Request  1:  314.3ms total  ( 310.4ms model    3.9ms overhead)
  Request  2:    9.6ms total  (   8.1ms model    1.5ms overhead)
  Request  3:   14.4ms total  (  13.2ms model    1.2ms overhead)
  Request  4:    8.8ms total  (   7.4ms model    1.4ms overhead)
  Request  5:    9.8ms total  (   8.5ms model    1.3ms overhead)
  Request  6:   10.2ms total  (   8.9ms model    1.2ms overhead)
  Request  7:   10.0ms total  (   8.8ms model    1.1ms overhead)
  Request  8:    9.5ms total  (   8.1ms model    1.3ms overhead)
  Request  9:    9.3ms total  (   7.9ms model    1.3ms overhead)
  Request 10:   13.4ms total  (  12.3ms model    1.2ms overhead)

  p50: 9.9ms
  p95: 314.3ms
  avg: 40.9ms
```

**Findings from V1 Experiment One:** The warm baseline is 9ms, which is snappy and I can observe that the requests are stable and consistent. There’s two notable spikes – Requests 3 and 10, but it’s probable that those may be contributed to OS jitter. Request 1 is defined by the cold start and thus skews the `p95`.

```
════════════════════════════════════════════════════════════
EXPERIMENT 2 — Concurrency Sweep
════════════════════════════════════════════════════════════
| Concurrency | p50 (ms) | p95 (ms) | avg (ms) |
|-------------|----------|----------|----------|
| 1           | 9.9      | 9.9      | 9.9      |
| 5           | 34.9     | 36.2     | 35.0     |
| 10          | 66.3     | 68.3     | 66.6     |
| 25          | 174.9    | 178.7    | 173.3    |
| 50          | 317.3    | 364.5    | 321.0    |
```

**Findings from V1 Experiment 2** This experiment clearly demonstrates linear scaling (latency grows proportionally) with concurrency. This is also an excellent demonstration of zero parallelism.

```
════════════════════════════════════════════════════════════
EXPERIMENT 3 — Throughput Ceiling
════════════════════════════════════════════════════════════
  Completed:    50 requests
  Duration:     0.38s
  Throughput:   131.3 requests/sec

  p50 latency:  322.8ms
  p95 latency:  378.6ms
  p99 latency:  379.1ms
```

**Findings from V1 Experiment 3** The throughput of 131 requests per second seems good until you realize it’s 32x slower than the warm baseline from Experiment 1. This demonstrates that high throughput with acceptable latency are mutually exclusive in V1.

### Important Observation - Warm/Cold Distinction:
Every time a new inference request is submitted, there is a “cold start.” As a result, the very first request will be noticeably larger in duration compared to following requests; this is normal and expected. In addition, PU caching, PyTorch computation, and tokenization are important considerations. When a model runs for the first time, nothing is in cache and therefore it must be fetched from RAM and initialized, thus adding to the duration. By the second request, the cached information is ready for rapid use; the weights (computation patterns) are ready in cache. Now, the first time a model is run, PyTorch will need to analyze the sequence of operations and build an execution plan. This analysis is a one-time action on the first request and is reused on subsequent requests, which adds to the duration. Lastly, the tokenizer must convert the text in this experiment into math and this also takes time to initialize. Once initialized, subsequent requests are able to use the loaded pattern which is sitting in memory. All of these factors pile upon each other and contribute to the noticeable initial delay.

Here is a small example of several sequential requests to a cold CPU:
```
  Request  1:  314.3ms total  ( 310.4ms model    3.9ms overhead)
  Request  2:    9.6ms total  (   8.1ms model    1.5ms overhead)
  Request  3:   14.4ms total  (  13.2ms model    1.2ms overhead)
  Request  4:    8.8ms total  (   7.4ms model    1.4ms overhead)
```

Note that the duration of Request 1 is significantly larger than those of Requests 2, 3, and 4. This is the cold start in visual form. Additionally, it is worth noting the skew points:
```
  p50: 9.9ms
  p95: 314.3ms
  avg: 40.9ms
```

While the `p95` value is seemingly alarming, it’s merely reflecting the cold start duration.

Most robust production infrastructure deployments will approach this in one of two ways:
* always have a single warm instance (i.e. the replica is set to a value of 1) to effectively eliminate this cold start delay.
* pre-warming - this method allows for a synthetic inference request to run upon startup to “pay the tax” prior to any real traffic and requests.

### Analyses for V1:
Version 1 demonstrates several core architectural deficiencies. First, I can clearly observe that the initial experiment is fully CPU bound due to the lack of await points within the `model.encode()`. This also means requests queue sequentially and due to the lack of parallelism, I can clearly see the manifestation of linear latency scaling. Secondly, I can observe that the overhead (from Experiment 1) is negligible at `1.2ms`. Profiling confirms that the model layer is the best target for optimization. Lastly, it is important to note that the current `FastAPI` queuing is not fault-tolerant; if the inference service crashes, so does the queue, and requests are flushed. An appropriate, robust production system would have a separate queuing system. 

## Version Two

### Summary:
In V1, I established clearly visible benchmarks using only a single worker to observe how the system handles a volume of sequential requests, concurrent requests, and throughput. In V2, I changed the approach to introduce two notable improvements:
* Added concurrency by handing `model.encode()` off to a `ThreadPoolExecutor` using `run_in_executor`.
* Configured an `await` to move the CPU bound work off of the event loop. This allows the event loop to accept and queue incoming requests while the worker threads handle the inference

By setting the worker quantity to a value of 4, I then endeavored to observe how the benchmarking values changed with this architectural tweak. The hope was that I could observe a significant, measurable improvement in latency, especially with concurrent requests.

### Architecture:
```
Benchmark script sends request
        ↓
Uvicorn receives raw HTTP bytes (network layer)
        ↓
FastAPI receives and routes to /embed endpoint
        ↓
Pydantic validates input — rejects malformed requests
before they reach the model
        ↓
Request queued for processing with the event loop
        ↓
model.encode() hands request off to a ThreadPoolExecutor and is free for the next request
      ↓
Worker thread handles CPU bound work (inference)
        ↓
await resumes the coroutine with the result
      ↓
converts sentence to 384 numbers
        ↓
FastAPI serializes response to JSON
        ↓
Uvicorn sends response back over the network
        ↓
Client receives embedding + inference_ms
```

### Version Two Results:
```
════════════════════════════════════════════════════════════
EXPERIMENT 1 — Baseline (sequential, no concurrency)
════════════════════════════════════════════════════════════
Sending 10 requests one at a time...

  Request  1:   80.1ms total  (  79.0ms model    1.1ms overhead)
  Request  2:   29.2ms total  (  28.2ms model    1.0ms overhead)
  Request  3:   28.3ms total  (  27.4ms model    0.9ms overhead)
  Request  4:    9.1ms total  (   8.2ms model    0.9ms overhead)
  Request  5:   27.5ms total  (  26.6ms model    0.9ms overhead)
  Request  6:   29.0ms total  (  28.1ms model    0.9ms overhead)
  Request  7:   29.2ms total  (  28.2ms model    0.9ms overhead)
  Request  8:   28.0ms total  (  27.0ms model    1.0ms overhead)
  Request  9:   10.0ms total  (   9.0ms model    1.0ms overhead)
  Request 10:    9.2ms total  (   8.3ms model    0.9ms overhead)

  p50: 28.2ms
  p95: 80.1ms
  avg: 27.9ms
```

**Findings from V2 Experiment 1:** I can clearly observe that the `p50` value has increased from the V1 benchmarking results. If I look at Request 4, Request 9, and Request 10 against the sentence input array, I can observe that they are the shortest available sentences, and thus are processed quickly. The longer sentences demonstrate input length sensitivity in a visual capacity. Because the sequential requests in V1 were an unrealistic production representation, it was actually masking this pattern. The introduction of thread pools is a cleaner architectural design and is exposing a more realistic data result.

```
════════════════════════════════════════════════════════════
EXPERIMENT 2 — Concurrency Sweep
════════════════════════════════════════════════════════════
| Concurrency | p50 (ms) | p95 (ms) | avg (ms) |
|-------------|----------|----------|----------|
| 1           | 9.2      | 9.2      | 9.2      |
| 5           | 29.2     | 37.1     | 30.4     |
| 10          | 53.7     | 68.7     | 47.1     |
| 25          | 107.2    | 167.9    | 101.5    |
| 50          | 189.4    | 318.4    | 183.1    |
```

**Findings from V2 Experiment 2:** I can observe that linear latency still exists, but the numbers are meaningfully smaller. This is directly correlated to the introduction of thread pool workers and the ability for queued requests to be processed across four available CPU bound workers, rather than a singular handler. By improving the code to allow `model.encode()` to hand off to these workers, I can see that the system is now free to process and queue incoming requests. So, why didn’t this improve more significantly? I've configured our available worker count to a quantity of 4, meaning the system can process 4 requests at a time. Therefore, the system remains constrained by the quantity of available workers. However, increasing the count of available workers too much introduces other concerns - namely cost and resource drain.

```
════════════════════════════════════════════════════════════
EXPERIMENT 3 — Throughput Ceiling
════════════════════════════════════════════════════════════
Sending 50 requests as fast as possible...

  Completed:    50 requests
  Duration:     0.33s
  Throughput:   150.6 requests/sec

  p50 latency:  184.1ms
  p95 latency:  316.1ms
  p99 latency:  330.3ms
```

**Findings from V2 Experiment 3:** This is better than V1 - the throughput is higher and the latency is lower due to the introduction of the workers, reducing the multiplier from 32 to 20. However, the thread pool workers didn’t completely eliminate the problem of high throughput with acceptable latency. It helped, yes, but due to the hard ceiling, I'm still seeing an undesirable result for the end user; end users at peak throughput will still experience `184ms` against the `9ms` warm baseline.

### Analyses for V2:
Version 2 adds noticeable improvements to throughput and latency, but still lacks some architectural finesse. The fundamental limitation is that each request is still paying the full fixed cost, independently across each worker. The cost is paid 4 times in parallel rather than just once, upfront. This could be alleviated by batching the requests to remove redundant CPU work and load the weight matrices and initialize the computation graph upfront. Secondly, I can see a secondary concern with V2 - input length sensitivity. The model processes sentences of varying length, and therefore the inference time varies; shorter sentences complete faster than longer ones. Because V2 uses a FIFO queue, the latency becomes unpredictable due to the fact that response times depend not just on the input, but also what queued and arrived prior. This is a significant concern when considering a reliable latency SLO. Lastly, as described in V1, fault tolerance remains non-existant; the in-process queue is still coupled to the inference service. As noted in V1, an appropriate, robust system would have a separate queuing service. 

## Version Three

### Summary:
In V2, I added thread pooling to help distribute load, but continued to observe high throughput at the cost of acceptable latency. In V3, I changed the architecture to add in batching in order to observe how the system will handle volumes of batched requests with the hopes of improving both concurrency and throughput while maintaining good sequential results. In V3, I changed the approach to introduce two notable architectural improvements:
* Added dynamic batching that stages and collect requests, groups them, and then runs a single forward pass with the batch, rather than one per request
* Configured a Future to ensure requests are returned to the appropriate client upon completion

By implementing both of these architectural changes, I can clearly see marked improvement in the results of the benchmarking tests. 

### Architecture:
```
Benchmark script sends request
        ↓
Uvicorn receives raw HTTP bytes (network layer)
        ↓
FastAPI receives and routes to /embed endpoint
        ↓
Pydantic validates input — rejects malformed requests
before they reach the model
        ↓
Request enters dynamic batching engine
        ↓
Future created — client pauses here waiting for result
        ↓
Request sits in staging area until batch fills or timer expires
        ↓
model.encode() runs single forward pass on entire batch
via ThreadPoolExecutor worker thread
        ↓
Worker thread handles CPU bound work (inference)
        ↓
await resumes the coroutine with the batch result
        ↓
Future filled with result — correct client resumes automatically
        ↓
converts sentences to 384 numbers per request
        ↓
FastAPI serializes response to JSON
        ↓
Uvicorn sends response back over the network
        ↓
Client receives embedding + inference_ms
```

### Version 3 Results:
```
════════════════════════════════════════════════════════════
EXPERIMENT 1 — Baseline (sequential, no concurrency)
════════════════════════════════════════════════════════════
Sending 10 requests one at a time...

  Request  1:  309.3ms total  ( 306.8ms model    2.5ms overhead)
  Request  2:   11.2ms total  (   9.8ms model    1.4ms overhead)
  Request  3:   14.9ms total  (  13.9ms model    1.0ms overhead)
  Request  4:    9.8ms total  (   8.8ms model    1.0ms overhead)
  Request  5:   10.0ms total  (   9.0ms model    1.0ms overhead)
  Request  6:   10.8ms total  (   9.9ms model    0.9ms overhead)
  Request  7:   10.4ms total  (   9.5ms model    0.9ms overhead)
  Request  8:   10.2ms total  (   9.3ms model    0.9ms overhead)
  Request  9:   10.8ms total  (   9.9ms model    0.9ms overhead)
  Request 10:   10.1ms total  (   9.2ms model    0.9ms overhead)

  p50: 10.6ms
  p95: 309.3ms
  avg: 40.7ms
```

**Findings from V3 Experiment 1:** Here, I can observe that the sequential baseline is back to a performance that is in line with V1 - about `10ms` per request. This is a meaningful observation because it demonstrates that batching doesn’t impact sequential performance. Additionally, I can observe improved latency consistency across the requests. Recall that in V2 I noted how the latency increased due to input latency sensitivity? Here, I can see this has been smoothed out because while input length does have impact, the overhead is more consistent and the use of batching with the single forward pass helps negate that input variability. 

```
════════════════════════════════════════════════════════════
EXPERIMENT 2 — Concurrency Sweep
════════════════════════════════════════════════════════════
| Concurrency | p50 (ms) | p95 (ms) | avg (ms) |
|-------------|----------|----------|----------|
| 1           | 13.4     | 13.4     | 13.4     |
| 5           | 31.5     | 31.6     | 27.6     |
| 10          | 29.4     | 29.8     | 27.9     |
| 25          | 40.6     | 41.9     | 39.5     |
| 50          | 43.8     | 58.5     | 47.4     |
```

**Findings from V3 Experiment 2:** In Experiment 2, I can observe there is a latency and efficiency tradeoff that becomes visibly apparent. When I compare the results for `concurrency=1` across V2 and V3, I can see V3 has increased by ~`3ms`. This is due to the batching engine’s defined `MAX_BATCH_SIZE` (32 requests) and the `MAX_BATCH_WAIT_MS` (`20ms`). Despite this, I can also clearly see that implementing batching has effectively removed linear scaling latency by grouping multiple requests into a single forward pass. That fixed cost is now being paid once per batch rather than once per request. As such, the times for each increasing concurrency value only increases incrementally as the quanity of requests increases.

Compare the differences:
```
V1: concurrency 50 → 317ms
V2: concurrency 50 → 189ms  
V3: concurrency 50 → 43ms
```

```
════════════════════════════════════════════════════════════
EXPERIMENT 3 — Throughput Ceiling
════════════════════════════════════════════════════════════
Sending 50 requests as fast as possible...

  Completed:    50 requests
  Duration:     0.06s
  Throughput:   831.4 requests/sec

  p50 latency:  44.1ms
  p95 latency:  58.3ms
  p99 latency:  58.5ms
```

**Findings from V3 Experiment 3:** The quantity of requests is much higher in the defined time period! 831 requests as compared to 150 in V2 and 131 in V1. At 831 requests per second, users are still experiencing only `44ms` `p50`, compared to `322ms` in V1, and `184ms` in V2 at far lower throughput. This observed, marked improvement is directly related to the implementation of the batching engine and it shows. What’s interesting here is that if I look at the multiplier, it’s only 19. V2’s multiplier was 20, but the latency was apparent as latency and throughput were proportional. By adjusting the architecture, the throughput and latency have been decoupled and are no longer mutually exclusive. 

### Analyses for V3:
Version 3 introduced dynamic batching as an architectural approach to improve latency while maintaining a high throughput. This was successful, as I can clearly observe in the results of the experiments. However, there are still improvements to be made and considered. First, the in-process queue concern remains; in an ideal production deployment, the queuing service would be decoupled from the inference service in order to introduce a more fault tolerant deployment. Secondly, the use of static configurations are great for this experimental use case, but are not appropriate for a production system as hardcoded values cannot adapt to variability in traffic patterns and volume. This could be resolved by the implementation of adaptive batching. Lastly, head-of-line blocking (when a batch contains mixed input lengths, shorter requests get held up behind longer requests) is still a concern; this was exposed in V2 and persists even with a batching engine. Long term, the best approach would be to include some sort of length aware batching to sort by input length prior to batching. 

# Conclusion:
This series of experiments explored how inference is CPU bound and how this limitation has measurable and visible impact on requests, latency, and throughput. In V1, the data clearly demonstrates linear scaling; when there are single sequential requests moving through the services, the single worker is fully CPU bound and can only handle one request at a time. This is most prominent in V1 Experiment Two as demonstrated below:
```
════════════════════════════════════════════════════════════
V1 EXPERIMENT 2 — Concurrency Sweep
════════════════════════════════════════════════════════════
| Concurrency | p50 (ms) | p95 (ms) | avg (ms) |
|-------------|----------|----------|----------|
| 1           | 9.9      | 9.9      | 9.9      |
| 5           | 34.9     | 36.2     | 35.0     |
| 10          | 66.3     | 68.3     | 66.6     |
| 25          | 174.9    | 178.7    | 173.3    |
| 50          | 317.3    | 364.5    | 321.0    |
```

Now, in V2, I made an architectural change to introduce a thread pool approach to try to distribute the work across a pool of 4 workers. The hope was that this would dramatically reduce latency by moving the CPU bound work off the single available worker. This was moderately successful, but there is still a fixed cost per request. This is most visible and prominent in V2 Experiment 2:
```
════════════════════════════════════════════════════════════
V2 EXPERIMENT 2 — Concurrency Sweep
════════════════════════════════════════════════════════════
| Concurrency | p50 (ms) | p95 (ms) | avg (ms) |
|-------------|----------|----------|----------|
| 1           | 9.2      | 9.2      | 9.2      |
| 5           | 29.2     | 37.1     | 30.4     |
| 10          | 53.7     | 68.7     | 47.1     |
| 25          | 107.2    | 167.9    | 101.5    |
| 50          | 189.4    | 318.4    | 183.1    |
```

Linear scaling is still observable, but at a much smaller increment. Lastly, in V3 I re-architected the system to introduce a dynamic batching engine to try to amortize the fixed cost across groups (batches) of requests. Due to this approach, the matrix multiplication scales sub-linearly by leveraging batch size. The results make this apparent:
```
════════════════════════════════════════════════════════════
V3 EXPERIMENT 2 — Concurrency Sweep
════════════════════════════════════════════════════════════
| Concurrency | p50 (ms) | p95 (ms) | avg (ms) |
|-------------|----------|----------|----------|
| 1           | 13.4     | 13.4     | 13.4     |
| 5           | 31.5     | 31.6     | 27.6     |
| 10          | 29.4     | 29.8     | 27.9     |
| 25          | 40.6     | 41.9     | 39.5     |
| 50          | 43.8     | 58.5     | 47.4     |
```

This is a significant, measurable improvement, particularly when concurrency=50. 

Lastly, I can compare across all three versions by looking at Experiment 3. On the same hardware, with the same model, I clearly observed a 6x throughput improvement across all three versions. Throughput went from 131 requests per second in V1 to 150 requests per second in V2 to a whopping 831 requests per second in V3 while also reducing the `p50` latency across the board.

With all these adjustments and changes across versions, this is still not an appropriate production-ready approach. An ideal production system would decouple the queuing service from the main service in order to introduce a more robust, fault-tolerant system; if the service topples, the queue remains intact and requests can resume once the service is restored. Additionally, a more appropriate production system would consider adaptive batching and length aware batching to better handle variability in traffic pattern spikes and input length variability. Looking forward, one should consider the next logical step - GPU. Improving the hardware by introducing parallelism would have a dramatic impact on efficiency. 


# How to Run This Yourself

### Prerequisites
- Python 3.12+
- Git

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/skeilson/inference-service.git
cd inference-service
```

**2. Create and activate the virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
**Note:** This will download PyTorch and sentence-transformers (~500MB). First install will take a few minutes.

### Running the Service

You'll need two terminal windows, both with `.venv` active.

**Terminal 1 — Start the service**

For V1:
```bash
uvicorn v1.service:app --port 8000
```

For V2:
```bash
uvicorn v2.service:app --port 8000
```

For V3:
```bash
uvicorn v3.service:app --port 8000
```

**Terminal 2 — Validate the service is running**
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "the quick brown fox"}'
```

You should receive a JSON response with 384 numbers and an `inference_ms` value.

### Running the Benchmark

With the service running in Terminal 1, run the following in Terminal 2:
```bash
python benchmark/benchmark.py
```

The benchmark runs three experiments automatically and prints results to the terminal. Repeat for each version to compare results directly.
