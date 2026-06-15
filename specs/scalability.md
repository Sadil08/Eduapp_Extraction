# Scalability notes: AI_Extraction_Marking_service
**Score:** 0.12
## Summary
The service exhibits multiple severe scalability and reliability anti-patterns that will cause observable failures under any meaningful production load. The most critical issues are: a hardcoded API credential committed to source, a local-filesystem CSV token log that makes horizontal scaling impossible (each replica writes to its own disconnected file with no locking), synchronous blocking Vertex AI SDK calls made inside a FastAPI async application that will starve the event loop under concurrency, a thread-unsafe singleton initialisation that races under concurrent startup, and zero retry/backoff logic on all external AI API calls. No caching, no rate-limiting, and tight synchronous request-response coupling to a high-latency LLM further compound latency and cost at scale. The service is essentially a single-instance, synchronous script dressed as a web service.
## Concerns
### [CRITICAL] stateful — list_models.py
A live Google Generative AI API key (`***REMOVED-GOOGLE-KEY***`) is hardcoded in source. Any process, replica, or person with repo access can exhaust quota, incur unbounded cost, or exfiltrate data. This also makes key rotation impossible without a code deploy.

**Fix:** Remove the key immediately, rotate it in the Google Cloud Console, and source it exclusively from a secret manager (GCP Secret Manager, env var injected at runtime). Never commit credentials.
### [CRITICAL] stateful — token_logger.py
Token usage is appended to a local CSV file on the instance's filesystem (`logs/token_usage_YYYY_MM.csv`). Running more than one replica means each instance accumulates its own disconnected ledger — usage data is silently split, incomplete, and unrecoverable if a container restarts. Additionally, concurrent requests within a single process write to the same file without any file lock, causing interleaved or truncated rows under load.

**Fix:** Replace CSV file writes with atomic writes to a shared store: BigQuery (fits naturally with GCP), Cloud Spanner, or at minimum a Cloud Storage object with generation-match preconditions. Add an asyncio lock or use a queue-based writer for intra-process safety.
### [CRITICAL] concurrency — main.py / model_loader.py
FastAPI uses an async event loop (uvicorn). The Vertex AI SDK's `model.generate_content()` is a synchronous, blocking network call — typically 2–30 seconds for LLM inference. Calling it directly inside a request handler (sync or async without `run_in_executor`) blocks the entire event loop, preventing all other in-flight requests from making progress. Under any concurrency > 1, latency collapses and timeouts cascade.

**Fix:** Wrap every blocking Vertex AI SDK call in `asyncio.get_event_loop().run_in_executor(None, ...)` or use `loop.run_in_executor` with a bounded `ThreadPoolExecutor`. Alternatively adopt the async Vertex AI client if/when available, or use a task queue (Cloud Tasks / Pub/Sub) to offload inference entirely.
### [CRITICAL] concurrency — model_loader.py — ModelLoader.load_model()
The singleton guard is a bare `if self._model is None` check with no lock. Under concurrent startup (multiple threads or the first burst of requests hitting an unwarmed instance), multiple threads will simultaneously observe `_model is None`, each call `vertexai.init()` and construct a `GenerativeModel`, resulting in redundant initialisation at best and corrupted shared state at worst. Class-level `_instance` / `_model` mutation also creates shared mutable global state.

**Fix:** Protect the initialisation block with `threading.Lock()` using double-checked locking, or use `functools.lru_cache` on a module-level factory function. Initialise the model eagerly at application startup (FastAPI `lifespan` event) rather than lazily on first request.
### [HIGH] coupling — main.py
Every extraction/marking request is synchronously coupled end-to-end to a remote LLM call (p99 latency 10–60 s). The HTTP client must hold an open connection for the entire duration. This means worker/connection slots are exhausted quickly under moderate concurrency, and any Vertex AI degradation directly stalls all user-facing requests with no isolation.

**Fix:** Move LLM inference to an async job queue (Cloud Tasks, Pub/Sub, or Celery with a Cloud broker). Accept the request, enqueue the job, return a job ID immediately (HTTP 202). Clients poll or receive a webhook callback. This decouples request handling from inference latency and allows independent scaling of the worker pool.
### [HIGH] connection — main.py / model_loader.py
There is no retry, exponential backoff, or circuit-breaker logic around any Vertex AI API call. Transient 429 (quota), 500, or 503 responses from Google APIs are surfaced directly as errors to the caller. Under load, quota exhaustion or momentary API hiccups will produce a flood of unhandled exceptions with no self-healing.

**Fix:** Wrap all `generate_content` calls with a retry decorator (e.g., `tenacity`) configured with exponential backoff, jitter, and a maximum retry count. Respect `Retry-After` headers on 429 responses. Add a circuit breaker (e.g., `pybreaker`) to shed load fast when the upstream is degraded.
### [HIGH] caching — main.py
Identical or near-identical extraction requests (same image/document, same subject) each invoke a full LLM round-trip. At scale, repeated submissions of the same exam page will multiply inference cost linearly with no benefit. There is no content-addressed cache or deduplication layer.

**Fix:** Compute a deterministic hash (SHA-256) of the input bytes + prompt key. Cache the LLM response in Redis (or Cloud Memorystore) with a reasonable TTL (e.g., 24 h). On a cache hit, return immediately. This can reduce inference costs by 40–80% on workloads with repeated content.
### [HIGH] stateful — token_logger.py — TokenLogger.__init__()
The `TokenLogger` instance resolves the log file path at construction time using `datetime.now()`. In a long-running process that spans a month boundary, it will continue writing to the previous month's file until the process restarts. More critically, the log directory is created relative to `__file__`, tying the service to a writable local filesystem — incompatible with read-only container images or multi-replica deployments.

**Fix:** Resolve the log file path per write call (not at construction). Replace local file I/O with a remote sink (BigQuery streaming insert, Cloud Logging structured log, or a metrics service). If local files must be used in dev, make the path configurable via env var.
### [MEDIUM] concurrency — token_logger.py
CSV row writes use a plain `open()` + `csv.writer` with no file lock. Concurrent requests in the same process will interleave writes, producing malformed rows or silent data loss under load.

**Fix:** Use a `threading.Lock` around file writes, or — better — collect log entries in an in-process queue and flush them with a single background writer thread, then migrate to a proper remote sink.
### [MEDIUM] caching — model_loader.py / check_models_script.py
`vertexai.init()` is called at module import time in `check_models_script.py` and inside `load_model()` in the loader. If model health checks or re-initialisations are triggered per-request (e.g., after an error resets state), this incurs repeated SDK setup overhead and unnecessary API round-trips.

**Fix:** Call `vertexai.init()` exactly once at application startup (FastAPI lifespan). Remove `check_models_script.py` from any code path that runs per-request; use it only as an offline diagnostic tool.
### [MEDIUM] memory — main.py
Uploaded files (`UploadFile`) and their base64/PIL conversions are held in memory for the duration of the request. With large exam images (multi-MB TIFFs or PDFs) and concurrent submissions, per-request heap usage will spike. No visible limit is enforced on upload size or number of files per request.

**Fix:** Set a maximum upload size via FastAPI's `File(...)` constraints or an upstream nginx `client_max_body_size`. Stream large files to Cloud Storage and pass the GCS URI to the Vertex AI multimodal API instead of inlining bytes in the request payload.
