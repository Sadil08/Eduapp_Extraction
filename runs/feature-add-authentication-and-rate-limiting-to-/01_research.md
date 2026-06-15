# Research: Add Authentication and Rate Limiting to All Endpoints

## Implementation patterns found

### Pattern A: API Key Header via FastAPI Security Dependency

- **Approach:** Use `fastapi.security.APIKeyHeader` to extract a key from `X-API-Key` on every request. A single `verify_key` async dependency is applied at the router or `app` level so every current and future route is protected automatically. Keys are stored as hashed values in environment variables (or a secrets store) and compared with `secrets.compare_digest` to prevent timing attacks.
- **Pros:**
  - Minimal code footprint; integrates natively with FastAPI's `Depends()` system
  - Applying at the router/app level ensures new routes cannot be accidentally left unprotected
  - No external auth infrastructure required — compatible with the project's env-var configuration pattern already used by `python-dotenv`
  - Straightforward to rotate keys without redeploying code
- **Cons:**
  - No user-level identity — all callers share the same privilege level unless multiple keys are issued
  - Key must be transmitted on every request; if TLS is not enforced end-to-end the key is exposed in transit
  - No built-in expiry semantics; revocation requires changing the env variable and restarting
- **Fits this project: Yes** — The security spec explicitly recommends this as the minimum viable fix for `main.py`. The project already has environment-variable-driven configuration via `python-dotenv`, so adding `SERVICE_API_KEY` to `.env` and GCP Secret Manager slots naturally into the existing pattern.

---

### Pattern B: GCP Identity-Aware Proxy (IAP) + Service Account Validation

- **Approach:** Deploy the FastAPI service behind Google Cloud's Identity-Aware Proxy. IAP validates a Google-signed OIDC `JWT` in the `X-Goog-IAP-JWT-Assertion` header before requests reach the application. The application verifies the assertion using `google-auth` (`google.auth.transport.requests` + `google.oauth2.id_token.verify_token`). Internal service-to-service calls use Workload Identity or service account tokens.
- **Pros:**
  - Zero-trust at the infrastructure layer; unauthenticated requests never reach Python code
  - Leverages GCP IAM roles the project already depends on (Vertex AI IAM is already in scope)
  - Audit trail provided by Cloud Audit Logs at no extra code cost
  - Supports Google Workspace identity, removing the need to manage credentials entirely
- **Cons:**
  - Requires IAP to be configured and enabled on the Cloud Run / GKE / Cloud Functions deployment — introduces infra coupling
  - Local development requires emulation (`IAP_DEV_JWT` bypass or `iap_curl`), adding developer friction
  - Not applicable if the service is called by non-Google-identity clients (e.g., third-party systems)
- **Fits this project: Conditionally** — The service is cloud-native on GCP and the architecture is explicitly GCP-first, making IAP a strong production target. However it requires infra changes beyond code and is a second-phase hardening step rather than an immediate fix.

---

### Pattern C: JWT Bearer Token (OAuth2 Password / Client Credentials flow)

- **Approach:** Use `fastapi.security.OAuth2PasswordBearer` or `OAuth2ClientCredentials`. Issue short-lived JWTs signed with `RS256`; validate via `python-jose` on every request. Supports per-user identity and role claims.
- **Pros:**
  - Stateless; no database lookup required on each request
  - Token expiry is built in (short-lived access tokens)
  - Can carry per-user rate-limit tier in claims
- **Cons:**
  - Requires an auth issuer (Google Identity Platform, Auth0, or a custom `/token` endpoint) — significant scope increase for a service with no existing auth infrastructure
  - RS256 key rotation is operationally non-trivial
  - Overkill for a backend AI service likely called by internal systems only
- **Fits this project: No (short-term)** — There is no user management, no DB, and no existing identity provider observed. This pattern is the right long-term direction if the service ever exposes a consumer-facing API, but is disproportionate for the immediate critical/high findings.

---

### Pattern D: `slowapi` Application-Layer Rate Limiting

- **Approach:** `slowapi` wraps `limits` and is the de-facto FastAPI rate-limiting library. A `Limiter` is instantiated with `key_func=get_remote_address` (or a custom key function based on the authenticated API key). Limits are declared as decorators (`@limiter.limit("10/minute")`) or applied globally via middleware. Backend storage can be in-memory (single instance) or Redis (multi-instance).
- **Pros:**
  - Explicitly recommended in the security specification for `main.py`
  - Native FastAPI/Starlette integration; returns `429 Too Many Requests` with `Retry-After` header automatically
  - Per-key limiting is achievable by swapping the `key_func` to extract the `X-API-Key` header
  - Redis backend enables correct limiting across multiple service replicas
- **Cons:**
  - In-memory backend is not safe for multi-replica deployments — limits would be per-pod
  - `get_remote_address` key is trivially bypassed behind a load balancer if `X-Forwarded-For` is not normalised
  - Does not protect against distributed attacks originating from many IPs
- **Fits this project: Yes** — Directly matches the architecture (FastAPI, Python) and is specifically called out in the security spec. Must use Redis backend if more than one replica is deployed.

---

## Recommended approach

**Immediate (addresses critical/high findings):** Implement **Pattern A** (API Key Header) combined with **Pattern D** (slowapi rate limiting).

1. Add `SERVICE_API_KEY_HASH` (bcrypt or SHA-256 hash of the key) to `.env` and GCP Secret Manager.
2. Create a `security.py` module with the `verify_key` dependency; apply it at the `APIRouter` level so all current routes and future routes inherit it.
3. Instantiate `slowapi.Limiter` with Redis backend (`redis://localhost:6379` locally, Cloud Memorystore in GCP). Apply conservative limits: `10/minute` per key for AI endpoints, `30/minute` for health/status.
4. Rate-limit key function should use the authenticated API key (post-auth), not remote IP, to avoid proxy bypass.

**Medium-term (production hardening):** Migrate to **Pattern B** (IAP) as the infrastructure-layer gate, keeping the in-app `verify_key` as defence-in-depth.

---

## Packages to evaluate

| Package | Stars / Popularity | Maintained | License | Fits Stack | Notes |
|---|---|---|---|---|---|
| `slowapi` | ~1 k GitHub stars, PyPI ~700 k/month downloads | Yes — active as of 2025 | MIT | ✅ Yes | FastAPI-native rate limiter; wraps `limits`. Use `slowapi==0.1.9`. |
| `limits` | Dependency of slowapi; ~3 M/month downloads | Yes | MIT | ✅ Yes (indirect) | Backend abstraction for Redis, Memcached, memory. Pin `limits>=3.6,<4`. |
| `fastapi` security utilities (`fastapi.security`) | Bundled with FastAPI — ~90 M/month downloads | Yes | MIT | ✅ Yes | `APIKeyHeader`, `OAuth2PasswordBearer` included; no extra install. |
| `python-jose[cryptography]` | ~11 M/month downloads | Maintenance-mode (last release 2022) — prefer `joserfc` or `PyJWT` | MIT | ⚠️ Conditional | Use only if JWT pattern is chosen. Prefer `PyJWT==2.9.0` with `cryptography>=42`. |
| `google-auth` | ~70 M/month downloads | Yes (Google-maintained) | Apache 2.0 | ✅ Yes | Required for IAP JWT validation; already a transitive dependency of `vertexai`. |
| `secure` | ~300 k/month downloads | Yes | MIT | ✅ Yes | One-liner HTTP security headers middleware. `secure==0.3.0`. Addresses the MEDIUM security-headers finding. |
| `detect-secrets` | ~3 M/month downloads | Yes (Yelp-maintained) | Apache 2.0 | ✅ Yes | Pre-commit hook to prevent future credential commits. `detect-secrets==1.5.0`. |
| `redis` (Python client) | ~25 M/month downloads | Yes | MIT | ✅ Yes | Required for `slowapi` Redis backend. `redis[hiredis]==5.0.8`. |

---

## Recommended packages

```
slowapi==0.1.9          # Rate limiting — FastAPI native
limits>=3.6,<4          # slowapi backend abstraction
redis[hiredis]==5.0.8   # slowapi Redis storage + Cloud Memorystore
secure==0.3.0           # Security headers middleware
google-auth>=2.29,<3    # IAP JWT validation (medium-term); already transitive dep
detect-secrets==1.5.0   # Pre-commit credential scanning
```

`fastapi.security.APIKeyHeader` requires no additional install.

---

## Security considerations

### Credential rotation
The hardcoded `***REMOVED-GOOGLE-KEY***` key must be **revoked immediately** (Google Cloud Console → APIs & Services → Credentials → Delete) before any code change is deployed. Rotating the key after deploying auth does not fix the exposure; the key is already in git history and any artefact derived from it.

### Timing-safe key comparison
Never compare API keys with `==`. Use `hmac.compare_digest(expected_hash, actual_hash)` or `secrets.compare_digest` to prevent timing oracle attacks that could enumerate valid key prefixes.

### Key storage
Store keys hashed (SHA-256 or bcrypt) in environment variables. The raw key is delivered out-of-band to callers. This means a stolen `.env` file does not immediately yield a usable credential.

### Rate-limit key function
Using `get_remote_address` behind GCP's load balancer will see the balancer's IP, making per-client limiting ineffective. Either:
- Configure `uvicorn` with `--proxy-headers` and trust the `X-Forwarded-For` header from known GCP CIDR ranges only, or
- Key rate limits on the authenticated `X-API-Key` value (preferred).

### File upload surface
Rate limiting alone does not bound cost from large uploads. The `10/minute` limit must be paired with:
- `MAX_UPLOAD_BYTES = 10_000_000` enforced before Pillow opens the file
- `Image.MAX_IMAGE_PIXELS = 50_000_000` (decompression bomb guard)
- MIME-type allowlist check before any PIL call

### Error response hygiene
`HTTPException` responses must never include raw GCP SDK exception text. Catch `google.api_core.exceptions.GoogleAPIError` specifically and re-raise a sanitised `HTTPException(500, "AI service error")`. Raw exceptions currently leak project IDs and service account metadata.

---

## Known pitfalls

1. **Applying auth to routes instead of the router.** Adding `Depends(verify_key)` to each individual route is fragile — new routes will be left unprotected. Apply the dependency at `APIRouter(dependencies=[Depends(verify_key)])` or at `app` level via `app = FastAPI(dependencies=[Depends(verify_key)])`.

2. **slowapi in-memory backend on multi-replica deployments.** The default `MemoryStorage` backend counts requests per process. On Cloud Run with concurrency > 1 or multiple replicas, each replica has its own counter, allowing callers to exceed limits by factor N. Always configure Redis/Memorystore for production.

3. **`X-Forwarded-For` spoofing.** If rate limiting keys on remote IP and the service is behind a proxy, clients can inject arbitrary IPs in the `X-Forwarded-For` header unless the proxy strips it. Key on the authenticated credential instead.

4. **Revealing 401 vs 403 semantics.** Returning `403 Forbidden` for an unrecognised API key leaks that authentication is API-key-based. Return `401 Unauthorized` with `WWW-Authenticate: ApiKey` and a generic message for all authentication failures.

5. **`slowapi` exception handler not registered.** Without `app.state.limiter = limiter` and `app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)`, rate limit breaches will raise unhandled exceptions rather than returning `429`. The handler setup is not automatic.

6. **Applying rate limits before auth.** Rate limiting unauthenticated requests by IP is trivially bypassed by rotating IPs. Auth must run first; the rate-limit key function must operate on the validated identity.

7. **Not revoking the leaked key first.** Deploying auth while the hardcoded key remains valid in git history provides no protection — the exposed key bypasses the new auth system entirely.

8. **Pillow decompression bombs before size validation.** `Image.open()` triggers decompression at call time. Read and size-check the raw bytes buffer *before* passing to `Image.open()`. Calling `file.read()` twice loses the stream position; buffer into a `BytesIO` once.

9. **`detect-secrets` baseline not committed.** Running `detect-secrets scan > .secrets.baseline` and committing the baseline file is required for the pre-commit hook to function correctly in CI. Without a baseline, the hook blocks all commits until the hardcoded key is removed from history via `git filter-repo`.

10. **GCP budget alerts ≠ hard cap.** GCP Budget Alerts send email notifications but do not stop billing by default. Set the alert action to trigger a Cloud Function that disables the billing account or kills the service, or configure `quotas` at the project level to enforce hard API call limits.

---

## Open questions

- [ ] **Key distribution mechanism:** How will `X-API-Key` values be issued and rotated to callers? Is there an internal secret management system (GCP Secret Manager, HashiCorp Vault) or will `.env` files be manually managed?
- [ ] **Multi-tenancy requirement:** Should different callers have different rate-limit tiers (e.g., internal services vs. external integrations)? This determines whether a single shared key or per-client keys with per-key limits are needed.
- [ ] **Redis / Memorystore availability:** Is Cloud Memorystore already provisioned, or will in-memory limiting be acceptable for an initial single-replica deployment?
- [ ] **IAP migration timeline:** Is GCP Identity-Aware Proxy on the infrastructure roadmap? This affects whether Pattern A is a permanent solution or a bridge implementation.
- [ ] **Git history remediation:** Has `git filter-repo` / BFG been run to purge the hardcoded key from all historical commits and force-push? Branch protection rules may block this without admin approval.
- [ ] **Caller identity model:** Are all callers internal services (service-to-service), human users via a frontend, or both? The answer determines whether API key auth is sufficient long-term or whether OAuth2/OIDC is needed.
- [ ] **Acceptable rate limits:** What is the expected legitimate request volume (requests/minute per caller)? This is required to set limits that block abuse without blocking production traffic.

---

## AI/LLM-specific research areas

### 1. Semantic caching strategy

**Which requests qualify for caching:**
AI extraction and marking requests using the same question paper image + subject are high-value cache candidates. Since inputs include file uploads (images), caching requires hashing the file content (SHA-256 of image bytes) plus the subject/prompt key as the cache key — not semantic embedding similarity. True semantic caching (cosine similarity) is only applicable to text-only prompt inputs; for image+text multimodal inputs used here, deterministic content hashing is more practical.

If text-only prompt variants exist (e.g., re-marking requests with the same extracted text), semantic caching via embedding comparison is viable:
- Embed the prompt text using `text-embedding-004` (Vertex AI) or `text-embedding-3-small` (OpenAI)
- Store embeddings in Redis with `redis-py`'s vector search (`RediSearch` module) or a lightweight in-process FAISS index
- Cosine similarity threshold: `≥ 0.97` for extraction prompts (highly deterministic expected output); `≥ 0.95` for marking prompts

**Cache store recommendation:** Redis (Cloud Memorystore) with a TTL of 24 hours for extraction results; 1 hour for marking results (marking prompts may vary more). Avoid in-memory caching — it does not survive pod restarts and is not shared across replicas.

**Cache invalidation:** Invalidate by subject key prefix when a prompt template is updated. Store a `prompt_version` hash alongside the cached result and invalidate on mismatch.

---

### 2. Token budget and cost control

**Per-request caps:**
- Set `max_output_tokens` on every `GenerativeModel.generate_content()` call. Recommended starting limits: `1024` tokens for extraction, `512` for marking/scoring responses. These prevent runaway generations from adversarial inputs designed to exhaust output quotas.
- Input token estimation: use `model.count_tokens()` before dispatch. Reject requests exceeding `8192` input tokens with `HTTP 413` and a user-readable message.

**Per-user/key daily budget:**
- Track cumulative input + output tokens per API key in Redis with a daily TTL counter (`INCR ai_tokens:{api_key}:{date}`).
- When the counter exceeds the configured daily budget (e.g., 500 000 tokens/key/day), return `HTTP 429` with `Retry-After: {seconds_until_midnight}`.
- The `token_logger.py` CSV logging already captures per-request token counts — replace or augment with Redis writes to enable real-time enforcement rather than post-hoc analysis.

**Cost circuit breaker:**
- Cloud Billing budget alerts should trigger a Pub/Sub message to a Cloud Function that sets a `CIRCUIT_OPEN` flag in Redis when daily spend crosses a threshold (e.g., 80% of budget).
- The FastAPI app checks this flag at request intake and returns `HTTP 503 Service Unavailable` with `Retry-After` until the flag clears at day rollover.

**Streaming vs. batch:**
- Streaming (`generate_content(..., stream=True)`) does not reduce token cost — billing is identical. Streaming is preferable for user-facing latency perception but complicates token counting (final usage metadata arrives in the last chunk). For batch/automated marking, non-streaming is simpler and allows full response validation before returning to caller.

---

### 3. Context window management

**Current service pattern:** Based on the architecture (extraction + marking pipeline), requests appear to be stateless single-turn interactions (submit image → receive extracted text / score). There is no observed multi-turn conversation history.

**If single-turn (likely):**
- No sliding window or summarization strategy is needed.
- Context window management reduces to: cap system prompt size + image bytes + question text ≤ 85% of model's context window to leave headroom for output.
- `gemini-1.5-pro` supports 1 M token context; `gemini-1.5-flash` supports 1 M tokens. For `gemini-pro` (standard), the limit is 32 768 tokens. Always retrieve the limit programmatically: `model.count_tokens()` before dispatch.

**If multi-turn is introduced later:**
- Sliding window: keep last N turns where cumulative tokens ≤ 60% of context limit
- Summarization trigger: when history exceeds threshold, summarize with `gemini-1.5-flash` (cheaper) rather than the primary model
- Store conversation state in Redis keyed by session ID with 30-minute TTL (not in-process memory)

---

### 4. Prompt injection and adversarial inputs

**System prompt isolation:**
- The system prompt (marking rubric, extraction instructions) must be passed in the `system_instruction` parameter of `GenerativeModel`, not prepended to user content. `vertexai.generative_models.GenerativeModel(model_name, system_instruction=RUBRIC)` keeps system and user content in separate message slots, making content-based injection harder.
- Never do: `prompt = SYSTEM_PROMPT + "\n\nUser input: " + user_text`

**Input sanitization for LLM payloads:**
- Strip Unicode control characters (`\x00`–`\x1F` except `\n\t`) from all text fields before constructing prompts
- Limit the `subject` form field to the `SUBJECT_PROMPTS` allowlist (already flagged in the security spec)
- For image inputs: do not extract text from images on the server before sending to the model (avoid a secondary injection surface from OCR output being concatenated into prompts)

**Jailbreak and prompt-leak detection:**
- Post-process model responses: if the output contains verbatim system prompt fragments (check for sentinel strings inserted into the system prompt, e.g., `##SYSTEM_END##`), reject the response and log a security event
- Block outputs matching known jailbreak response patterns (e.g., responses beginning with "Sure, I'll ignore my previous instructions")
- For a lightweight defence, use a secondary `gemini-flash` call as a response classifier on a random sample (5%) of outputs to detect policy violations without blocking latency on every request

---

### 5. PII handling before external LLM calls

**PII risk in this service:**
- Student answer sheets (uploaded images) are the primary PII source. They may contain student names, ID numbers, dates of birth, and handwritten personal details.
- The extraction step sends raw image bytes to Vertex AI / Gemini. Vertex AI Enterprise tier includes a Data Processing Addendum (DPA) stating that input data is not used to train models. Confirm the project's tier and ensure the DPA is in place.

**Strategy — masking over scrubbing for images:**
- For image inputs, PII scrubbing is not practical pre-upload (would require OCR + NER on the image before sending to the model — computationally expensive and circular). Instead:
  - Rely on contractual DPA with Google (Vertex AI Enterprise)
  - Add a system instruction: "Do not repeat any personal identifying information visible in the image in your response."
  - Post-process extraction output: run a lightweight NER pass (`spacy` with `en_core_web_sm` or Google Cloud DLP API) to detect and redact PII from the extracted text before it is stored or logged

**For text inputs:**
- Apply Cloud DLP `projects.content.deidentify` before constructing prompts if inputs contain free-text student responses with possible PII
- Use tokenization (replace PII with `[STUDENT_ID_1]`) rather than deletion to preserve semantic structure for marking

**Data retention:**
- Do not persist raw model inputs or outputs beyond the request lifecycle (no logging of raw content — only token counts, as the security spec requires)
- The `token_logger.py` CSV must not log prompt text or model output, only numeric metadata

---

### 6. Model reliability and fallback

**Recommended fallback chain for this GCP-native service:**

```
Primary:   gemini-1.5-pro   (Vertex AI, asia-south1)
Fallback:  gemini-1.5-flash (Vertex AI, asia-south1) — cheaper, faster
Emergency: gemini-1.0-pro   (Vertex AI) — stable, well-tested
Final:     Return cached result if available; else HTTP 503
```

**Retry policy:**
- `google.api_core.retry.AsyncRetry` with exponential backoff is built into the Vertex AI SDK. Configure: initial delay 1 s, multiplier 2, max delay 30 s, deadline 120 s for `RESOURCE_EXHAUSTED` (429) errors only.
- Do not retry on `INVALID_ARGUMENT` (400) — these are deterministic failures.
- Immediate fallback (no retry) on `UNAVAILABLE` (503) and `INTERNAL` (500) — retry at the application layer with the next model in the chain.

**Circuit breaker:**
- Track consecutive failures per model in Redis (`ai_failures:{model}` counter with 5-minute TTL).
- Open the circuit after 5 consecutive failures: skip the model and go directly to fallback.
- Half-open after TTL expiry: allow one probe request before fully closing the circuit.
- Use `tenacity` (`tenacity==8.3.0`) for retry + circuit-breaker logic at the Python layer, supplementing the SDK's built-in retry.

---

### 7. Async / streaming architecture

**Current context:** The service uses synchronous Vertex AI SDK calls. File uploads (images) are accepted via FastAPI `UploadFile`, which is already async-compatible. Marking/extraction tasks are likely CPU+network-bound with latency in the 5–30 second range.

**Recommended pattern — async request handling with timeout:**
- Convert all `GenerativeModel.generate_content()` calls to use `asyncio.to_thread()` (wraps sync SDK calls without blocking the event loop) until Vertex AI provides a native async client.
- Set a per-request timeout: `asyncio.wait_for(asyncio.to_thread(model.generate_content, ...), timeout=60.0)` — return `HTTP 504` on timeout.
- Do not use `stream=True` for the marking pipeline unless the frontend renders incremental output — streaming complicates token counting and error handling.

**For long-running batch jobs (future):**
- If multi-page exam papers are introduced, use Cloud Tasks or Pub/Sub to enqueue jobs asynchronously.
- Return `HTTP 202 Accepted` with a job ID; expose a `GET /jobs/{id}` status endpoint.
- Store job state in Firestore or Cloud Memorystore.

**Graceful cancellation:**
- Wrap `generate_content` in a task; register a cancellation handler that calls `model.cancel()` if the client disconnects (FastAPI `Request.is_disconnected()` check in a background polling loop).

---

### 8. Audit logging for compliance

**Required log fields (per request):**

```python
{
    "timestamp": "2026-06-12T10:23:41Z",       # ISO 8601 UTC
    "request_id": "uuid4",                      # Correlate with upstream
    "api_key_hash": "sha256:abc123...",          # Hash of X-API-Key — NOT raw key
    "subject": "mathematics",                   # From allowlisted form field
    "model": "gemini-1.5-pro",                  # Actual model used (may differ from primary)
    "input_tokens": 1842,
    "output_tokens": 312,
    "latency_ms": 4201,
    "cost_usd": 0.002341,                       # Calculated from token counts + pricing table
    "cache_hit": false,
    "status": "success",                        # or "error", "rate_limited", "circuit_open"
    "error_code": null                          # GCP error code if applicable — NOT error message
}
```

**Fields that must NOT be logged:**
- Raw prompt text or constructed prompt content (may contain PII from student papers)
- Raw model output / extracted text
- Full file name or file content
- Raw exception messages (contain GCP project metadata)
- The raw `X-API-Key` value

**Log destination:**
- Replace the current `token_logger.py` CSV approach with `google-cloud-logging` (`google-cloud-logging==3.10.1`) structured JSON logs sent to Cloud Logging. This moves logs off local filesystem, applies IAM access controls, and enables Cloud Log Analytics queries.
- Set log retention: 90 days in Cloud Logging (configurable via Log Buckets), export to Cloud Storage / BigQuery for long-term compliance retention.

**Access controls on audit logs:**
- Create a dedicated log bucket `ai-audit-logs` with `_Default` exclusion so audit logs are not mixed with application logs.
- Grant `roles/logging.viewer` only to the compliance/security team service account. Restrict all other accounts.