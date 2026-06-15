```markdown
---
feature: add-authentication-and-rate-limiting-to-all-endpoints
status: draft
date: 2026-06-12
author: senel.ephraims@gmail.com
risk-level: critical
---

# Feature spec: Add authentication and rate limiting to all endpoints

## User story

As a **platform operator**, I want every endpoint of the AI_Extraction_Marking_service to require a valid API key and to be subject to per-client rate limits, so that only authorised callers can invoke paid Vertex AI / Gemini model calls and the service cannot be abused for cost-exhaustion or data-exfiltration attacks.

---

## Problem statement

The AI_Extraction_Marking_service currently exposes file-upload and form-submission endpoints in `main.py` with **zero authentication, zero rate limiting, and no input validation**. A hardcoded live Google Generative AI API key in `list_models.py` further means any repository reader can impersonate the service. The combined result is:

- Any unauthenticated caller can trigger arbitrarily many paid Vertex AI / Gemini model invocations (Denial-of-Wallet / billing DoS).
- Uploaded files have no size cap, no MIME-type allowlist, and are passed directly to Pillow, exposing a decompression-bomb and CVE surface.
- Raw exception output leaks GCP project IDs and service-account metadata into logs accessible to anyone with log-read access.
- The hardcoded credential (`***REMOVED-GOOGLE-KEY***`) must be revoked and removed as a **blocking pre-condition** to deploying this feature.

This feature closes all four gaps in a single, coordinated change.

---

## Functional requirements

| ID  | Requirement | Priority |
|-----|-------------|----------|
| F01 | System shall reject any HTTP request that does not include a valid `X-API-Key` header with HTTP 401 and a generic `{"detail": "Unauthorized"}` body (no internal detail). | must |
| F02 | System shall validate the `X-API-Key` value using a constant-time comparison (`secrets.compare_digest`) against the SHA-256 hash of the value stored in `SERVICE_API_KEY` environment variable. | must |
| F03 | System shall apply the authentication dependency at the FastAPI application or router level so that any future routes are protected without per-route annotation. | must |
| F04 | System shall enforce a rate limit of no more than 20 requests per minute per unique client IP address on all AI-invocation endpoints; requests exceeding this limit shall receive HTTP 429 with a `Retry-After` header. | must |
| F05 | System shall enforce a rate limit of no more than 200 requests per minute per unique client IP across all non-AI utility endpoints. | must |
| F06 | System shall reject file uploads larger than 10 MB with HTTP 413. | must |
| F07 | System shall reject file uploads whose `Content-Type` is not in the allowlist `{image/jpeg, image/png, image/webp}` with HTTP 415. | must |
| F08 | System shall set `PIL.Image.MAX_IMAGE_PIXELS` to `89_478_485` (≈ 89 MP) before opening any uploaded image to prevent decompression-bomb attacks. | must |
| F09 | System shall validate that the `subject` form field value exists as a key in `SUBJECT_PROMPTS`; if not, it shall return HTTP 400 with the list of valid subject values. | must |
| F10 | System shall load all credentials and configuration (including `SERVICE_API_KEY`, `GOOGLE_GENAI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`) exclusively from environment variables via `python-dotenv`; no credential may appear as a string literal in source code. | must |
| F11 | System shall emit structured JSON log entries for every request using Python's `logging` module, including `timestamp`, `method`, `path`, `status_code`, `client_ip_hash`, `latency_ms`, and `request_id`; raw exception objects shall never appear in log output. | must |
| F12 | System shall add HTTP response headers `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Content-Security-Policy: default-src 'none'` to every response via Starlette middleware. | must |
| F13 | System shall configure `CORSMiddleware` with an explicit `allow_origins` allowlist sourced from the `CORS_ALLOWED_ORIGINS` environment variable; wildcard `*` origin shall be rejected at startup. | must |
| F14 | User can rotate the API key by updating `SERVICE_API_KEY` in the environment and restarting the service without any code change. | must |
| F15 | System shall pre-commit hook integration (via `detect-secrets` baseline) block any commit that introduces a string matching known GCP/Google API key patterns. | must |
| F16 | System shall return HTTP 429 with a `cost-budget-exceeded` reason header when the per-user daily token budget is exhausted. | must |
| F17 | System shall write a per-request audit log entry containing `timestamp`, `user_id_hash` (HMAC of the API key), `model`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd` to Google Cloud Logging; raw prompt or response content shall never appear in this entry. | must |
| F18 | System shall scrub PII (names, email addresses, phone numbers, national IDs, payment card numbers) from all user-supplied text before constructing any LLM prompt. | must |
| F19 | System shall detect prompt-injection patterns in user input before forwarding to the LLM; on detection, the request shall be rejected with HTTP 400 and the attempt logged with severity WARNING. | must |
| F20 | System shall fall back to a secondary model when the primary Vertex AI model returns 3 consecutive 5xx errors within a single request attempt, and fall back to a cached best-match response if the secondary also fails. | should |
| F21 | System shall enforce a per-request token budget (input + output) of 8 192 tokens; requests that would exceed this limit shall be rejected with HTTP 400 before the model call is made. | must |

---

## Non-functional requirements

| ID   | Category | Requirement | Metric |
|------|----------|-------------|--------|
| NF01 | Performance | Authentication and rate-limit middleware latency overhead | p95 overhead < 5 ms; p99 overhead < 15 ms (measured at middleware layer, excluding LLM call) |
| NF02 | Security | All endpoints comply with OWASP API Security Top 10 (2023); transport encrypted | OWASP API Top 10 (2023) fully addressed; TLS 1.2+ enforced on all ingress; 0 high/critical findings in quarterly pen-test |
| NF03 | Scalability | Service handles burst AI-endpoint traffic under rate-limit constraints | Sustains 20 req/min per IP across ≥ 50 concurrent clients without degradation; horizontally scalable via stateless replicas behind a load balancer |
| NF04 | Reliability | Service availability and error recovery | 99.9% uptime (monthly); RTO < 5 min; auth layer available independently of LLM backend availability |
| NF05 | Maintainability | Code quality of new auth/rate-limit modules | Cyclomatic complexity ≤ 10 per function; unit-test coverage ≥ 85% for all new modules; no function exceeds 50 lines |
| NF06 | Observability | All errors and rate-limit events are traceable | 100% of HTTP 4xx/5xx responses include a `X-Request-ID` trace header; p99 request latency tracked in Cloud Monitoring; all rate-limit hits emitted as structured log events |
| NF07 | AI Cost Control | Token budget enforced per-request and per-API-key/day; daily spend circuit breaker | Per-request cap: 8 192 tokens (input + output); per-key daily budget: $2.00/day; global circuit breaker fires and halts all LLM calls when daily total exceeds $50; p95 per-request cost < $0.02 |
| NF08 | AI Semantic Cache | Semantically similar queries served from cache without LLM call | Cache hit rate > 30% on repeat-topic workloads; cache lookup p95 < 50 ms |
| NF09 | AI Safety | Prompt injection blocked; PII scrubbed before any external LLM call | 0 prompt-injection payloads reach the model (verified by integration tests with OWASP LLM01 corpus); 0 PII fields present in LLM request payload (verified by unit tests asserting scrubbed payload against mock) |
| NF10 | AI Reliability | LLM API errors handled with fallback chain; circuit breaker active | p99 availability of extraction/marking response > 99.5%; fallback fires within 3 consecutive LLM 5xx errors; circuit breaker half-opens after 30 s |
| NF11 | AI Compliance | All LLM calls audit-logged (hash, tokens, cost); PII absent from log entries | 100% of LLM calls present in Cloud Logging audit stream; 0 PII fields in any log entry (verified by automated log scanner); 90-day retention then automated purge |
| NF12 | AI Context | Conversation/prompt history managed within model context window | History never exceeds 80% of the target model's `max_tokens`; summarisation triggers automatically at 70% utilisation; prompt construction fails-safe with HTTP 400 if summarisation itself would exceed the window |

---

## Data model changes

### New entities

**`ApiKeyRecord`** (in-memory / `.env`-backed; no DB required in MVP)

| Field | Type | Notes |
|-------|------|-------|
| `key_hash` | `str` (SHA-256 hex) | Stored in env var `SERVICE_API_KEY_HASH`; never the raw key |
| `daily_token_usage` | `int` | Tracked in Redis under key `budget:{user_id_hash}:{date}` |
| `daily_cost_usd` | `float` | Tracked in Redis under key `cost:{user_id_hash}:{date}` |

**`RateLimitState`** (managed by `slowapi` / Redis backend)

| Field | Type | Notes |
|-------|------|-------|
| `client_ip_hash` | `str` | Hashed client IP used as bucket key |
| `window_count` | `int` | Rolling-window request count |
| `window_start_ts` | `datetime` | Start of current window |

### Modified entities

**`main.py` application startup** — Add middleware stack registration order:
1. `SecurityHeadersMiddleware` (new)
2. `CORSMiddleware` (new, strict origin list)
3. `SlowAPI` rate-limit middleware (new)
4. `verify_api_key` dependency (new, applied at app level)

---

## API contract

### Modified: all existing endpoints

All existing routes gain the following behaviours with no path changes:

**Request headers (required)**

```
X-API-Key: <raw-api-key>          # Required; validated via constant-time hash compare
X-Request-ID: <uuid>              # Optional; generated server-side if absent; echoed in response
```

**Response headers (always present)**

```
X-Request-ID: <uuid>
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'none'
RateLimit-Limit: 20
RateLimit-Remaining: <n>
RateLimit-Reset: <unix-ts>
```

**Error responses (new, standardised)**

```json
// HTTP 401 — missing or invalid API key
{"detail": "Unauthorized"}

// HTTP 429 — rate limit exceeded
{"detail": "Rate limit exceeded"}
// Headers: Retry-After: <seconds>

// HTTP 413 — upload too large
{"detail": "File too large. Maximum size is 10 MB."}

// HTTP 415 — unsupported media type
{"detail": "Unsupported file type. Allowed: image/jpeg, image/png, image/webp"}

// HTTP 400 — invalid subject / token budget exceeded / prompt injection detected
{"detail": "<specific safe message>"}
```

### New: `GET /health`

Unauthenticated liveness probe (no API key required; excluded from rate limit).

```
GET /health HTTP/1.1

HTTP/1.1 200 OK
Content-Type: application/json

{"status": "ok", "version": "<semver>"}
```

---

## Security considerations

### Authentication

- **Mechanism:** `APIKeyHeader(name="X-API-Key")` FastAPI dependency injected at application level via `app.dependency_overrides` or router-level `dependencies=[Depends(verify_api_key)]`.
- **Key storage:** Only the SHA-256 hash of the key is stored (in `SERVICE_API_KEY_HASH` env var). The raw key is never persisted.
- **Comparison:** `secrets.compare_digest(submitted_hash, stored_hash)` prevents timing-oracle attacks.
- **Rotation:** Update env var and restart; zero code change required.

### Credential remediation (blocking pre-condition)

- The hardcoded key `***REMOVED-GOOGLE-KEY***` in `list_models.py` **must be revoked** in Google Cloud Console → APIs & Services → Credentials before this branch is merged. Git history must be scrubbed with `git filter-repo` or BFG.
- All scripts must source `GOOGLE_GENAI_API_KEY` from environment, matching the pattern already used in `check_models_script.py`.

### New attack surface introduced by this feature

| Surface | Risk | Mitigation |
|---------|------|-----------|
| `X-API-Key` header over HTTP | Key interception | Enforce TLS 1.2+ at ingress (Cloud Load Balancer or nginx); reject non-HTTPS in middleware if `ENFORCE_HTTPS=true` |
| Rate-limit state in Redis | Cache poisoning, state bypass | Redis accessible only on private VPC; no anonymous access; TLS on Redis connection |
| `GET /health` (unauthenticated) | Information disclosure | Returns only `{"status":"ok","version":"..."}` — no model names, project IDs, or dependency versions |
| Pre-commit hook bypass | Attacker disables hook locally | Enforce the same `detect-secrets` scan in CI pipeline as a required check |

### Input validation

| Input | Validation applied |
|-------|-------------------|
| `X-API-Key` header | Constant-time hash comparison; reject if absent or wrong length |
| Uploaded file (`UploadFile`) | `content_type` in `{image/jpeg, image/png, image/webp}`; read length ≤ 10 MB; `PIL.Image.MAX_IMAGE_PIXELS` set before `Image.open()` |
| `subject` form field | Must be a key in `SUBJECT_PROMPTS`; HTTP 400 with allowlist if not |
| All text/prompt fields | PII scrubbed via regex + NER; prompt-injection classifier run; user content never interpolated into system-prompt string |
| CORS `Origin` header | Matched against `CORS_ALLOWED_ORIGINS` env var list; wildcard rejected at startup |

### OWASP API Security Top 10 (2023) applicability

| Item | Status after this feature |
|------|--------------------------|
| API1 — Broken Object Level Authorization | N/A (no per-object resources) |
| API2 — Broken Authentication | **Addressed** — API key auth added |
| API3 — Broken Object Property Level Authorization | N/A |
| API4 — Unrestricted Resource Consumption | **Addressed** — rate limiting + file size cap + token budget |
| API5 — Broken Function Level Authorization | **Addressed** — all routes protected by default |
| API6 — Unrestricted Access to Sensitive Business Flows | **Addressed** — rate limit on AI endpoints |
| API7 — Server-Side Request Forgery | Low risk; no URL-fetching from user input |
| API8 — Security Misconfiguration | **Addressed** — security headers + strict CORS |
| API9 — Improper Inventory Management | **Partially addressed** — `/health` only non-auth route; documented |
| API10 — Unsafe Consumption of APIs | **Addressed** — fallback chain + circuit breaker for LLM APIs |

---

## Dependencies

| Package | Version | Justification |
|---------|---------|---------------|
| `slowapi` | `^0.1.9` | FastAPI-native rate limiting via `Limits` decorators; uses Redis or in-memory backend |
| `redis` | `^5.0` | Backend store for distributed rate-limit state and per-key daily budget counters |
| `detect-secrets` | `^1.5` | Pre-commit hook and CI scan to prevent credential commits |
| `secure` | `^0.3` | One-liner HTTP security-header injection (CSP, X-Frame-Options, etc.) |
| `presidio-analyzer` | `^2.2` | NER-based PII detection before LLM prompt construction |
| `presidio-anonymizer` | `^2.2` | PII replacement/scrubbing in detected text spans |
| `structlog` | `^24.0` | Structured JSON logging replacing raw `print()` / `logging.basicConfig` calls |

---

## Spec files to create/update

- [ ] `specs/modules/auth.md` — Define `verify_api_key` dependency contract, key-hash storage pattern, rotation runbook
- [ ] `specs/modules/rate_limiting.md` — Define per-endpoint limits, Redis key schema, `429` response contract, GCP quota alignment
- [ ] `specs/modules/file_upload_validation.md` — Define size/MIME allowlist, Pillow hardening settings
- [ ] `specs/modules/input_validation.md` — Define PII scrub pipeline, prompt-injection classifier threshold, `subject` allowlist validation
- [ ] `specs/modules/ai_cost_control.md` — Define token budget schema, Redis budget counter TTL, circuit-breaker thresholds
- [ ] `specs/modules/audit_logging.md` — Define Cloud Logging log schema, PII exclusion contract, 90-day retention policy
- [ ] `specs/modules/security_headers.md` — Define middleware order, CSP policy, CORS allowlist env-var format
- [ ] `specs/modules/model_fallback.md` — Define primary/fallback model chain, circuit-breaker state machine
- [ ] `specs/modules/semantic_cache.md` — Define embedding model, similarity threshold, Redis TTL, cache-key design
- [ ] `core/list_models.py` — Remove hardcoded API key; replace with `os.getenv('GOOGLE_GENAI_API_KEY')` pattern

---

## Out of scope

The following items are **explicitly excluded** from this feature:

1. **OAuth 2.0 / OpenID Connect / GCP Identity-Aware Proxy integration** — Acknowledged as the preferred long-term auth path but deferred to a follow-on feature. The API-key layer is designed to be replaced without endpoint contract changes.
2. **Per-user or per-role access control** — All valid API-key holders have identical privilege level in this version; RBAC is a future feature.
3. **User management UI or key-issuance API** — Keys are issued and rotated via environment-variable update only; no admin portal is in scope.
4. **Database persistence of API keys, usage logs, or rate-limit state** — Redis (ephemeral) and Cloud Logging are the only persistence layers in this feature. A relational audit store is out of scope.
5. **Billing integration or invoice generation** — Cost tracking is advisory (Cloud Monitoring alerts) and does not trigger billing actions in this feature.
6. **Frontend / UI changes** — This is a backend service; no UI is in scope.
7. **Changes to Vertex AI IAM policies or GCP service-account permissions** — Infrastructure IAM hardening is a separate infrastructure ticket.
8. **Multi-tenancy / tenant isolation** — All requests share a single GCP project; tenant separation is out of scope.
9. **mTLS between internal services** — Internal service-to-service auth is out of scope; only external API surface is addressed.
10. **Automated key rotation** — Key rotation is manual (env-var update + restart); automated rotation via Secret Manager is a future improvement.
11. **LLM output filtering / content moderation** — PII scrubbing applies to *inputs* only; output moderation is a separate feature.
12. **Load testing or performance benchmarking infrastructure** — NFR targets are defined here but the load-testing tooling setup is a separate DevOps task.

---

## Open questions

- [ ] **Redis availability:** Should the rate-limit and budget Redis instance be a dedicated Cloud Memorystore instance, or can it share an existing Redis cluster? Decision needed from infrastructure team before implementation starts.
- [ ] **API key distribution:** What is the approved mechanism for securely distributing `SERVICE_API_KEY` to authorised callers (e.g., Secret Manager, manual out-of-band, Vault)? Security team sign-off required.
- [ ] **GCP budget alert threshold:** The $50/day global circuit-breaker threshold is a placeholder. Finance/product owner must confirm the acceptable maximum daily GCP API spend.
- [ ] **CORS allowed origins:** What is the definitive list of trusted origins to populate `CORS_ALLOWED_ORIGINS`? Product owner or frontend team to provide before deployment.
- [ ] **Hardcoded key revocation ownership:** Who is the named owner responsible for revoking `***REMOVED-GOOGLE-KEY***` in Google Cloud Console? This is a blocking prerequisite and needs an immediate assignee.
- [ ] **Presidio model size:** `presidio-analyzer` default NER model is ~500 MB. Is this acceptable in the container image, or should a lighter pattern-only scrubber be used for the first iteration?

---

## AI/LLM-specific requirements

### Semantic cache

- **Similarity threshold for cache hit:** Cosine similarity > 0.92 (using `text-embedding-004` embeddings).
- **Cache store + TTL:** Redis (Cloud Memorystore); TTL = 3 600 s (1 hour); eviction policy `allkeys-lru`.
- **Cache key design:** The cache key is the base64-encoded 768-dimensional embedding vector quantised to 4 bits, prefixed by `semcache:{model_id}:{subject_hash}:`; the lookup performs an approximate nearest-neighbour search using Redis Stack's `VSIM` command. Raw prompt text is never stored as the key.

### Token budget and cost control

- **Per-request token cap (input + output):** 8 192 tokens total; requests whose estimated prompt token count alone exceeds 6 000 tokens are rejected before the model call.
- **Per-key daily budget:** $2.00 USD/day tracked in Redis under `cost:{user_id_hash}:{iso-date}` with TTL = 86 400 s.
- **Circuit breaker threshold:** Halt all LLM calls for the remainder of the calendar day when the global daily total (summed across all keys) reaches $50.00 USD; emit a CRITICAL Cloud Monitoring alert.
- **Enforcement:** HTTP 429 with body `{"detail": "Daily token budget exceeded"}` and header `cost-budget-exceeded: true` returned when per-key budget is exhausted.

### Context window management

- **History strategy:** Sliding window with automatic summarisation fallback.
- **Summarisation triggers at:** 70% of the target model's `max_tokens` (e.g., for a 32 768-token model, trigger at 22 937 tokens of accumulated context).
- **Summarisation model:** `gemini-1.5-flash` (low-cost, fast) for summarisation sub-calls; summarised output replaces the oldest turn segments.

### Prompt injection protection

- **User input handling:** User-supplied content is always passed as a separate `contents` part, never interpolated into the system instruction string. System prompt is static and loaded from a versioned template file; no user variable is substituted into it.
- **Detection approach:** Two-layer: (1) pattern matching against OWASP LLM01 injection signatures (regex blocklist); (2) lightweight classifier using `google/flan-t5-base` fine-tuned on prompt-injection detection (confidence threshold > 0.85 triggers rejection).
- **Response on detection:** HTTP 400 with body `{"detail": "Input rejected"}` (no injection detail disclosed); log entry at WARNING severity with sanitised pattern-match label (not the raw input).

### PII scrubbing

- **PII types in scope:** Full names, email addresses, phone numbers (international formats), national/government IDs, payment card numbers (PAN), street addresses.
- **Scrub strategy:** `presidio-analyzer` (NER + pattern recognisers) detects entity spans; `presidio-anonymizer` replaces detected spans with typed placeholders (e.g., `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>`). A secondary regex pass covers common GCP project ID and service-account email patterns not in Presidio's default recognisers.
- **Verification:** Unit tests assert that a fixture payload containing all in-scope PII types produces zero matching entities in the post-scrub text, using a mock LLM call to capture the actual prompt sent.

### Model fallback chain

- **Primary:** `gemini-1.5-pro` via Vertex AI (`vertexai.generative_models.GenerativeModel`).
- **Fallback 1:** `gemini-1.5-flash` via Vertex AI (same region, lower cost, higher availability quota).
- **Fallback 2:** Semantic cache best-match response (cosine > 0.85, lower threshold than cache-hit path); if no match exists, return HTTP 503 with `{"detail": "Service temporarily unavailable"}`.
- **Circuit breaker:** Opens after 3 consecutive 5xx responses from the current model within a 60 s window; remains open for 30 s (half-open state); closes after 1 successful probe response in half-open state.

### Async / streaming

- **Response delivery:** Standard JSON response (no streaming in this iteration); SSE streaming deferred to a follow-on feature.
- **Long-task timeout:** 60 s hard limit enforced via `asyncio.wait_for`; if exceeded, the LLM call is cancelled and HTTP 504 is returned.
- **Cancellation:** Clients may abort via TCP connection drop; FastAPI's `Request.is_disconnected()` is polled every 5 s during LLM await; on disconnect, the pending Vertex AI call is cancelled via the SDK's `cancel()` method if available, otherwise abandoned with a logged warning.

### Audit logging

- **Log fields:** `timestamp` (ISO 8601 UTC), `request_id` (UUID), `user_id_hash` (HMAC-SHA256 of API key using a separate `AUDIT_HMAC_KEY` env var), `model`, `input_tokens` (int), `output_tokens` (int), `latency_ms` (int), `cost_usd` (float, 6 decimal places), `cache_hit` (bool), `fallback_used` (bool), `rate_limited` (bool).
- **NOT logged:** Raw prompt text, raw response text, uploaded file content, any field that could contain PII, the raw API key, or client IP in plaintext (only hashed).
- **Retention:** 90 days in Cloud Logging; automated log sink exclusion filter purges entries older than 90 days; BigQuery export for aggregated cost analysis retained for 12 months (aggregated, no per-request content).
- **Access:** Cloud Logging viewer role granted to ops team only; developers have no IAM role granting prod log read access; quarterly access review enforced.
```

---
*status: needs-human-review | score: 0.83 | iterations: 1*