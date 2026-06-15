# Test plan: Rate Limiting on AI-Invocation Endpoints

---

## Regression tests (prove the bug is fixed)

| ID | Test | Input | Expected output | Pass/Fail |
|----|------|-------|-----------------|-----------|
| R01 | Verify `slowapi` limiter is mounted as Starlette middleware on app startup | Import and inspect `app.middleware_stack` after `main.py` initialisation | `SlowAPIMiddleware` present in the middleware stack; no `AttributeError` | - |
| R02 | Single endpoint does not exceed allowed request budget within window | Send `RATE_LIMIT_REQUESTS` (e.g. 10) POST requests to `/extract` within `RATE_LIMIT_WINDOW` (e.g. 60 s) from the same IP | All 10 requests return `200 OK`; no `429` raised before limit is reached | - |
| R03 | 11th request within same window is rejected | Send 11 POST requests to `/extract` in < 60 s from same IP | Requests 1–10 return `200`; request 11 returns `429 Too Many Requests` with body `{"detail": "Rate limit exceeded"}` | - |
| R04 | Counter resets after window expires | Send 10 requests, wait for window to expire (60 s), then send 1 more | The post-window request returns `200 OK` | - |
| R05 | Rate limit applies independently per-endpoint | Exhaust limit on `/extract` (10 requests) then send 1 request to `/mark` | `/extract` returns `429`; `/mark` returns `200` (separate bucket) | - |
| R06 | `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW` are read from environment, not hardcoded | Start service with `.env` values `RATE_LIMIT_REQUESTS=2`, `RATE_LIMIT_WINDOW=30`; send 3 requests | Request 3 returns `429` after only 2 successful requests; confirms env override is respected | - |
| R07 | `Retry-After` header present in 429 response | Trigger rate limit on `/mark` | `429` response contains `Retry-After` header with a positive integer value (seconds until window reset) | - |

---

## Unit tests

| ID | Function/method | Scenario | Expected | Pass/Fail |
|----|----------------|----------|----------|-----------|
| U01 | `get_limiter()` factory | Happy path — `RATE_LIMIT_REQUESTS=10`, `RATE_LIMIT_WINDOW=60` set in env | Returns a `slowapi.Limiter` instance with key function set to `get_remote_address`; limit string equals `"10/minute"` | - |
| U02 | `get_limiter()` factory | Missing `RATE_LIMIT_REQUESTS` env var (unset) | Falls back to default (e.g. `5`) without raising; returns valid `Limiter` with `"5/minute"` | - |
| U03 | `get_limiter()` factory | `RATE_LIMIT_REQUESTS` set to `"0"` | Raises `ValueError("RATE_LIMIT_REQUESTS must be a positive integer")` | - |
| U04 | `get_limiter()` factory | `RATE_LIMIT_REQUESTS` set to `"-1"` | Raises `ValueError("RATE_LIMIT_REQUESTS must be a positive integer")` | - |
| U05 | `get_limiter()` factory | `RATE_LIMIT_WINDOW` set to `"abc"` (non-numeric) | Raises `ValueError("RATE_LIMIT_WINDOW must be a positive integer")` | - |
| U06 | `_rate_limit_exceeded_handler(request, exc)` | `RateLimitExceeded` exception passed in | Returns `JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})` and includes `Retry-After` header | - |
| U07 | `_rate_limit_exceeded_handler(request, exc)` | Handler does **not** log raw exception repr (prevents info leakage) | Log output contains sanitised message only; `repr(exc)` / GCP project details absent from log line | - |
| U08 | Limiter decorator on `/extract` route | Call `app.routes` and inspect decorator metadata | Route `/extract` has a `@limiter.limit(...)` decorator matching the configured limit string | - |
| U09 | Limiter decorator on `/mark` route | Call `app.routes` and inspect decorator metadata | Route `/mark` has a `@limiter.limit(...)` decorator matching the configured limit string | - |
| U10 | `get_remote_address` key function | IPv4 request with `X-Forwarded-For: 203.0.113.5` header | Returns `"203.0.113.5"` (trusted proxy header is respected) | - |
| U11 | `get_remote_address` key function | Request with no `X-Forwarded-For` header, direct IP `127.0.0.1` | Returns `"127.0.0.1"` | - |

---

## API tests

```bash
# ── Prerequisites ──────────────────────────────────────────────────────────────
# BASE_URL: replace with actual service URL, e.g. http://localhost:8000
# TOKEN: bearer token (once core/auth.py is deployed; use empty string before Day 3)
export BASE_URL="http://localhost:8000"
export TOKEN="<bearer_token_placeholder>"

# ── R01 — Confirm limiter middleware is registered (introspection endpoint) ───
curl -X GET "$BASE_URL/healthz" \
  -H "Authorization: Bearer $TOKEN"
# Expected: 200 OK
# {"status": "ok", "middlewares": ["SlowAPIMiddleware", ...]}

# ── R02 — 10 requests within window all succeed ──────────────────────────────
for i in $(seq 1 10); do
  curl -s -o /dev/null -w "Request $i: %{http_code}\n" \
    -X POST "$BASE_URL/extract" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"document_id": "doc-abc-001", "prompt": "Extract key facts."}'
done
# Expected: All 10 lines print "200"

# ── R03 — 11th request triggers 429 ─────────────────────────────────────────
for i in $(seq 1 11); do
  curl -s -o /dev/null -w "Request $i: %{http_code}\n" \
    -X POST "$BASE_URL/extract" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"document_id": "doc-abc-001", "prompt": "Extract key facts."}'
done
# Expected: Requests 1–10 → 200; Request 11 → 429

# ── R04 — Confirm 429 body shape and Retry-After header ─────────────────────
curl -i \
  -X POST "$BASE_URL/extract" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc-abc-001", "prompt": "Extract key facts."}'
# Expected (after limit is exhausted):
# HTTP/1.1 429 Too Many Requests
# Retry-After: <positive_integer>
# Content-Type: application/json
# {"detail": "Rate limit exceeded"}

# ── R05 — /mark endpoint independently rate-limited ─────────────────────────
# First exhaust /extract
for i in $(seq 1 10); do
  curl -s -o /dev/null \
    -X POST "$BASE_URL/extract" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"document_id": "doc-abc-001", "prompt": "Extract key facts."}'
done
# Then call /mark — should still be 200
curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST "$BASE_URL/mark" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"submission_id": "sub-xyz-999", "rubric_id": "rubric-001"}'
# Expected: 200

# ── S01 — Unauthenticated request (post Day 3 auth deployment) ───────────────
curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST "$BASE_URL/extract" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc-abc-001", "prompt": "Extract key facts."}'
# Expected: 401 Unauthorized

# ── S02 — Malformed payload rejected before AI call ─────────────────────────
curl -i \
  -X POST "$BASE_URL/extract" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": 99999, "prompt": null}'
# Expected: 422 Unprocessable Entity
# {"detail": [{"loc": ["body", "prompt"], "msg": "none is not an allowed value", ...}]}

# ── S03 — Oversized prompt rejected (input validation guard) ─────────────────
curl -i \
  -X POST "$BASE_URL/extract" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"document_id\": \"doc-abc-001\", \"prompt\": \"$(python3 -c 'print("A"*100001)')\"}"
# Expected: 413 Payload Too Large  OR  422 with detail "prompt exceeds maximum length"

# ── S04 — Rate limit not bypassable via X-Forwarded-For spoofing ─────────────
for i in $(seq 1 11); do
  curl -s -o /dev/null -w "Request $i: %{http_code}\n" \
    -X POST "$BASE_URL/extract" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -H "X-Forwarded-For: 10.0.0.$i" \
    -d '{"document_id": "doc-abc-001", "prompt": "Extract key facts."}'
done
# Expected: Request 11 → 429 (service uses real remote addr, not spoofed header from untrusted proxies)

# ── S05 — Leaked API key no longer present in source (CRITICAL-001 companion) ─
grep -r "***REMOVED-GOOGLE-KEY***" ./core/
# Expected: No output (grep exits non-zero — key has been removed and rotated)
```

---

## Security tests

| ID | Test | Method | Expected | Pass/Fail |
|----|------|--------|----------|-----------|
| S01 | Unauthenticated request rejected after Day 3 auth deployment | `POST /extract` with no `Authorization` header | `401 Unauthorized`; body `{"detail": "Not authenticated"}`; no AI call initiated | - |
| S02 | Malformed JSON payload returns structured error, not raw exception | `POST /extract` with `Content-Type: application/json`, body `{not_valid_json` | `422 Unprocessable Entity` with `{"detail": [...]}` pydantic error; no GCP project ID or stack trace in response body | - |
| S03 | Cost-exhaustion DoS — rapid burst beyond limit is blocked | Send 200 requests to `/extract` as fast as possible from single IP in one window | All requests beyond the configured threshold return `429`; total AI API calls ≤ `RATE_LIMIT_REQUESTS`; verified via GCP console billing spike absent | - |
| S04 | `X-Forwarded-For` spoofing does not bypass per-IP limit | Cycle `X-Forwarded-For` header through 200 distinct fake IPs across 200 requests (untrusted proxy scenario) | Rate limit triggers based on real socket IP; spoofed header ignored; requests 11+ return `429` | - |
| S05 | Hardcoded API key absent from codebase (`CRITICAL-001` regression) | `grep -r "AIzaSyCGqeqalFqiok"` across entire repo | Zero matches; `list_models.py:4` reads from `os.environ["GOOGLE_API_KEY"]` | - |
| S06 | Error responses do not leak internal GCP configuration | Trigger a deliberate upstream Vertex AI error by providing an invalid model name via env override | Response is `502 Bad Gateway` or `500` with generic message; no `project_id`, region, service-account email, or Python traceback in response body | - |
| S07 | Rate limit headers present and accurate | Send any request that is within the allowed budget | Response includes `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers with correct integer values | - |
| S08 | Environment variable `RATE_LIMIT_REQUESTS=0` blocked at startup | Start service with `RATE_LIMIT_REQUESTS=0` in `.env` | Service refuses to start; raises `ValueError` logged to stderr; no HTTP port bound | - |

---

## Test results summary

| Metric | Count |
|--------|-------|
| Total tests | 26 |
| Regression tests | 7 (R01–R07) |
| Unit tests | 11 (U01–U11) |
| API tests | 8 (curl blocks R01–S05) |
| Security tests | 8 (S01–S08) |
| **Passed** | - |
| **Failed** | - |
| **Blocked** | - |

> **Note — Day 3 dependency:** Tests S01 (auth on endpoints) are blocked until `core/auth.py` bearer-token middleware is deployed (72-hour window per sequencing plan). All other tests are executable immediately after the `slowapi` middleware patch is deployed and the CRITICAL-001 key is rotated.