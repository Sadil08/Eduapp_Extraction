# Test plan: Add Authentication and Rate Limiting to All Endpoints

## Regression tests (prove the bug is fixed)

| ID | Test | Input | Expected output | Pass/Fail |
|----|------|-------|-----------------|-----------|
| R01 | Unauthenticated request is now rejected | `GET /` with no headers | HTTP 401, body `{"detail": "Unauthorized"}` | - |
| R02 | Hardcoded API key removed from source | `grep -r "***REMOVED-GOOGLE-KEY***" .` run against entire repo | Zero matches in any file | - |
| R03 | Valid `X-API-Key` header now grants access | `POST /extract` with header `X-API-Key: correct-key`, valid payload | HTTP 200 with extraction result | - |
| R04 | Rate limit blocks 21st request within 60 s from same IP | 21 sequential `POST /extract` calls within 60 s from `127.0.0.1`, each with valid `X-API-Key` | First 20: HTTP 200; 21st: HTTP 429 with `Retry-After` header present | - |
| R05 | Raw exception output no longer leaks GCP project ID | Trigger a model invocation error by passing a malformed model name | Response body contains only `{"detail": "Internal server error"}`; GCP project ID absent from response and structured logs | - |
| R06 | File upload endpoint protected by auth | `POST /upload` with no `X-API-Key` header and a valid PDF file | HTTP 401, body `{"detail": "Unauthorized"}`; file never touches Pillow | - |
| R07 | Oversized file rejected before processing | `POST /upload` with valid `X-API-Key`, multipart file of 52 MB (limit assumed 50 MB) | HTTP 413, body `{"detail": "File too large"}` | - |
| R08 | Disallowed MIME type rejected | `POST /upload` with valid `X-API-Key`, file with `Content-Type: application/x-executable` | HTTP 415, body `{"detail": "Unsupported media type"}` | - |

---

## Unit tests

| ID | Function/method | Scenario | Expected | Pass/Fail |
|----|----------------|----------|----------|-----------|
| U01 | `verify_api_key(key)` | Correct plaintext key matching stored SHA-256 hash | Returns `True`; no exception raised | - |
| U02 | `verify_api_key(key)` | Wrong key (one character off: `wrong-key`) | Returns `False` | - |
| U03 | `verify_api_key(key)` | Empty string `""` | Returns `False`; does not raise `ValueError` | - |
| U04 | `verify_api_key(key)` | Timing: compare correct vs. incorrect key 10 000 times each | Mean execution time difference < 5 µs (constant-time guaranteed by `secrets.compare_digest`) | - |
| U05 | `rate_limiter.is_allowed(ip)` | First 20 calls within 60 s from `"192.168.1.1"` | Returns `True` for all 20 | - |
| U06 | `rate_limiter.is_allowed(ip)` | 21st call within same 60 s window from `"192.168.1.1"` | Returns `False` | - |
| U07 | `rate_limiter.is_allowed(ip)` | First call in a new 60 s window after previous window elapsed | Returns `True` | - |
| U08 | `rate_limiter.is_allowed(ip)` | Calls from two distinct IPs (`"10.0.0.1"`, `"10.0.0.2"`) each making 20 calls within 60 s | Both return `True` for all 20; counters are isolated per IP | - |
| U09 | `validate_upload(file)` | File of exactly 50 MB with `Content-Type: image/jpeg` | Returns `True`; no exception | - |
| U10 | `validate_upload(file)` | File of 50 MB + 1 byte with `Content-Type: image/jpeg` | Raises `FileTooLargeError` | - |
| U11 | `validate_upload(file)` | File with `Content-Type: image/png` (allowlisted) | Returns `True` | - |
| U12 | `validate_upload(file)` | File with `Content-Type: application/x-sh` (not allowlisted) | Raises `UnsupportedMediaTypeError` | - |
| U13 | `sanitise_exception(exc)` | `Exception("project/my-gcp-project service-account@...")` passed in | Returns generic string `"Internal server error"`; original message not present in return value | - |
| U14 | `get_api_key_dependency()` | `SERVICE_API_KEY` env var not set at startup | Raises `EnvironmentError` with message `"SERVICE_API_KEY must be set"` before the server binds | - |

---

## API tests

```bash
# R01 — Request with no X-API-Key header is rejected with 401
curl -i -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Grade this essay.", "content": "The mitochondria is the powerhouse of the cell."}'
# Expected: HTTP/1.1 401 Unauthorized
#           {"detail": "Unauthorized"}

# R02 — Request with incorrect X-API-Key is rejected with 401
curl -i -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -H "X-API-Key: totally-wrong-key" \
  -d '{"prompt": "Grade this essay.", "content": "The mitochondria is the powerhouse of the cell."}'
# Expected: HTTP/1.1 401 Unauthorized
#           {"detail": "Unauthorized"}

# R03 — Request with correct X-API-Key is accepted
curl -i -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $VALID_SERVICE_API_KEY" \
  -d '{"prompt": "Grade this essay.", "content": "The mitochondria is the powerhouse of the cell."}'
# Expected: HTTP/1.1 200 OK
#           {"result": "<grading output>"}

# R04 — 21st request within 60 s triggers rate limit
for i in $(seq 1 21); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/extract \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $VALID_SERVICE_API_KEY" \
    -d '{"prompt": "Grade this essay.", "content": "Short answer."}';
done
# Expected: first 20 lines print 200; 21st line prints 429

# R05 — 429 response includes Retry-After header
curl -i -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $VALID_SERVICE_API_KEY" \
  -d '{"prompt": "test", "content": "test"}'
# (Run after exhausting rate limit window)
# Expected: HTTP/1.1 429 Too Many Requests
#           Retry-After: <integer seconds>
#           {"detail": "Rate limit exceeded"}

# R06 — File upload without auth is rejected
curl -i -X POST http://localhost:8000/upload \
  -F "file=@/tmp/sample.pdf"
# Expected: HTTP/1.1 401 Unauthorized
#           {"detail": "Unauthorized"}

# R07 — File upload with disallowed MIME type is rejected
curl -i -X POST http://localhost:8000/upload \
  -H "X-API-Key: $VALID_SERVICE_API_KEY" \
  -F "file=@/tmp/malicious.sh;type=application/x-sh"
# Expected: HTTP/1.1 415 Unsupported Media Type
#           {"detail": "Unsupported media type"}

# R08 — Valid PDF upload with correct auth is accepted
curl -i -X POST http://localhost:8000/upload \
  -H "X-API-Key: $VALID_SERVICE_API_KEY" \
  -F "file=@/tmp/sample.pdf;type=application/pdf"
# Expected: HTTP/1.1 200 OK
#           {"status": "uploaded", "filename": "sample.pdf"}

# R09 — Error response does not leak GCP project metadata
curl -i -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $VALID_SERVICE_API_KEY" \
  -d '{"prompt": "", "content": ""}'
# Expected: HTTP/1.1 4xx or 5xx
#           Body does NOT contain "projects/", "service-account", or "asia-south1"

# R10 — Health check endpoint also requires authentication
curl -i -X GET http://localhost:8000/health
# Expected: HTTP/1.1 401 Unauthorized
#           {"detail": "Unauthorized"}
```

---

## Security tests

| ID | Test | Method | Expected | Pass/Fail |
|----|------|--------|----------|-----------|
| S01 | Auth required — `/extract` endpoint | `POST /extract` with no `X-API-Key` header | HTTP 401; body exactly `{"detail": "Unauthorized"}`; no stack trace or project info in body | - |
| S02 | Auth required — `/upload` endpoint | `POST /upload` with no `X-API-Key` header, multipart PDF payload | HTTP 401; file not written to disk or passed to Pillow | - |
| S03 | Auth required — all future routes | `GET /nonexistent-route` with no `X-API-Key` | HTTP 401 (auth checked before routing); not HTTP 404 | - |
| S04 | Timing-safe key comparison | 10 000 requests with a one-character-wrong key vs. 10 000 with a completely random 64-char key; measure mean response time | Mean latency difference < 10 ms; confirms `secrets.compare_digest` prevents timing oracle | - |
| S05 | Hardcoded credential absent from codebase | Static scan: `grep -rn "AIzaSyCGqeqalFqiok"` across all `.py`, `.env.example`, Dockerfiles, CI configs | Zero matches | - |
| S06 | `SERVICE_API_KEY` not present in any committed file | `git log -p -- .env` and `trufflehog filesystem .` | No secrets detected; `.env` absent from git history | - |
| S07 | Rate-limit bypass via `X-Forwarded-For` spoofing | `POST /extract` 21 times with rotating `X-Forwarded-For` header values (`1.2.3.1`–`1.2.3.21`) from single real IP | Service uses real socket IP (or trusted proxy header only); all 21 requests counted against same client; 21st returns HTTP 429 | - |
| S08 | Decompression-bomb file rejected before Pillow decode | `POST /upload` with valid `X-API-Key`, a 42 KB zip-bomb disguised as `image/png` | HTTP 413 or HTTP 415 returned; Pillow `Image.open()` never called; server memory unaffected | - |
| S09 | SQL/template injection in prompt field | `POST /extract` with `X-API-Key: $VALID_SERVICE_API_KEY`, body `{"prompt": "'; DROP TABLE users;--", "content": "x"}` | HTTP 200 with sanitised AI output **or** HTTP 422 validation error; no 500; no raw exception detail in response | - |
| S10 | Error responses do not leak internal config | Force a Vertex AI SDK exception by providing invalid model name `"models/does-not-exist"` via valid authenticated request | Response body is exactly `{"detail": "Internal server error"}`; GCP project ID, region (`asia-south1`), service account email absent from body and from HTTP headers | - |

---

## Test results summary

| Category | Tests | Passed | Failed |
|----------|-------|--------|--------|
| Regression | 8 | - | - |
| Unit | 14 | - | - |
| API | 10 | - | - |
| Security | 10 | - | - |
| **Total** | **42** | **-** | **-** |

> **Blocking criteria:** All R-series and S-series tests must pass before this feature is merged. U-series failures in timing tests (U04) are blocking. API test R09 (no metadata leakage) is blocking. No deployment to any environment may proceed while the hardcoded credential (`***REMOVED-GOOGLE-KEY***`) is detectable in the repository history (S05, S06).