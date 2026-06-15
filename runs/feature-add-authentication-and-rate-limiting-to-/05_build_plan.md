# Build plan: Add authentication and rate limiting to all endpoints

## Phase 1 — Credential Revocation & Project Hardening

| Step | Task | Files affected | Test |
|------|------|---------------|------|
| 1.1 | **Revoke the live key and purge it from source.** Immediately revoke `***REMOVED-GOOGLE-KEY***` in the Google Cloud console. Remove the `genai.configure(api_key=…)` hard-coded call from `list_models.py` and replace it with `genai.configure(api_key=os.environ["GOOGLE_API_KEY"])`. | `core/list_models.py`, `.env` | Run `git grep 'AIzaSy'` across the entire repo — zero matches required. Confirm the revoked key returns `HTTP 403` when called directly against the Gemini API. |
| 1.2 | **Audit and seal all secret surfaces.** Grep every file for patterns matching API key strings. Ensure `.env` is listed in `.gitignore`. Create a `.env.example` with placeholder values and a comment for every variable required at runtime (`GOOGLE_API_KEY`, `GCP_PROJECT`, `GCP_REGION`, `SERVICE_API_KEY`). | `.gitignore`, `.env.example`, `core/check_models_script.py` | `git status` shows `.env` as untracked/ignored. Running `python -c "from dotenv import dotenv_values; print(dotenv_values('.env.example'))"` loads without error and prints only placeholder strings. |
| 1.3 | **Bootstrap the test infrastructure.** Add `pytest`, `pytest-asyncio`, `httpx`, and `pytest-mock` to `requirements-dev.txt`. Create `tests/__init__.py`, `tests/conftest.py` with a minimal FastAPI `TestClient` fixture, and `tests/test_placeholder.py` with a single always-passing smoke test. | `requirements-dev.txt`, `tests/conftest.py`, `tests/test_placeholder.py` | `pytest tests/` exits 0; test count ≥ 1. |

---

## Phase 2 — API-Key Authentication Layer

| Step | Task | Files affected | Test |
|------|------|---------------|------|
| 2.1 | **Add `SERVICE_API_KEY` startup validation.** In application startup (FastAPI `lifespan` or `on_event("startup")`), read `SERVICE_API_KEY` from the environment. If absent or empty, raise `RuntimeError` with the message `"SERVICE_API_KEY environment variable is not set"` to prevent the server from starting without a configured secret. Pre-hash and cache `hashlib.sha256(SERVICE_API_KEY.encode()).digest()` in a module-level variable so per-request hashing is constant-cost. | `main.py`, `core/config.py` (new) | Set `SERVICE_API_KEY=""` and assert `pytest` raises `RuntimeError` at startup fixture. Set a non-empty value and assert startup completes cleanly. |
| 2.2 | **Implement the `verify_api_key` FastAPI dependency.** Create `core/auth.py`. The dependency extracts the `X-API-Key` header via `fastapi.Security(APIKeyHeader(name="X-API-Key", auto_error=False))`. Hash the incoming value with `hashlib.sha256` and compare against the cached digest using `secrets.compare_digest`. On failure (missing or wrong key), raise `HTTPException(status_code=401, detail="Unauthorized")` — no internal detail in the body. | `core/auth.py` | Unit-test three cases: (a) correct key → dependency returns without raising; (b) wrong key → `401`; (c) missing header → `401`. Assert the response body is exactly `{"detail": "Unauthorized"}` in both failure cases. |
| 2.3 | **Wire the dependency globally at the FastAPI app level.** Pass `dependencies=[Depends(verify_api_key)]` to the `FastAPI()` constructor (or the top-level `APIRouter`) so all current and future routes inherit it without per-route annotation. Remove any existing per-route stubs. | `main.py` | Use the `TestClient`. Call every existing route (GET `/`, POST `/upload`, etc.) without a header → all return `401`. Call them with a valid `X-API-Key` header → all return their normal response codes. |

---

## Phase 3 — Per-Client Rate Limiting

| Step | Task | Files affected | Test |
|------|------|---------------|------|
| 3.1 | **Install and initialise `slowapi`.** Add `slowapi` and `limits` to `requirements.txt`. In `main.py`, create a `Limiter` instance keyed on `get_remote_address`. Register `_rate_limit_exceeded_handler` with the FastAPI app so exceeded limits return `HTTP 429`. Store the limiter on `app.state.limiter` per `slowapi` convention. | `requirements.txt`, `main.py` | Import the module in isolation (`python -c "import slowapi"`). Confirm the `Limiter` instance is accessible at `app.state.limiter` in the `TestClient` fixture. |
| 3.2 | **Apply the 20 req/min limit to all AI-invocation routes.** Decorate every route that triggers a Vertex AI or Gemini call with `@limiter.limit("20/minute")`. If multiple routes exist, create a shared `APIRouter` with the limit applied at the router level via a dependency to avoid repetition. | `main.py` (or `core/routes.py` if router is extracted) | In a unit test, mock the AI call and issue 21 sequential requests from the same IP with a valid API key. Assert the first 20 return `2xx` and the 21st returns `429`. |
| 3.3 | **Add the `Retry-After` header to 429 responses.** Override the default `slowapi` rate-limit handler to inject `Retry-After: 60` (seconds until the window resets) into the response headers alongside the `429` status. | `main.py`, `core/rate_limit.py` (new handler) | In the 21st-request test from step 3.2, assert `response.headers["Retry-After"]` is present and its integer value is `> 0` and `≤ 60`. |
| 3.4 | **Phase 3 integration test.** Write `tests/test_rate_limiting.py` combining auth + rate limiting: authenticated requests exhaust the limit and receive `429` + `Retry-After`; unauthenticated requests still receive `401` (auth fires before rate limiting). | `tests/test_rate_limiting.py` | `pytest tests/test_rate_limiting.py -v` — all assertions green. |

---

## Phase 4 — Input Validation & Safe Error Handling

| Step | Task | Files affected | Test |
|------|------|---------------|------|
| 4.1 | **Enforce file-upload size cap.** Add a FastAPI dependency `validate_upload_size` that reads `Content-Length` and rejects requests exceeding `MAX_UPLOAD_BYTES` (default `10 MB`, configurable via env). If `Content-Length` is absent or too large, return `HTTP 413` with `{"detail": "Payload too large"}`. Apply this dependency to all file-upload routes. | `core/validation.py` (new), `main.py` | Test with a mocked upload of 10 MB + 1 byte → `413`. Test at exactly 10 MB → `2xx` (with valid key). |
| 4.2 | **Enforce MIME-type allowlist on file uploads.** In `validate_upload` (extend 4.1's module), inspect the uploaded file's declared `content_type` and validate it against an allowlist (e.g., `image/jpeg`, `image/png`, `application/pdf`). For disallowed types, return `HTTP 415` with `{"detail": "Unsupported media type"}`. Do **not** rely solely on the client-declared type — read the first 512 bytes and validate with `python-magic` as a secondary check. | `core/validation.py`, `requirements.txt` | Test with `content_type="application/zip"` → `415`. Test with a valid PNG header → passes. Test with a file whose declared type is `image/png` but whose magic bytes indicate a ZIP → `415`. |
| 4.3 | **Sanitise all exception and error responses.** Replace every `except Exception as e: print(e)` / `return str(e)` pattern with structured logging (`logging.exception("…")`) that writes to `stderr` but never echoes internal detail into the HTTP response body. All unhandled exceptions should be caught by a FastAPI `exception_handler` returning `HTTP 500` with `{"detail": "Internal server error"}`. | `main.py`, `core/*.py` | Trigger an intentional `ValueError` inside a mocked route. Assert the HTTP response body is exactly `{"detail": "Internal server error"}` with no stack trace, GCP project ID, or service-account metadata. Check `capfd` (pytest's captured stderr) to confirm the real error was logged server-side. |

---

## Phase 5 — Integration, Documentation & Regression

| Step | Task | Files affected | Test |
|------|------|---------------|------|
| 5.1 | **End-to-end integration test suite.** Write `tests/test_integration.py` covering the full happy path (valid key + valid file + under rate limit → `200`) and the four failure paths in sequence (no key → `401`, valid key + oversized file → `413`, valid key + bad MIME → `415`, valid key + 21 requests → `429`). Mock all Vertex AI / Gemini SDK calls so no live GCP calls are made. | `tests/test_integration.py` | `pytest tests/test_integration.py -v` — all 5 scenario assertions green; no network calls made (assert mocks were called, not real SDKs). |
| 5.2 | **Update environment documentation.** Update `.env.example` to document `SERVICE_API_KEY` (with generation instructions: `python -c "import secrets; print(secrets.token_hex(32))"`), `MAX_UPLOAD_BYTES`, and the existing GCP variables. Update `README.md` (or create one) with a "Running locally" section that lists the mandatory env vars and describes the auth header format (`X-API-Key: <value>`). | `.env.example`, `README.md` | Peer review / diff check — every variable referenced in `core/config.py` has a corresponding entry in `.env.example`. |
| 5.3 | **Full regression and coverage gate.** Run the complete test suite with coverage. Assert overall line coverage ≥ 80 % for `core/auth.py`, `core/validation.py`, and `core/rate_limit.py`. Fix any regressions introduced during the feature build. | All test files | `pytest --cov=core --cov-report=term-missing` exits 0; coverage report shows ≥ 80 % on the three new modules; zero pre-existing tests broken. |

---

## Definition of done

- [ ] Hardcoded API key (`***REMOVED-GOOGLE-KEY***`) is revoked in GCP console and no longer appears anywhere in source or git history (`git grep 'AIzaSy'` returns zero results)
- [ ] `F01` — all routes return `HTTP 401 {"detail": "Unauthorized"}` when `X-API-Key` header is absent or invalid
- [ ] `F02` — key validation uses `secrets.compare_digest` on SHA-256 digests; no timing-oracle possible
- [ ] `F03` — authentication is applied at app/router level; no per-route annotation required
- [ ] `F04` — 20 req/min per-IP enforced; 21st request returns `HTTP 429` with `Retry-After` header
- [ ] File uploads reject payloads > `MAX_UPLOAD_BYTES` (`HTTP 413`) and non-allowlisted MIME types (`HTTP 415`)
- [ ] No raw exception detail, GCP project ID, or service-account metadata appears in any HTTP response body
- [ ] All new modules (`core/auth.py`, `core/validation.py`, `core/rate_limit.py`) have ≥ 80 % line coverage
- [ ] `pytest` exits 0 with zero failing or skipped tests
- [ ] `.env.example` documents every required environment variable
- [ ] No regressions in existing extraction / marking pipeline behaviour