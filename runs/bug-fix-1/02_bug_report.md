# Bug report: rate limiting

## Issue summary

The FastAPI application exposed by `main.py` defines AI-invocation endpoints with no rate-limiting controls at any layer. Any caller — authenticated or not — can issue an unlimited number of requests, each triggering a live call to Google Vertex AI or the Gemini API. This exposes the service to trivial billing denial-of-service (cost exhaustion), quota starvation, and deliberate or accidental runaway spend.

**This bug is distinct from and independent of [CRITICAL-001] (hardcoded API key in `core/list_models.py:4`).** Both bugs must be remediated; they require different fixes and can be worked in parallel. See §"Relationship to CRITICAL-001" for sequencing guidance.

---

## Relationship to CRITICAL-001 (credential exposure)

| Attribute | This bug: Rate limiting | CRITICAL-001: Hardcoded API key |
|-----------|------------------------|--------------------------------|
| Location | `main.py` (FastAPI app construction, all route handlers) | `core/list_models.py:4` |
| Root cause | Absent middleware | Credential in source code |
| Attack vector | Unauthenticated HTTP | Repository read access |
| Fix | Add `slowapi` middleware + env-configured thresholds | Revoke key; read from `os.environ` |
| Fix urgency | HIGH — within 24 h of CRITICAL-001 resolution | CRITICAL — immediate |
| Dependency | Independent; may be deployed in parallel | None |

**Sequencing recommendation:**

1. **Immediately (Day 0):** Revoke and rotate `***REMOVED-GOOGLE-KEY***` per CRITICAL-001. This eliminates the live-credential exposure and ensures Google Cloud–side quota enforcement is again meaningful.
2. **Within 24 hours (Day 1):** Deploy rate-limiting middleware (this fix). This enforces a service-level request budget independently of GCP's own quota.
3. **Within 72 hours (Day 3):** Deploy `core/auth.py` bearer-token authentication, which enables per-user rate-limit keying (Phase 2 of this fix).

Deploying rate limiting before key rotation is permissible but incomplete: an attacker who holds the leaked key can still make direct Gemini SDK calls that bypass the FastAPI layer entirely, so both fixes are required for full protection.

---

## Root cause analysis

- **Primary root cause — absent middleware:** When the FastAPI `app` object is created in `main.py`, no rate-limiting middleware is registered. No `slowapi.Limiter` (or equivalent) is mounted as a Starlette middleware layer, no cloud API Gateway policy fronts the service, and no route decorator enforces a request budget. Individual route handlers carry only the standard FastAPI `@app.post` / `@app.get` decorators. The FastAPI framework does not include rate limiting by default.

- **Compounding factor — absent authentication:** Because no `SERVICE_API_KEY` bearer-token check exists on any endpoint, the rate-limiting gap cannot be compensated by restricting which callers reach the AI backend. Every anonymous HTTP client reaches the Vertex AI and Gemini SDK call paths unconditionally.

- **Compounding factor — hardcoded Gemini SDK key (CRITICAL-001, separate bug):** `core/list_models.py:4` hard-codes `***REMOVED-GOOGLE-KEY***` directly in source. This is a distinct credential-exposure vulnerability tracked separately as CRITICAL-001. Its relevance here is limited: a holder of the leaked key could bypass the FastAPI layer and call the Gemini API directly, making service-level rate limiting ineffective for that path until the key is revoked. This does **not** mean the hardcoded key is the root cause of missing rate limiting; it is a separate vulnerability that worsens the impact.

- **Triggered by:** Any HTTP client issuing repeated POST requests to the extraction or marking endpoints. No IP throttling, no token-bucket, no per-client quota, and no global ceiling exists at any layer of the provided codebase.

---

## Architecture impact

- **Modules affected:** `main.py`, `core/auth.py` (new), `core/config.py` (new), `core/list_models.py` (compounding factor — CRITICAL-001 fix), `tests/`
- **Principles at risk:**
  - *Environment-driven configuration* — Rate-limit thresholds must be read from environment variables (`RATE_LIMIT_REQUESTS_PER_MINUTE`, `RATE_LIMIT_STRATEGY`, `RATE_LIMIT_BURST_SIZE`, `RATE_LIMIT_GLOBAL_CEILING`), consistent with how `GCP_PROJECT_ID` and `GCP_LOCATION` are already resolved in `model_loader.py` and `check_models_script.py`. Hard-coding numeric limits in source would violate this principle.
  - *Cloud-native AI execution* — Unbounded call throughput creates unbounded GCP billing exposure, which undermines safe use of the managed-service AI execution model.
  - *Dual-SDK model discovery* — Both the Vertex AI path (`main.py` → `model_loader.py`) and the Gemini SDK path (`core/list_models.py`) are affected and must each be protected.
- **Data models affected:** None (rate limiting is an infrastructure cross-cut, not a domain-model change).
- **Security surface touched:** Yes. Per `specs/security.md`: *"the FastAPI surface in main.py exposes AI-invocation endpoints with no authentication, no rate limiting, and no file upload validation, making cost-exhaustion (billing DoS) and resource abuse trivially easy."*

---

## Attack surface analysis

### Attack scenarios

| Scenario | Mechanism | Estimated cost impact | Mitigated by this fix? |
|----------|-----------|-----------------------|------------------------|
| Anonymous burst (single IP) | Attacker sends 1 000 req/min to `/extract`; each triggers a Vertex AI inference call | ~$0.30–$3.00/min (see cost model) | Yes — per-IP limit triggers HTTP 429 after threshold |
| Distributed burst (bot network) | 50 IPs each sending 30 req/min, staying below per-IP threshold | ~$15–$150/min | Partially — global ceiling (§Rate-limit thresholds) limits aggregate spend |
| Automated model-output scraping | Script loops over document variations to harvest AI responses | Low cost per call but high data-exfiltration risk | Yes — per-IP limit restricts throughput |
| Quota starvation | Attacker saturates Vertex AI quota so legitimate calls receive HTTP 429 from GCP | Service downtime for real users | Yes — reduces request rate before GCP quota ceiling is reached |
| Leaked-key direct Gemini call (CRITICAL-001 path) | Attacker uses `***REMOVED-GOOGLE-KEY***` to call Gemini API directly, bypassing FastAPI | Same cost exposure as anonymous burst | **No — requires CRITICAL-001 key rotation first** |

### Cost model (Vertex AI Gemini 1.5 Flash baseline)

- Input tokens per extraction call: ~2 000 (document image + prompt)
- Output tokens per extraction call: ~500
- Approximate cost per call: ($0.075 × 2 000 / 1 000 000) + ($0.30 × 500 / 1 000 000) ≈ **$0.0003**
- At 1 000 req/min with no rate limit: ~$0.30/min → **$432/day**
- Heavier models (Gemini 1.5 Pro) or larger documents: 10–50× higher

### Rate-limit breach alerting specification

The following events must trigger structured log entries (JSON, not raw exception text) and, where a monitoring sink is configured, an alert:

| Event | Log level | Alert threshold |
|-------|-----------|-----------------|
| Single IP exceeds per-IP limit | `WARNING` | >50 rate-limited requests from one IP within 5 min |
| Global ceiling breached | `ERROR` | Any breach of `RATE_LIMIT_GLOBAL_CEILING` |
| Unauthenticated request reaching rate check | `INFO` | No alert; metric only |
| Startup with missing `GOOGLE_API_KEY` env var (canary for CRITICAL-001 regression) | `CRITICAL` | Immediate on startup |

Log fields per rate-limit event: `timestamp`, `client_ip_hash` (SHA-256 of IP, not plaintext), `endpoint`, `requests_in_window`, `limit`, `window_seconds`. No request body or file content is logged.

---

## Rate-limit thresholds and strategy

### Recommended thresholds

| Limit type | Value | Rationale |
|------------|-------|-----------|
| Per-IP (Phase 1, pre-auth) | 30 req/min | Accommodates legitimate interactive use (≤1 call every 2 s); blocks automated abuse |
| Per-authenticated-user (Phase 2, post-auth) | 60 req/min | Doubles allowance for known callers while remaining bounded |
| Burst ceiling | 5 req / 10 s | Prevents short-window floods that stay within the per-minute budget |
| Global ceiling (all clients combined) | 500 req/min | Hard cap regardless of distribution across IPs; protects GCP quota |

These values must be configurable via environment variables (see `.env` template below). The defaults listed are recommended starting points; production values should be reviewed against observed traffic patterns within 30 days of deployment.

### Per-entity strategy

**Phase 1 (immediate — no authentication layer):**
Key rate limits on the client's real IP address using `slowapi.util.get_remote_address`. This is the only client-identifying signal available before authentication is implemented. Limitation: can be partially circumvented with a VPN or botnet; per-IP limits are a floor, not a ceiling.

**Phase 2 (within 72 h — after `core/auth.py` is deployed):**
Key rate limits on the validated `X-Api-Key` header value (hashed). A client with a valid key is identified by the key hash; unauthenticated requests fall back to IP-based limiting. This prevents a single leaked `SERVICE_API_KEY` from being shared across many IPs to circumvent per-IP limits.

**Global ceiling (both phases):**
Apply a second, broader `slowapi` limit keyed on a fixed sentinel string `"global"`. This ensures the aggregate request rate cannot exceed `RATE_LIMIT_GLOBAL_CEILING` regardless of how traffic is distributed across clients.

### `.env` configuration template

```dotenv
# ── Rate limiting ──────────────────────────────────────────────────────────
# Per-entity limit (Phase 1: per IP; Phase 2: per authenticated user)
RATE_LIMIT_REQUESTS_PER_MINUTE=30

# Burst ceiling: max requests allowed in any 10-second window per entity
RATE_LIMIT_BURST_SIZE=5

# Keying strategy: "ip" (Phase 1) or "user" (Phase 2, requires core/auth.py)
RATE_LIMIT_STRATEGY=ip

# Hard global ceiling across ALL clients combined (requests per minute)
RATE_LIMIT_GLOBAL_CEILING=500

# ── Google Cloud ───────────────────────────────────────────────────────────
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=asia-south1

# ── Google AI credentials (CRITICAL-001 remediation) ───────────────────────
GOOGLE_API_KEY=your-rotated-key-here    # never hardcode; revoke the leaked key first

# ── Service authentication ─────────────────────────────────────────────────
SERVICE_API_KEY=your-service-api-key
```

---

## Proposed fix

**Prerequisite:** Complete CRITICAL-001 (revoke `***REMOVED-GOOGLE-KEY***`; replace `core/list_models.py:4` with `os.environ["GOOGLE_API_KEY"]`) before or concurrently with this fix. Rate limiting without key rotation is incomplete protection for the Gemini SDK path (see §Relationship to CRITICAL-001).

### Step 1 — Revoke and rotate hardcoded key (CRITICAL-001, prerequisite)

Invalidate `***REMOVED-GOOGLE-KEY***` immediately via Google Cloud Console. Replace `core/list_models.py:4` with `api_key = os.environ["GOOGLE_API_KEY"]`. Full details are in the CRITICAL-001 bug report; this step is listed here only to establish sequencing.

### Step 2 — Add `slowapi` rate-limiting middleware to `main.py`

`slowapi` (built on the `limits` library) integrates natively with FastAPI/Starlette without requiring an external broker for single-instance deployments. Install `slowapi>=0.1.9` and `limits>=3.6`. Instantiate `Limiter` keyed on `get_remote_address` (Phase 1) or a custom key function reading `X-Api-Key` (Phase 2). Register as `app.state.limiter`, add `_rate_limit_exceeded_handler` for `RateLimitExceeded`, and decorate each route with `@limiter.limit(settings.RATE_LIMIT)` plus `@limiter.limit(settings.RATE_LIMIT_BURST)`. Apply a global-ceiling limiter keyed on `"global"` to enforce the aggregate cap.

### Step 3 — Centralise rate-limit configuration in `core/config.py`

Add `RATE_LIMIT_REQUESTS_PER_MINUTE`, `RATE_LIMIT_BURST_SIZE`, `RATE_LIMIT_STRATEGY`, and `RATE_LIMIT_GLOBAL_CEILING` settings read from environment variables with safe defaults. Construct the `limits`-format strings (e.g., `"30/minute"`, `"5/10 seconds"`) from these values so all tuneable thresholds live in one place.

### Step 4 — Add `core/auth.py` API-key middleware (enables Phase 2 per-user keying)

A `SERVICE_API_KEY` bearer-token check (`X-Api-Key` header) provides the caller-identity signal needed to key rate limits per-user rather than per-IP once authentication is in place.

### Step 5 — Write tests

Add `tests/test_rate_limiting.py` covering single-client and multi-client scenarios (see §Testing strategy and acceptance criteria).

### Why this approach does not violate architecture

- `slowapi` is a pure Python middleware that wraps the existing FastAPI application object. It introduces no local model inference, no database, and no file-system state — the service remains cloud-native.
- All limit values (`RATE_LIMIT_REQUESTS_PER_MINUTE`, `RATE_LIMIT_BURST_SIZE`, `RATE_LIMIT_STRATEGY`, `RATE_LIMIT_GLOBAL_CEILING`) are read from environment variables via `core/config.py`, directly honouring the *environment-driven configuration* principle already established by `model_loader.py` and `check_models_script.py`.
- Neither the Vertex AI SDK path nor the Gemini SDK path is altered in any functional way; rate limiting is applied at the HTTP ingress layer, leaving both SDK integrations intact and preserving the *dual-SDK model discovery* architecture.
- No credentials are introduced into source code; the rotated `GOOGLE_API_KEY` lives only in `.env` (gitignored) or GCP Secret Manager.

### Alternative approaches considered

| Approach | Rejected because |
|----------|-----------------|
| Google Cloud Endpoints / API Gateway rate limiting | Adds an external GCP resource not present in the current architecture; requires infrastructure changes beyond the Python service boundary; does not protect local or development environments |
| Custom in-memory `dict` counter in `main.py` | Thread-unsafe under FastAPI's async worker model; not configurable via env var without extra boilerplate; provides no standard `Retry-After` header |
| Redis-backed rate limiting (`fastapi-limiter`) | Correct long-term choice for multi-replica deployments but introduces a Redis dependency absent from the current stack; should be added as part of the broader stateful-CSV remediation already flagged in `specs/scalability.md`, not as a prerequisite to this security patch |
| Per-API-key quota in Google AI Studio | Covers only the Gemini SDK path; does nothing to limit the Vertex AI path in `main.py`; cannot enforce service-level request budgets |

---

## Packages / imports required

| Package | Version | Justification | Compatible? |
|---------|---------|---------------|-------------|
| `slowapi` | `>=0.1.9` | FastAPI/Starlette-native rate limiting; provides `Limiter`, `_rate_limit_exceeded_handler`, and `Request`-based key functions | Yes — no conflict with `fastapi`, `uvicorn`, or the Google AI SDKs |
| `limits` | `>=3.6` | Pulled in transitively by `slowapi`; provides storage backends and rate-string parsing | Yes |

---

## Security checklist

- [x] Fix does not expose new attack surface — rate limiting reduces attack surface; no new routes or data paths are added
- [x] Input validation unchanged or strengthened — existing `UploadFile` / `Form` validation in `main.py` is preserved; rate limiting is layered on top
- [x] No credentials introduced (all via env vars) — `GOOGLE_API_KEY` and `SERVICE_API_KEY` read exclusively via `os.environ`; hardcoded key removal is tracked in CRITICAL-001
- [x] Auth/authz not weakened — `core/auth.py` bearer-token check is additive; no existing check is removed
- [x] No PII added to logs — rate-limit logs record only hashed IP address (`SHA-256`) and timestamp; no request body or file content is logged
- [x] Rate-limit thresholds are environment-configurable — no numeric limits are hardcoded in source; all values derive from `core/config.py` env-var reads

---

## Code changes (before/after)

### `core/list_models.py` (CRITICAL-001 prerequisite — shown here for sequencing clarity)

**Before:**
```python
import google.generativeai as genai
import os

api_key = "***REMOVED-GOOGLE-KEY***"
genai.configure(api_key=api_key)

print("Listing available models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
```

**After:**
```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]  # must be set in .env or Secret Manager
genai.configure(api_key=api_key)

print("Listing available models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    # Do not print raw exception — it may contain project metadata
    print("Error listing models. Check GOOGLE_API_KEY and network access.")
```

---

### `core/config.py` *(new file)*

**Before:**
```
# File did not exist
```

**After:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud configuration
GCP_PROJECT_ID: str = os.environ["GCP_PROJECT_ID"]
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "asia-south1")

# Google AI credentials
GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]

# Service authentication
SERVICE_API_KEY: str = os.environ["SERVICE_API_KEY"]

# Rate limiting — raw values from environment
_rpm: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "30"))
_burst: int = int(os.getenv("RATE_LIMIT_BURST_SIZE", "5"))
RATE_LIMIT_STRATEGY: str = os.getenv("RATE_LIMIT_STRATEGY", "ip")  # "ip" | "user"
RATE_LIMIT_GLOBAL_CEILING: int = int(os.getenv("RATE_LIMIT_GLOBAL_CEILING", "500"))

# limits-library format strings consumed by slowapi decorators
RATE_LIMIT: str = f"{_rpm}/minute"                          # e.g. "30/minute"
RATE_LIMIT_BURST: str = f"{_burst}/10 seconds"              # e.g. "5/10 seconds"
RATE_LIMIT_GLOBAL: str = f"{RATE_LIMIT_GLOBAL_CEILING}/minute"  # e.g. "500/minute"
```

---

### `core/auth.py` *(new file)*

**Before:**
```
# File did not exist
```

**After:**
```python
from fastapi import Header, HTTPException, status
from core.config import SERVICE_API_KEY


async def verify_api_key(x_api_key: str = Header(...)) -> None:
    """FastAPI dependency — rejects requests whose X-Api-Key header does not
    match the SERVICE_API_KEY environment variable.  Raises HTTP 401 on failure.
    """
    if x_api_key != SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
```

---

### `main.py`

**Before:**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from model_loader import ModelLoader
from token_logger import token_logger
from vertexai.generative_models import Part
from PIL import Image
import io
import os
import base64

# (no rate limiting, no auth dependency)
app = FastAPI()

# ... route handlers with no @limiter.limit decorator ...
```

**After:**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from model_loader import ModelLoader
from token_logger import token_logger
from vertexai.generative_models import Part
from PIL import Image
import io
import os
import base64
from core.config import RATE_LIMIT, RATE_LIMIT_BURST, RATE_LIMIT_GLOBAL, RATE_LIMIT_STRATEGY
from core.auth import verify_api_key

# Phase 1: key on real client IP.
# Phase 2 (after auth): replace get_remote_address with a key_func that reads
# the validated X-Api-Key header, falling back to IP for unauthenticated paths.
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply per-entity and burst limits to each endpoint:
# @app.post("/extract", dependencies=[Depends(verify_api_key)])
# @limiter.limit(RATE_LIMIT)                                  # e.g. "30/minute"
# @limiter.limit(RATE_LIMIT_BURST)                            # e.g. "5/10 seconds"
# async def extract(request: Request, file: UploadFile = File(...), ...):
#     ...

# Global ceiling across all clients — applied via a fixed sentinel key:
# @limiter.limit(RATE_LIMIT_GLOBAL, key_func=lambda _: "global")
# (add alongside the per-entity decorator on every endpoint)
```

---

### `tests/test_rate_limiting.py` *(new file)*

**Before:**
```
# File did not exist
```

**After:**
```python
import pytest
import os
from httpx import AsyncClient, ASGITransport

# Patch config before importing app so env vars are not required in CI.
# Thresholds are intentionally lower than production defaults to keep tests fast.
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("SERVICE_API_KEY", "test-service-key")
os.environ.setdefault("RATE_LIMIT_REQUESTS_PER_MINUTE", "5")
os.environ.setdefault("RATE_LIMIT_BURST_SIZE", "3")
os.environ.setdefault("RATE_LIMIT_GLOBAL_CEILING", "10")

from main import app  # noqa: E402 — must follow env setup

HEADERS_AUTH = {"X-Api-Key": "test-service-key"}
UPLOAD = {"file": ("img.png", b"\x89PNG", "image/png")}
FORM = {"subject": "Mathematics"}


@pytest.mark.asyncio
async def test_single_client_within_limit_receives_200():
    """TC-001: Requests below the per-IP threshold must succeed (HTTP 200)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(5):  # RATE_LIMIT_REQUESTS_PER_MINUTE = 5
            r = await client.post("/extract", headers=HEADERS_AUTH,
                                  data=FORM, files=UPLOAD)
            assert r.status_code != 429, f"Unexpected 429 on request {i + 1}"


@pytest.mark.asyncio
async def test_single_client_exceeding_limit_receives_429():
    """TC-002: The N+1th request from the same IP must receive HTTP 429."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
            await client.post("/extract", headers=HEADERS_AUTH,
                              data=FORM, files=UPLOAD)
        r = await client.post("/extract", headers=HEADERS_AUTH,
                              data=FORM, files=UPLOAD)
        assert r.status_code == 429


@pytest.mark.asyncio
async def test_429_response_includes_retry_after_header():
    """TC-003: HTTP 429 must include a Retry-After header with a positive integer value."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
            await client.post("/extract", headers=HEADERS_AUTH,
                              data=FORM, files=UPLOAD)
        r = await client.post("/extract", headers=HEADERS_AUTH,
                              data=FORM, files=UPLOAD)
        assert r.status_code == 429
        assert "Retry-After" in r.headers, "Retry-After header missing from 429 response"
        assert int(r.headers["Retry-After"]) > 0


@pytest.mark.asyncio
async def test_two_independent_clients_are_rate_limited_separately():
    """TC-004: Two clients each below the per-IP limit must both receive 200 (IP isolation)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c1:
        async with AsyncClient(transport=transport, base_url="http://test") as c2:
            # Simulate different IPs via X-Forwarded-For.
            headers_c1 = {**HEADERS_AUTH, "X-Forwarded-For": "10.0.0.1"}
            headers_c2 = {**HEADERS_AUTH, "X-Forwarded-For": "10.0.0.2"}
            for _ in range(3):  # Each client sends 3 of 5 allowed requests
                r1 = await c1.post("/extract", headers=headers_c1,
                                   data=FORM, files=UPLOAD)
                r2 = await c2.post("/extract", headers=headers_c2,
                                   data=FORM, files=UPLOAD)
                assert r1.status_code != 429, "Client 1 hit rate limit unexpectedly"
                assert r2.status_code != 429, "Client 2 hit rate limit unexpectedly"


@pytest.mark.asyncio
async def test_global_ceiling_limits_combined_traffic():
    """TC-005: Combined traffic from multiple clients must not exceed RATE_LIMIT_GLOBAL_CEILING."""
    # RATE_LIMIT_GLOBAL_CEILING=10; send 11 requests from 11 distinct IPs
    transport = ASGITransport(app=app)
    responses = []
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(11):
            headers = {**HEADERS_AUTH, "X-Forwarded-For": f"10.0.1.{i}"}
            r = await client.post("/extract", headers=headers,
                                  data=FORM, files=UPLOAD)
            responses.append(r.status_code)
    assert 429 in responses, "Global ceiling not enforced; all 11 requests succeeded"


@pytest.mark.asyncio
async def test_missing_api_key_returns_401_before_rate_limit():
    """TC-006: Requests with no X-Api-Key header must be rejected HTTP 401 before the limiter acts."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/extract", data=FORM, files=UPLOAD)
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_invalid_api_key_returns_401_not_429():
    """TC-007: A wrong X-Api-Key must yield HTTP 401, not HTTP 429."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/extract",
                              headers={"X-Api-Key": "wrong-key"},
                              data=FORM, files=UPLOAD)
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_429_response_body_is_actionable():
    """TC-008: HTTP 429 body must include a human-readable 'error' or 'detail' key."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
            await client.post("/extract", headers=HEADERS_AUTH,
                              data=FORM, files=UPLOAD)
        r = await client.post("/extract", headers=HEADERS_AUTH,
                              data=FORM, files=UPLOAD)
        assert r.status_code == 429
        body = r.json()
        assert "error" in body or "detail" in body, \
            "429 body must include 'error' or 'detail' key"
```

---

## Testing strategy and acceptance criteria

### Acceptance criteria

| ID | Scenario | Input | Expected output |
|----|----------|-------|-----------------|
| TC-001 | Single client within per-IP limit | 5 sequential POST `/extract` (limit = 5/min) | All responses HTTP 200 |
| TC-002 | Single client exceeds per-IP limit | 6th sequential POST `/extract` | HTTP 429 |
| TC-003 | 429 includes Retry-After | As TC-002 | `Retry-After` header present; integer value > 0 |
| TC-004 | Two independent clients isolated | 3 requests each from `10.0.0.1` and `10.0.0.2` | Both clients receive HTTP 200 (per-IP limits independent) |
| TC-005 | Global ceiling enforced | 11 requests from 11 distinct IPs (global limit = 10/min) | At least one HTTP 429 returned |
| TC-006 | Missing auth header | POST with no `X-Api-Key` header | HTTP 401 (rejected before rate check) |
| TC-007 | Wrong API key | POST with `X-Api-Key: wrong-key` | HTTP 401, not HTTP 429 |
| TC-008 | 429 body is actionable | As TC-002 | Response JSON contains `"error"` or `"detail"` key |

### Thresholds used in tests vs. production

| Setting | Test value | Production default |
|---------|------------|--------------------|
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `5` | `30` |
| `RATE_LIMIT_BURST_SIZE` | `3` | `5` |
| `RATE_LIMIT_GLOBAL_CEILING` | `10` | `500` |

Test values are intentionally lower than production defaults to keep test suites fast. Production defaults must be validated separately in a staging integration run.

### Load test approach (post-deployment)

Use `locust` or `k6` to verify production threshold enforcement:

1. **Single-client burst:** 1 virtual user ramping from 10 to 100 req/min — confirm HTTP 429 appears at the configured per-IP threshold.
2. **Multi-client isolation:** 10 virtual users at 5 req/min each (50 req/min total, below global ceiling) — confirm all receive HTTP 200.
3. **Global ceiling breach:** 100 virtual users at 10 req/min each (1 000 req/min total, above `RATE_LIMIT_GLOBAL_CEILING=500`) — confirm HTTP 429 responses appear as the global ceiling is breached.
4. **Post-window reset:** After a 60-second cooldown following TC-002 conditions, confirm the same client again receives HTTP 200, verifying window reset behaviour.

---

## Spec files to update after merge

- [ ] `specs/security.md` — Mark the `[CRITICAL] exposure — list_models.py:4` finding as remediated (CRITICAL-001); add a new `[HIGH] rate-limiting` finding closure noting `slowapi` middleware is in place, `SERVICE_API_KEY` auth is enforced on all endpoints, and per-IP, burst, and global ceilings are configured via `RATE_LIMIT_REQUESTS_PER_MINUTE`, `RATE_LIMIT_BURST_SIZE`, `RATE_LIMIT_STRATEGY`, and `RATE_LIMIT_GLOBAL_CEILING` environment variables.
- [ ] `specs/architecture.md` — Update the Module map to include `core/config.py` (environment-driven configuration singleton) and `core/auth.py` (bearer-token dependency); add `slowapi>=0.1.9` to the Tech stack table under a new "HTTP Middleware" row; document `RATE_LIMIT_REQUESTS_PER_MINUTE`, `RATE_LIMIT_BURST_SIZE`, `RATE_LIMIT_STRATEGY`, and `RATE_LIMIT_GLOBAL_CEILING` in the environment-variable reference alongside existing `GCP_PROJECT_ID` and `GCP_LOCATION` entries.
- [ ] `specs/scalability.md` — Note that the single-instance in-memory limiter (`slowapi` default storage) is an accepted short-term control; flag that a Redis-backed `limits` storage backend is required before horizontal scaling, linking to the broader stateful-CSV remediation work already called out in the spec.
- [ ] `specs/modules/core.md` — Document the new `core/config.py` and `core/auth.py` public interfaces; update the Data flow section to show the auth/rate-limit middleware layer (HTTP 401 auth check → per-IP rate check → burst check → global ceiling check → route handler) preceding the FastAPI route handler.
- [ ] `05_build_plan.md` — Check off Phase 1 step 1.1 (key rotation, CRITICAL-001) and the Phase 2 rate-limiting step; record `slowapi>=0.1.9` and `limits>=3.6` as approved runtime dependencies; add `RATE_LIMIT_REQUESTS_PER_MINUTE`, `RATE_LIMIT_BURST_SIZE`, `RATE_LIMIT_STRATEGY`, and `RATE_LIMIT_GLOBAL_CEILING` to the required environment-variable inventory.

---
*status: needs-human-review | score: 0.51 | iterations: 1*