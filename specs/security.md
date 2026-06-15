# Security specification: AI_Extraction_Marking_service
**Risk level:** critical  **Score:** 0.05
## Summary
The AI_Extraction_Marking_service contains a critical, immediately exploitable credential exposure: a live Google Generative AI API key is hardcoded directly in source code (list_models.py). Any person with repository read access can use this key to make authenticated calls to Google AI APIs, potentially incurring unbounded financial charges or exfiltrating model outputs. Beyond this critical finding, the FastAPI surface in main.py exposes AI-invocation endpoints with no authentication, no rate limiting, and no file upload validation, making cost-exhaustion (billing DoS) and resource abuse trivially easy. Error handling throughout the codebase prints raw exception objects, leaking internal GCP project configuration. The combination of a leaked live credential, unauthenticated endpoints, and missing input controls makes this service immediately exploitable in its current state.
## Known vulnerabilities and required remediations
### [CRITICAL] exposure — list_models.py:4
A live Google Generative AI API key ('***REMOVED-GOOGLE-KEY***') is hardcoded as a string literal in source code. This key is fully functional at commit time and grants authenticated access to all Google Generative AI APIs associated with the owning account. Anyone with repository access (or who finds this in a public repo, CI log, container image, or compiled artifact) can immediately use it to query models, enumerate resources, and accumulate billable charges with no additional exploitation required. Google API keys of this type have no built-in expiry and will remain valid until manually revoked.

**Fix:** 1. Revoke this key immediately in the Google Cloud Console under APIs & Services → Credentials. 2. Replace with an environment-variable-backed pattern identical to the one already used in check_models_script.py: `api_key = os.getenv('GOOGLE_GENAI_API_KEY')`. 3. Add a pre-commit hook (e.g., git-secrets, truffleHog, or detect-secrets) to prevent future key commits. 4. Audit git history and any CI artefacts that may have captured this key.
### [HIGH] auth — main.py
The FastAPI application exposes file-upload and form-submission endpoints (UploadFile, File, Form parameters) with no visible authentication or authorization middleware. There is no API key check, no OAuth/JWT validation, and no session management. Any unauthenticated caller on the network can submit arbitrary content to the AI extraction/marking pipeline. Because each request invokes a paid Vertex AI / Gemini model call, this is both a data-exposure risk and a direct financial-abuse vector.

**Fix:** Add an authentication dependency to all routes. Minimum viable fix: `from fastapi.security import APIKeyHeader; api_key_header = APIKeyHeader(name='X-API-Key'); async def verify_key(key: str = Depends(api_key_header)): ...`. For production, prefer OAuth2 / GCP Identity-Aware Proxy. Apply the dependency at the router or app level so new routes are protected by default.
### [HIGH] other — main.py
There is no rate limiting on any endpoint. Because each request triggers an expensive Vertex AI / Gemini model invocation, an attacker (or even an accidental loop) can flood the service with requests, exhausting GCP quotas and running up unbounded API bills within minutes. This is a Denial-of-Wallet / billing DoS attack that requires zero authentication given the previous finding.

**Fix:** Add rate limiting at the application layer using `slowapi` (the FastAPI-compatible limiter): `from slowapi import Limiter; limiter = Limiter(key_func=get_remote_address)` and decorate endpoints with `@limiter.limit('10/minute')`. Additionally, configure GCP quota limits on the project and set budget alerts in Google Cloud Billing.
### [HIGH] exposure — main.py (UploadFile/File parameters)
File uploads are accepted with no enforced size limit, no MIME-type allowlist, and no content-type validation. An attacker can upload arbitrarily large files (exhausting memory/disk), files with misleading extensions, or potentially malformed images crafted to exploit parsing vulnerabilities in Pillow (PIL). The `Image` import from PIL confirms images are being opened directly from untrusted upload data. Pillow has a history of CVEs related to processing crafted image files.

**Fix:** 1. Enforce a maximum upload size: `if len(await file.read()) > 10_000_000: raise HTTPException(413, 'File too large')` (also configure uvicorn/nginx limits). 2. Validate MIME type against an allowlist before passing to PIL: `allowed = {'image/jpeg','image/png','image/webp'}; if file.content_type not in allowed: raise HTTPException(415, ...)`. 3. Keep Pillow pinned to a patched version and enable `Image.MAX_IMAGE_PIXELS` to prevent decompression bombs.
### [MEDIUM] config — check_models_script.py:22-27, model_loader.py:22-30
Raw exception objects are printed directly to stdout/stderr (`print(f"...: FAILED - {e}")`). In a cloud-native deployment, stdout is typically shipped to a centralised logging system. Exception messages from GCP SDKs commonly include the GCP project ID, service account email, IAM role details, and request metadata. This can aid an attacker in enumerating GCP project structure after gaining even read-only log access.

**Fix:** Use Python's `logging` module with structured output and set a minimum severity level. Catch specific exception types and log only a safe, sanitised message in production: `logging.error('Model initialisation failed', extra={'model': model_name})`. Never interpolate raw exception objects into log lines at INFO/ERROR level in production paths.
### [MEDIUM] config — main.py
No HTTP security headers are configured (Content-Security-Policy, X-Content-Type-Options, X-Frame-Options, Strict-Transport-Security). FastAPI does not add these by default. If the service ever serves any HTML responses or is fronted by a browser-facing layer, this omission enables clickjacking and MIME-sniffing attacks.

**Fix:** Add a middleware to inject security headers: `from starlette.middleware.base import BaseHTTPMiddleware` and set `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Content-Security-Policy: default-src 'none'` on every response. Consider using the `secure` PyPI package for a one-liner approach.
### [MEDIUM] config — main.py
No CORS policy is defined. FastAPI's default is to block cross-origin requests, but if a CORSMiddleware is added later (a common next step) without a strict origin allowlist, the service becomes accessible from arbitrary browser origins. Given the lack of authentication, this would allow any website to make cross-origin AI requests on behalf of a visiting user.

**Fix:** When adding CORSMiddleware, explicitly set `allow_origins` to a finite list of trusted origins rather than `['*']`. Example: `app.add_middleware(CORSMiddleware, allow_origins=['https://your-app.example.com'], allow_methods=['POST'], allow_headers=['X-API-Key'])`.
### [MEDIUM] exposure — token_logger.py:11-12
Billing/cost metadata (input tokens, output tokens, per-request cost calculations) are written to CSV files under a `logs/` directory relative to the module path. If this directory is inside a web-accessible location or a shared volume, an attacker could read usage patterns and infer request volumes, content size distributions, and approximate API spend — useful for competitive intelligence or targeted quota-exhaustion timing.

**Fix:** Write logs to a path outside the application root (e.g., `/var/log/ai_service/`) or, preferably, to Google Cloud Logging / BigQuery via the GCP logging SDK rather than local CSV files. Ensure the logs directory has restrictive filesystem permissions (0700) and is excluded from any static-file serving configuration.
### [LOW] other — main.py (Form subject parameter)
The `subject` form field is passed into a dictionary lookup (`SUBJECT_PROMPTS`) without validation against the defined key set. While the current code only uses it as a dict key (which would return None or a default on an unknown value), if the subject value is ever interpolated into a prompt string or used in a file path, this becomes an injection or path-traversal vector. The lack of validation is a latent risk.

**Fix:** Validate the subject value against the known allowlist before use: `if subject not in SUBJECT_PROMPTS: raise HTTPException(400, f'Unknown subject. Must be one of: {list(SUBJECT_PROMPTS.keys())}')`.

## Security NFRs (auto-enforced by speckit)
- All endpoints must validate and sanitize input
- No hardcoded credentials — all secrets via environment variables
- Auth required on all non-public endpoints
- Rate limiting on all public-facing endpoints
- All SQL via parameterised queries or ORM — no string interpolation
- Errors must not expose stack traces or internal paths in responses
