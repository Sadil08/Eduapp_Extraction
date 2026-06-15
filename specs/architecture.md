```markdown
# AI_Extraction_Marking_service — Architecture

## Overview
AI_Extraction_Marking_service is a Python service that uses Google's generative AI
infrastructure to perform automated extraction and marking (grading/scoring) tasks.
It integrates with both the Google Vertex AI SDK and the Google Generative AI (Gemini)
SDK to discover and invoke large-language models capable of generating content. The
service is designed to run in a Google Cloud environment, defaulting to the
`asia-south1` region.

## Tech stack
| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| AI runtime (primary) | Google Vertex AI (`vertexai` SDK) — `GenerativeModel` |
| AI runtime (secondary) | Google Generative AI SDK (`google.generativeai`) |
| Configuration | `python-dotenv` — `.env` file loaded at startup |
| Infra / Cloud | Google Cloud Platform (GCP) — Vertex AI, Gemini models |
| Framework | None observed — plain Python scripts |
| DB | None observed in provided files |
| Testing | None observed in provided files |

## Module map
| Module | Responsibility |
|--------|---------------|
| `core` (5 files) | Houses the primary application logic: model initialisation, AI model discovery (`list_models.py`, `check_models_script.py`), and presumably the extraction and marking pipeline (remaining 3 files not provided) |

## Architecture principles
- **Cloud-native AI execution.** All generative workloads are delegated to Google
  Cloud managed services (Vertex AI, Gemini); no local model inference is observed.
- **Dual-SDK model discovery.** The project maintains two complementary integration
  paths — `vertexai.generative_models.GenerativeModel` for production Vertex AI
  endpoints and `google.generativeai` for direct Gemini API access — enabling
  flexible model targeting.
- **Environment-driven configuration.** Runtime parameters (project ID, region) are
  resolved exclusively through environment variables loaded via `python-dotenv`,
  keeping deployment configuration outside source code.
- **Fail-fast on missing configuration.** Critical environment values (e.g.
  `GCP_PROJECT_ID`) are validated at startup; the process exits immediately if they
  are absent rather than propagating errors deeper.
- **Defensive model invocation.** Model calls are wrapped in `try/except` blocks,
  catching and surfacing exceptions at the call site to prevent unhandled crashes.
- **Script-oriented structure.** The core module is composed of standalone Python
  scripts rather than a web framework or daemon, suggesting a batch or
  command-line-invoked execution model.
- **Region-aware deployment.** A hardcoded default region (`asia-south1`) signals
  intentional data-residency or latency decisions for the target deployment
  geography.

## Cross-cutting concerns
- **Logging:** `print()` statements only; no structured logging framework (`logging`,
  `structlog`, etc.) is observed in the provided files. Log output goes to stdout.
- **Error handling:** Localised `try/except Exception as e` blocks catch errors at
  the point of model invocation and print a diagnostic message. No centralised error
  handler or retry logic is visible.
- **Configuration:** Environment variables are the sole configuration mechanism.
  `python-dotenv` loads a `.env` file at module import time via `load_dotenv()`.
  Required variables: `GCP_PROJECT_ID`; optional with defaults: `GCP_LOCATION`
  (default `"asia-south1"`). **One file (`list_models.py`) contains a hardcoded
  API key — this is a deviation from the env-var pattern and constitutes a security
  defect.**
- **Testing:** No test files, test framework, or test fixtures are present in the
  provided material. Test strategy is undetermined from available evidence.

## Security model
- **Authentication:** Vertex AI authentication follows the GCP Application Default
  Credentials (ADC) chain, initialised via `vertexai.init(project=..., location=...)`.
  The secondary SDK path uses a bare API key passed to `genai.configure()`.
- **Authorisation:** Authorisation is governed by GCP IAM roles bound to the service
  account or user credential resolved by ADC. No application-layer authorisation
  logic is observed.
- **Secrets management:** GCP project identity is stored in `.env` and loaded via
  `python-dotenv`. **`list_models.py` contains a plaintext API key hardcoded in
  source — this key must be rotated and moved to an environment variable or a secret
  manager (e.g. Google Secret Manager) immediately.**
- **Input validation:** No input validation logic is visible in the provided files.
  Inputs to the generative models are assumed to be passed directly; sanitisation
  and prompt-injection controls are not observed.

## Data flow (top level)
1. **Startup** — The process loads environment variables from `.env` via
   `load_dotenv()` and validates that `GCP_PROJECT_ID` is present.
2. **Vertex AI initialisation** — `vertexai.init()` establishes the GCP project and
   regional context using ADC-resolved credentials.
3. **Model selection** — Available generative models that support `generateContent`
   are enumerated (via `genai.list_models()`) or a specific `GenerativeModel` is
   instantiated by name on the Vertex AI endpoint.
4. **Extraction / Marking** — (Inferred from project name; implementation not visible
   in provided files) Input content — likely documents or answers — is submitted to
   the selected generative model, which returns extracted information or a
   marked/scored result.
5. **Output** — Results are emitted to stdout or consumed by a downstream caller;
   no persistence layer is observed in the available code.
```