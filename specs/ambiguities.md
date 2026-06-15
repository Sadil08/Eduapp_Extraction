# Architectural ambiguities: AI_Extraction_Marking_service
## Summary
The service has a critical security vulnerability (hardcoded credential in source code) and suffers from pervasive configuration inconsistency across its own files. Beyond those acute issues, the dual-SDK strategy, mixed script/production code cohabitation in core, and three undocumented pipeline files create significant onboarding and maintenance risk.
## Ambiguities (10 found)
### [HIGH] security
**Locations:** list_models.py

A live Google Generative AI API key ('***REMOVED-GOOGLE-KEY***') is hardcoded directly in list_models.py. This key is now permanently in version-control history, grants unauthenticated access to the Gemini API under the project's billing account, and will be exposed to every developer, CI runner, and log aggregator that touches this repo.

**Recommendation:** Revoke the key immediately via Google Cloud Console. Re-issue a new key stored only in the .env file (or Secret Manager) and read via os.getenv(), consistent with how check_models_script.py and model_loader.py handle credentials. Add a pre-commit hook (e.g., detect-secrets or gitleaks) to prevent recurrence.
### [HIGH] config
**Locations:** check_models_script.py, list_models.py, model_loader.py

Three files use three different credential/configuration strategies for essentially the same Google AI backend: check_models_script.py reads project/location from env vars via dotenv; list_models.py uses a hardcoded API key with the google.generativeai SDK; model_loader.py reads env vars via dotenv for Vertex AI. There is no single authoritative configuration source or loading contract, so a new engineer cannot know which pattern to follow when adding a fourth file.

**Recommendation:** Define one configuration module (e.g., core/config.py) that calls load_dotenv() once and exposes typed constants (GCP_PROJECT_ID, GCP_LOCATION, GOOGLE_API_KEY). All other modules import from it. This eliminates repeated dotenv calls and the temptation to inline credentials.
### [MEDIUM] config
**Locations:** model_loader.py

model_loader.py reads os.getenv() at class-body evaluation time (as class-level variable defaults), before any caller can guarantee load_dotenv() has already run. If another module imports ModelLoader before dotenv is loaded, _project_id and _location will silently be None, causing a deferred and confusing Vertex AI initialization failure rather than an early, clear startup error.

**Recommendation:** Move env-var reads inside get_instance() (or a dedicated _initialize() method called lazily), after load_dotenv() is confirmed to have been called. Alternatively, use the centralized config module described above so that load_dotenv() is guaranteed to run at import time in one place only.
### [MEDIUM] naming
**Locations:** check_models_script.py, list_models.py, model_loader.py

The service maintains two parallel AI SDK integrations — vertexai.generative_models.GenerativeModel and google.generativeai — with no documented policy distinguishing when each is used. The architecture spec describes this as 'dual-SDK model discovery enabling flexible model targeting,' but flexible without a decision rule is just ambiguous. A new engineer adding a model call has no basis for choosing between the two SDKs, risking drift into a third implicit variant.

**Recommendation:** Document the explicit rule (e.g., 'Vertex AI SDK for all production inference; google.generativeai only for local dev/model enumeration where Vertex quotas are unavailable'). Enforce it via a linting rule or an ADR (Architecture Decision Record) checked into the repo.
### [MEDIUM] coupling
**Locations:** check_models_script.py, list_models.py, core

check_models_script.py and list_models.py are standalone diagnostic scripts (they call exit(), print to stdout, and have no callable API) but are co-located inside the core module alongside production code (main.py, model_loader.py, token_logger.py). Any import of the core package risks pulling in module-level side effects (vertexai.init, genai.configure, print statements, exit() calls) from these scripts, which would break the FastAPI application process.

**Recommendation:** Move diagnostic/utility scripts to a top-level scripts/ or tools/ directory outside the importable package. They should never share a namespace with production modules.
### [HIGH] data-flow
**Locations:** main.py, token_logger.py, core

Three of the five files in the core module (main.py, token_logger.py, and one unnamed file) are not documented and their code is not provided. The architecture spec names the service an 'extraction and marking pipeline' but nothing in the provided material explains what data enters the pipeline, what schema it has, how extraction results are separated from marking/scoring results, or where outputs go. This is the primary business logic and it is a black box.

**Recommendation:** Document the end-to-end data flow: input schema (file type, payload structure), which module performs extraction vs. marking, what the output schema is, and where results are written (response body, database, pub/sub, etc.). At minimum, add docstrings and type annotations to main.py and token_logger.py.
### [MEDIUM] error-handling
**Locations:** check_models_script.py, list_models.py, model_loader.py, main.py

Error handling strategy is inconsistent and unspecified across the module. check_models_script.py calls exit(1) on missing config (process-killing, inappropriate in a long-running service). list_models.py catches a broad Exception and prints it. model_loader.py's error handling is unknown. For a FastAPI service, none of these patterns translate correctly — unhandled exceptions in request handlers produce 500s with no structured error response.

**Recommendation:** Define a service-wide error handling policy: startup config errors should raise immediately with a descriptive message (not exit()); model invocation errors should raise typed exceptions (e.g., ModelUnavailableError) that a FastAPI exception handler converts to structured JSON responses with appropriate HTTP status codes. Document this in a CONTRIBUTING or ARCHITECTURE doc.
### [HIGH] security
**Locations:** main.py, core

There is no documented authorization model for the FastAPI service. The architecture spec does not mention authentication middleware, API key validation, IAM constraints, or any access control mechanism. For a service that proxies generative AI calls (which have billing implications and may process sensitive student/exam content given the 'marking' use case), unauthenticated public access would be a serious risk.

**Recommendation:** Define and implement an authorization strategy before production deployment: at minimum, require a service-account token or a gateway-enforced API key on all endpoints. Document the trust boundary (e.g., 'only callable from internal VPC' or 'requires Google IAP token') in the architecture spec.
### [LOW] coupling
**Locations:** check_models_script.py

check_models_script.py ends with the comment '# List available models is not directly supported', indicating the script was written to explore Vertex AI model listing, found it unsupported, and was abandoned mid-implementation. It performs only initialization (vertexai.init) and then does nothing, but remains in the core module as if it were complete.

**Recommendation:** Either complete the intended functionality or delete the file. If it is a diagnostic reference, move it to scripts/ with a clear README explaining what it was attempting and why it is incomplete. Dead investigative code in a production module misleads future engineers about the module's capabilities.
### [LOW] naming
**Locations:** core, check_models_script.py, list_models.py, model_loader.py

The service is named AI_Extraction_Marking_service, but the module spec summary describes it as a 'multimodal content-extraction service' with no mention of marking/scoring. The concept of 'marking' (grading) is referenced in the overview but does not map to any named module, class, or file in the provided code. It is unclear whether marking is a feature that exists, is planned, or was removed.

**Recommendation:** Align the service name, architecture overview, and module responsibilities. If marking is a distinct concern from extraction, it should be reflected in at least one identifiable file or class. If it is part of the undocumented three files, document it explicitly.
