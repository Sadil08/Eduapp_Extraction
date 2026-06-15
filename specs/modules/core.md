---
module: core
affects: []
files: ['check_models_script.py', 'list_models.py', 'main.py', 'model_loader.py', 'token_logger.py']
---

# Core Module

## Purpose
This module provides a FastAPI-based multimodal content-extraction service backed by Google Vertex AI generative models. It manages model lifecycle via a singleton loader, routes image+text requests through subject-specific prompts, and records per-request token consumption and cost to monthly CSV logs.

## Public interfaces
- **`ModelLoader` (model_loader.py)** — Singleton class; `get_instance()` returns the shared instance, `load_model()` lazily initialises Vertex AI and returns the `GenerativeModel`.
- **`TokenLogger` (token_logger.py)** — Utility class that writes token usage records to monthly CSV files under `logs/`; exposed as a module-level `token_logger` singleton imported by `main.py`.
- **`SUBJECT_PROMPTS` (main.py)** — `dict` mapping subject names (`"Mathematics"`, `"Physics"`, …) to structured LaTeX-aware extraction prompt strings consumed when building model requests.
- **FastAPI application (main.py)** — HTTP API accepting `UploadFile` image files and `Form` fields; constructs `Part`-based multimodal payloads and returns extracted text.

## Data flow
- An HTTP request carrying an image (`UploadFile`) and optional subject string enters the FastAPI app.
- The subject key selects a prompt from `SUBJECT_PROMPTS`; the image bytes are decoded (via `PIL.Image` / `base64`) and wrapped in a Vertex AI `Part`.
- `ModelLoader.get_instance().load_model()` supplies the `GenerativeModel`; the combined prompt + image part is sent for inference.
- The response text is returned to the caller; token counts from the response are passed to `token_logger` for CSV persistence.

## Architecture principles
- **Singleton model loader** — `ModelLoader._instance` and `ModelLoader._model` are class-level; the model is constructed only once per process lifetime to avoid repeated Vertex AI init overhead.
- **Mandatory project config** — `load_model()` raises `ValueError` if `GCP_PROJECT_ID` is unset; `GCP_LOCATION` defaults to `"asia-south1"`.
- **Monthly cost accounting** — `TokenLogger` hard-codes Gemini 1.5 Flash rates (`INPUT_COST_PER_MILLION = 0.075`, `OUTPUT_COST_PER_MILLION = 0.30`) and partitions logs by year/month (`token_usage_YYYY_MM.csv`), creating the file with headers on first use.
- **Subject-driven prompting** — Prompt selection is table-driven via `SUBJECT_PROMPTS`; adding a subject requires only a new dict entry, not new code paths.

## Dependencies
- **Internal:** `main.py` imports `ModelLoader` from `model_loader` and `token_logger` from `token_logger`.
- **External:** `vertexai` / `vertexai.generative_models` (Vertex AI SDK), `google.generativeai` (used in `list_models.py`), `fastapi`, `pydantic`, `uvicorn`, `Pillow` (`PIL`), `python-dotenv`.

## Known gaps / TODOs
- `model_loader.py` and `main.py` are truncated — the full `load_model()` implementation (credentials handling, model name selection) and all API endpoint signatures are unknown.
- `token_logger.py`'s `_ensure_log_file_exists()` is truncated; CSV column headers and the public logging method name/signature are not visible in the source.
- `list_models.py` contains a hardcoded API key (`AIzaSy…`) — a clear security gap that should be moved to environment config.
- Pricing constants reference Gemini 1.5 Flash, but `check_models_script.py` primarily targets `gemini-2.5-flash`; cost calculations will be incorrect if the latter model is used in production.