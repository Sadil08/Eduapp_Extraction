# Pipeline failed

**Error:** Claude subscription usage limit reached. Wait for the 5-hour window to reset, then re-run — completed pipeline artifacts are reused on retry.
  CLI said: {"type":"result","subtype":"success","is_error":true,"api_error_status":429,"duration_ms":547,"duration_api_ms":0,"num_turns":1,"result":"You've hit your session limit · resets 3:50am (Asia/Colombo)","stop_reason":"stop_sequence","session_id":"a0fcfa3d-3233-41ab-9c0c-39b329ea68b2","total_cost_usd":0

**Run log:**

[18:43:52] Starting pipeline: issue #1
[18:43:52] Added extra context: 10574 chars
[18:43:52] Issue loaded: 'AI endpoint authentication and rate limiting'
[18:43:52] Classifying issue
[18:44:03] Classified: feature | critical | modules=['core/auth.py', 'core/config.py', 'main.py', 'core/list_models.py', 'tests']
[18:44:03] Searching spec index: api_key authentication rate_limiting X-API-Key
[18:44:03] Spec files selected: 3 files
[18:44:04] Source files via keyword grep (specs were thin): +3 file(s)
[18:44:04] Drafting bug report
[18:45:34] Starting judge loop: threshold=0.85
[18:46:03] Judge iteration 1: score=0.57 | 10 gap(s)
[18:53:48] Judge threshold not met — human review required: score=0.57 < 0.85
[18:53:48] Writing test plan
[18:54:57] Writing code despite low judge score (--force-code): score=0.57 — review the diff carefully
[18:54:57] Branch created: fix/issue-1-ai-endpoint-authentication-and-rate-limi (from development)
[18:54:57] Writing code fix (attempt 1): 3 source file(s)
[18:55:21] Code applied: 1 file(s): core/list_models.py
[18:55:21] Generating tests for the fix
[19:00:01] Tests generated: 1 test file(s)
[19:00:01] Running tests: pytest
[19:00:01] Tests failed: exit code 1
[19:00:01] Writing code fix (attempt 2): 3 source file(s)
[19:03:51] Code applied: 1 file(s): core/list_models.py
[19:03:51] Running tests: pytest
[19:03:52] Tests failed: exit code 1
[19:03:52] Writing code fix (attempt 3): 3 source file(s)
[19:03:54] PIPELINE FAILED: Claude subscription usage limit reached. Wait for the 5-hour window to reset, then re-run — completed pipeline artifacts are reused on retry.
  CLI said: {"type":"result","subtype":"success","is_error":true,"api_error_status":429,"duration_ms":547,"duration_api_ms":0,"num_turns":1,"result":"You've hit your session limit · resets 3:50am (Asia/Colombo)","stop_reason":"stop_sequence","session_id":"a0fcfa3d-3233-41ab-9c0c-39b329ea68b2","total_cost_usd":0