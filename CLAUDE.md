# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts Google Gemini's web interface into an OpenAI-compatible API by reverse-engineering the StreamGenerate endpoint. Pure Python stdlib with one optional dependency (`httpx`, for true streaming; falls back to `urllib` buffered mode without it). Python 3.8+ — do not use PEP 604 union annotations (`X | Y`) or other 3.10+ syntax.

`AGENTS.md` (added by upstream PR #100) documents the same architecture with the MIRROR RULE and DOC RULE — read it before making changes.

## Commands

```bash
# Run server (single-file version) — http://localhost:8081/v1
pip install httpx
python gemini_web2api.py [--port 8081] [--config config.json] [--cookie-file cookie.txt] [--proxy http://...]

# Run server (modular package version)
python -m gemini_web2api [same flags]

# Run all tests (stdlib unittest, no pytest; no network — upstream calls are mocked)
python -m unittest discover tests

# Run one test class / single test
python -m unittest tests.test_modular_sync.StreamingEndpointTests
python -m unittest tests.test_modular_sync.StreamingEndpointTests.test_chat_stream_starts_with_assistant_role

# Check whether the local IP is blocked by upstream (before debugging the proxy itself)
python probe_upstream.py

# Docker
docker compose up -d   # builds package version, mounts ./config.json
```

## Critical: Two Implementations, Manually Synced

The same server exists twice and must be kept in sync (MIRROR RULE in AGENTS.md):

- `gemini_web2api.py` — standalone single-file version (README's headline "single file" deliverable)
- `gemini_web2api/` — modular package (what pip installs, Docker builds, and tests exercise)

Tests only cover the package, so a package-only fix can silently break the single-file version — mirror every change. Known intentional divergences: single-file `messages_to_prompt` has no `tool_choice` param; single-file `_resolve_model` returns 400 on unknown model names while the package falls back to `default_model`; single-file has `fetch_latest_bl`/`update_bl_if_needed` (page fetch) where the package has `refresh_bl_and_xsrf`/`refresh_auth`.

`cloudflare/worker.js` is an independent JavaScript port (own model list, slot79-style routing) — not part of the mirror. `gemini-cookie-sync-extension/` is a Chrome MV3 extension (v2: background auto-sync to a native host); also independent.

## Architecture (package)

Request flow for every API surface:

```
client request → server.py (route + resolve_model + ticket_for)
               → tools.py    (flatten messages to ONE text prompt + collect images; tool schemas injected as prompt)
               → multimodal.py (upload images via Scotty, get file refs)
               → gemini.py   (payload + ticket header → POST to Gemini Web → parse strict frames → check_routing echo)
               → server.py   (map text back to API format, incl. fenced tool_calls)
```

Three API surfaces in `server.py` (`GeminiHandler`), all fed by the same core:
1. `/v1/chat/completions` — OpenAI Chat Completions
2. `/v1/responses` — OpenAI Responses API (Codex CLI), synthesizes the full SSE event sequence **after** collecting the complete upstream response (never streams from upstream)
3. `/v1beta/models/{model}:generateContent` / `:streamGenerateContent` — Google native API (Gemini CLI)

Module responsibilities:
- `config.py` — global mutable `CONFIG` dict (tests mutate and restore it in setUp/tearDown). `model_tickets` maps ticket keys → header values (they expire; refresh from a fresh browser capture). Discovery: `--config` → `GEMINI_WEB2API_CONFIG` env → `./config.json` → `~/.config/gemini-web2api/config.json`
- `models.py` — 6-model list keyed by (family, variant): `inner[79]`=family, `inner[80]`=variant. **Upstream only honors these fields when the `X-Goog-Ext-525001261-Jspb` ticket header is present** (Issue #82: without it everything routes to the account default; ticket wins when both present). `resolve_model()` parses `@think=N`; `ticket_for()` looks up the ticket
- `gemini.py` — protocol. Positionally-indexed payload; strict response parsing (`_iter_frames`/`_is_answer_frame`/`_candidate_texts`: only the main answer frame carrying conversation/response ids, candidate 0, joined segments); `upstream_echo`/`check_routing` verify routing from the response echo; `refresh_auth()` re-fetches SNlM0e + bl with the cookie session and persists to the cookie JSON file; `_reqid` is a thread-safe in-process counter (+100000)
- `tools.py` — prompt engineering. Tool calling simulated via prompts; `parse_tool_calls` tolerates 5 output formats (` ```tool_call `/` ```function_call `/` ```json ` fences, `[tool_call: ...]` shorthand, raw JSON) and drops calls to undeclared tools when `valid_names` is passed. Multi-turn history flattened into one prompt — each request is an independent Gemini conversation
- `multimodal.py` — Scotty resumable upload, page-token scraping with 10-min cache, magic-byte MIME sniffing

## Key Invariants

- **Zero required dependencies**: anything beyond stdlib must be optional (`HAS_HTTPX` pattern) and Python 3.8-compatible
- **Chat streaming works with AND without tools**: no-tools → forward deltas raw; with tools → fenced streaming (`TOOL_CALL_MARKER` buffering in server.py): prose streams live, fences are held back (including a marker straddling delta boundaries), parsed at end, emitted as OpenAI-spec `tool_calls` deltas with `index`. Malformed fences are forwarded raw, never silently dropped. `/v1/responses` still buffers the full upstream response
- **Stream retry safety**: deltas are only emitted when the new answer extends what was emitted; mid-stream rewrite freezes already-emitted text; once anything was emitted, interruptions end the stream cleanly (no retry); strict-parser misses fall back to a full rescan
- **Upstream errors**: HTTP 429 fails immediately (retrying extends the block); BardErrorInfo (tolerant JSPB regex) raises with code hints (1060 IP blocked / 1037 quota / 1013 transient / 1185 rejected) and is not retried except 1013; HTTP 400/405 triggers `refresh_bl_and_xsrf()` then an immediate retry (payload/headers rebuilt per attempt)
- **Cookies**: `load_cookie()` accepts JSON (`{cookie, sapisid, auth_user, xsrf_token, gemini_bl}` — extras sync into CONFIG), Netscape `cookies.txt`, or a raw string; cached by (mtime, size)
- Token usage in responses is an estimate (`len(str) // 4`)
- API-key auth accepts `Authorization: Bearer`, `x-api-key`, `x-goog-api-key`, or `?key=`; empty `api_keys` disables auth entirely

## Operational Gotchas

- `gemini_bl` (default in `config.py`) is a dated Google build tag — HTTP 405 means stale; SNlM0e XSRF rotates every few minutes, so static exports go stale → 400 xsrf (auto-refreshed when a cookie session is configured; startup fetch in `__main__`)
- Model tickets carry embedded timestamps and expire — `check_routing` logs a "Routing mismatch" warning when one stops working; refresh by copying the header value from a fresh browser StreamGenerate request (DevTools → Copy as cURL)
- `gemini-3.1-pro` only routes to real Pro with a Gemini Advanced (paid) cookie; otherwise silently falls back. The 3.x point version in a name is a label — routing is decided by (family, variant)
- Docker's default bridge network can cause empty responses from upstream — use `--network host`
- CI (`.github/workflows/docker.yml`) builds multi-arch images to ghcr.io on pushes to main and version tags; it builds the package, not the single file

## Upstream PR Merge Notes (Sep 2026)

This fork merged 8 PRs from `Sophomoresty/gemini-web2api` (upstream unmaintained): #92, #99, #91 (docs/cloudflare only), #87, #88, #100 (superset of #96 — skipped), #93, #95 (extension only). #80 (image gen via curl_cffi) was deliberately skipped to preserve the zero-dependency design. Conflict resolutions are documented in each merge commit message.
