#!/usr/bin/env python3
"""
gemini-web2api - Gemini Web to OpenAI API proxy.

Converts Google Gemini's web interface into an OpenAI-compatible API server.
Zero authentication required. Works on any platform (Windows/macOS/Linux).

Usage:
    pip install httpx
    python gemini_web2api.py [--port 8081] [--config config.json]

Client configuration (Cherry Studio, ChatBox, etc.):
    Base URL: http://localhost:8081/v1
    API Key: (anything or empty)

How it works:
    Sends requests directly to Gemini's public StreamGenerate endpoint.
    The backend does not verify authentication for basic text generation.
    Model selection via MODE_CATEGORY field [79] in the request payload.
    This is NOT a user-tier spoofing attack - the endpoint simply doesn't
    require auth for anonymous access.
"""
import json
import urllib.request
import urllib.parse
import time
import ssl
import sys
import uuid
import re
import os
import random
import threading
import hashlib
import argparse
import base64
import binascii
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

__version__ = "1.1.0"

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.6-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "temporary_chats": False,
    # Per-model upstream tickets (X-Goog-Ext-525001261-Jspb). The browser mints
    # one per model family and the server routes BY TICKET, ignoring the
    # f.req [79]/[80] fields when it is absent (falls back to account default).
    # Tickets carry embedded timestamps and expire; refresh by copying the
    # header value from a fresh browser StreamGenerate request
    # (DevTools -> Copy as cURL) into the matching key ("flash"/"pro"/...).
    # The proxy logs a routing-mismatch warning when a ticket stops working.
    "model_tickets": {
        "flash": '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,1,1,"561701FD-A2E8-4275-98B0-636DFE1554F9",null,null,[[6,908199999],[1789884088,624000000]]]',
        "pro": '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,3,1,"32C786FF-9AE2-49E5-A67C-0A35421F63A6",null,null,[[6,620699999],[1789897904,515000000]]]',
        "lite": '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,6,1,"32C786FF-9AE2-49E5-A67C-0A35421F63A6",null,null,[[null,95100000],[1789898801,131000000]]]',
        "flash-thinking": '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,1,2,"279B5F21-C196-4B10-8EC7-31625C0CABE6",null,null,[[null,332100000],[1789899397,281000000]]]',
        "lite-thinking": '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,6,2,"279B5F21-C196-4B10-8EC7-31625C0CABE6",null,null,[[2,950300000],[1789899759,320000000]]]',
        "pro-thinking": '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,3,2,"279B5F21-C196-4B10-8EC7-31625C0CABE6",null,null,[[null,85000000],[1789900208,389000000]]]',
    },
}

CONFIG = dict(DEFAULT_CONFIG)

# ─── Models ──────────────────────────────────────────────────────────────────
# Mapping from JS source: MODE_CATEGORY enum (028-6eb337387583.js)
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

# Model selection uses TWO fields in the f.req inner array (decoded from live
# browser captures, Sep 2026):
#   inner[79] = family: 1=flash, 3=pro, 4=auto, 5=dynamic-thinking, 6=flash-lite
#   inner[80] = variant: 1=standard, 2=extended/thinking
# HOWEVER: the server only honors these fields when the request also carries
# the per-model ticket header X-Goog-Ext-525001261-Jspb. Without the ticket
# the server falls back to the account default regardless of [79]/[80].
# The ticket wins over the body fields when both are present.

TICKET_HEADER = "X-Goog-Ext-525001261-Jspb"

MODELS = {
    "gemini-3.8-flash": {
        "mode": 1, "think": 4, "variant": 1, "ticket": "flash",
        "desc": "Alias for gemini-3.6-flash (point version is a label; same routing)",
    },
    "gemini-3.7-flash": {
        "mode": 1, "think": 4, "variant": 1, "ticket": "flash",
        "desc": "Alias for gemini-3.6-flash (point version is a label; same routing)",
    },
    "gemini-3.6-flash": {
        "mode": 1, "think": 4, "variant": 1, "ticket": "flash",
        "desc": "All-around model (Gemini 3.6 Flash)",
    },
    "gemini-3.6-flash-thinking": {
        "mode": 1, "think": 1, "variant": 2, "ticket": "flash-thinking",
        "desc": "Extended thinking on Flash (~20k chars)",
    },
    "gemini-3.5-flash-lite": {
        "mode": 6, "think": 4, "variant": 1, "ticket": "lite",
        "desc": "Cost-efficient high-capacity model (Gemini 3.5 Flash-Lite)",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5, "think": 1, "variant": 2, "ticket": "lite-thinking",
        "desc": "Extended thinking on Flash-Lite",
    },
    "gemini-3.1-pro": {
        "mode": 3, "think": 4, "variant": 1, "ticket": "pro",
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-3.1-pro-thinking": {
        "mode": 3, "think": 1, "variant": 2, "ticket": "pro-thinking",
        "desc": "Extended thinking on Pro",
    },
}


def ticket_for(model_name: str):
    """Return the upstream ticket header value for a model, or None."""
    key = (MODELS.get(model_name) or {}).get("ticket")
    if not key:
        return None
    return (CONFIG.get("model_tickets") or {}).get(key)


# Fence marker the model is instructed to wrap tool calls in (see prompt builders).
TOOL_CALL_MARKER = "```tool_call"


# ─── Utilities ───────────────────────────────────────────────────────────────

# Browser-like per-session RPC counter. The real Gemini web app sends an
# ever-increasing _reqid (+100000 per RPC) for every request in a page
# session. Deriving it from a timestamp made concurrent requests (e.g. a
# cronjob firing while the user is chatting) send identical ids.
_REQID_LOCK = threading.Lock()
_REQID_NEXT = random.randrange(10000, 99999)


def _next_reqid() -> int:
    global _REQID_NEXT
    with _REQID_LOCK:
        _REQID_NEXT += 100000
        return _REQID_NEXT


def log(msg: str):
    if CONFIG["log_requests"]:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def load_cookie() -> tuple:
    """Load cookie from file. Returns (cookie_str, sapisid)."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file:
        return "", None
    if not os.path.exists(cookie_file):
        return "", None
    try:
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
            # Cookie file may also carry auth_user / xsrf_token / gemini_bl.
            # Sync them into CONFIG, otherwise StreamGenerate goes out
            # without the `at` param and Gemini answers 400.
            if data.get("xsrf_token"):
                CONFIG["xsrf_token"] = data["xsrf_token"]
            if "auth_user" in data and data["auth_user"] not in (None, ""):
                CONFIG["auth_user"] = data["auth_user"]
            if data.get("gemini_bl"):
                CONFIG["gemini_bl"] = data["gemini_bl"]
        elif "# Netscape HTTP Cookie File" in content or content.startswith("#HttpOnly_"):
            # Netscape cookies.txt (tab-separated: domain, flag, path, secure, expiry, name, value)
            pairs = {}
            for line in content.splitlines():
                if line.startswith("#HttpOnly_"):
                    line = line[len("#HttpOnly_"):]
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 7:
                    continue
                pairs[parts[5]] = parts[6]
            cookie_str = "; ".join(f"{k}={v}" for k, v in pairs.items())
            sapisid = pairs.get("SAPISID", "")
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error: {e}")
        return "", None


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


def apply_chat_persistence_flags(inner: list) -> None:
    """Apply Gemini Web persistence flags to an outgoing request payload."""
    if CONFIG.get("temporary_chats", False):
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def extract_auth_from_html(html: str) -> tuple:
    """Extract (xsrf_token, gemini_bl) from Gemini app page HTML."""
    xsrf = None
    m = re.search(r'"SNlM0e"\s*:\s*"([^"]+)"', html)
    if m:
        raw = m.group(1)
        try:
            xsrf = raw.encode().decode("unicode_escape")
        except Exception:
            xsrf = raw
        xsrf = xsrf.replace("\\u003d", "=").replace("\\u0026", "&")
    bl = None
    b = re.search(r"(boq_assistant-bard-web-server_\d+\.\d+_p\d+)", html)
    if b:
        bl = b.group(1)
    return xsrf, bl


def persist_auth_to_file(xsrf, bl) -> None:
    """Write refreshed xsrf_token/gemini_bl back to the JSON cookie file."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return
    try:
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if not content.startswith("{"):
            return
        data = json.loads(content)
        if xsrf:
            data["xsrf_token"] = xsrf
        if bl:
            data["gemini_bl"] = bl
        with open(cookie_file, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except Exception as e:
        log(f"Auth persist failed: {e}")


def refresh_auth() -> bool:
    """Refresh xsrf_token/gemini_bl from the authenticated Gemini page.

    SNlM0e rotates every few minutes, so a statically exported token goes
    stale and Gemini answers 400 with an xsrf error. Re-fetch it with the
    cookie session and persist it back to the cookie file. Returns True if
    a usable token was obtained.
    """
    cookie_str, sapisid = load_cookie()
    if not cookie_str:
        return False
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cookie": cookie_str,
        }
        if sapisid:
            headers["Authorization"] = make_sapisidhash(sapisid)
        url = f"https://gemini.google.com{account_prefix()}/app"
        ctx = ssl.create_default_context()
        proxy = CONFIG.get("proxy")
        req = urllib.request.Request(url, headers=headers, method="GET")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx))
            resp = opener.open(req, timeout=30)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=30)
        html = resp.read().decode("utf-8", errors="replace")
        xsrf, bl = extract_auth_from_html(html)
        if xsrf and xsrf != CONFIG.get("xsrf_token"):
            CONFIG["xsrf_token"] = xsrf
        if bl and bl != CONFIG.get("gemini_bl"):
            CONFIG["gemini_bl"] = bl
        if xsrf:
            persist_auth_to_file(xsrf, bl)
            log("Auth refreshed from Gemini page")
            return True
        log("Auth refresh found no token")
        return False
    except Exception as e:
        log(f"Auth refresh failed: {e}")
        return False


def is_xsrf_error(e) -> bool:
    """Check whether an upstream error is a 400 xsrf rejection."""
    import urllib.error as _urlerr
    if isinstance(e, _urlerr.HTTPError) and e.code == 400:
        try:
            return "xsrf" in e.read().decode("utf-8", errors="replace")
        except Exception:
            return True
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 400:
        return True
    return False


def fetch_latest_bl() -> Optional[str]:
    """Fetch the latest gemini_bl (and xsrf token when cookies are set) from gemini.google.com page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        cookie_str, _ = load_cookie()
        if cookie_str:
            headers["Cookie"] = cookie_str
        req = urllib.request.Request("https://gemini.google.com/app", headers=headers)
        ctx = ssl.create_default_context()
        proxy = CONFIG.get("proxy")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx))
            resp = opener.open(req, timeout=15)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")
        xsrf, bl = extract_auth_from_html(html)
        if xsrf and xsrf != CONFIG.get("xsrf_token"):
            CONFIG["xsrf_token"] = xsrf
        if bl:
            return bl
    except Exception as e:
        log(f"BL auto-update fetch failed: {e}")
    return None


def update_bl_if_needed() -> bool:
    """Attempt to fetch and refresh gemini_bl + xsrf_token. Returns True if anything changed."""
    old_bl = CONFIG["gemini_bl"]
    old_xsrf = CONFIG.get("xsrf_token")
    new_bl = fetch_latest_bl()  # also refreshes xsrf_token when cookies are set
    if new_bl:
        CONFIG["gemini_bl"] = new_bl
    if CONFIG["gemini_bl"] != old_bl or CONFIG.get("xsrf_token") != old_xsrf:
        log(f"BL/XSRF refreshed: bl {old_bl} -> {CONFIG['gemini_bl']}, xsrf {'-> new' if CONFIG.get('xsrf_token') != old_xsrf else 'unchanged'}")
        return True
    return False


def upload_images(images: list) -> list:
    """Upload parsed OpenAI image parts and return Gemini file references."""
    if not images:
        return None
    from gemini_web2api.multimodal import detect_image_mime, fetch_image_bytes, upload_image

    file_refs = []
    for item in images:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        data, mime = item
        if isinstance(data, str):
            data = fetch_image_bytes(data)
            mime = mime or "image/png"
        if not data:
            raise RuntimeError("image fetch failed")
        mime = detect_image_mime(data, mime or "image/png")
        try:
            file_refs.append(upload_image(data, "image.png", mime or "image/png"))
        except Exception as e:
            raise RuntimeError(f"image upload failed: {e}") from e
    return file_refs if file_refs else None


# ─── Gemini Protocol ─────────────────────────────────────────────────────────

def gemini_stream_generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, ticket: str = None) -> str:
    """Send prompt to Gemini StreamGenerate with retry."""
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    log(f"Upstream model family={model_id} variant={(extra_fields or {}).get(80)}")

    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    body = urllib.parse.urlencode(params).encode()
    prefix = account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])

    if load_cookie()[0] and not CONFIG.get("xsrf_token"):
        refresh_auth()
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    if ticket:
        headers[TICKET_HEADER] = ticket

    def build_url() -> str:
        return (
            f"https://gemini.google.com{prefix}/_/BardChatUi/data/"
            "assistant.lamda.BardFrontendService/StreamGenerate"
            f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={_next_reqid()}&rt=c"
        )

    last_err = None
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            req = urllib.request.Request(build_url(), data=body, headers=headers, method="POST")
            ctx = ssl.create_default_context()
            proxy = CONFIG.get("proxy")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
            raw = resp.read().decode("utf-8", errors="replace")
            check_routing(raw, model_id, extra_fields, ticket)
            return raw
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RuntimeError("Gemini upstream rate-limited this IP (HTTP 429); retrying immediately would extend the block")
            if e.code in (400, 405) and update_bl_if_needed():
                log("Retrying with refreshed BL/XSRF...")
                last_err = e
                continue
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
        except Exception as e:
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def gemini_stream_generate_iter(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, ticket: str = None):
    """Send prompt and yield incremental text deltas using httpx streaming."""
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    log(f"Upstream model family={model_id} variant={(extra_fields or {}).get(80)}")

    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    body = urllib.parse.urlencode(params)
    prefix = account_prefix()
    url = (
        f"https://gemini.google.com{prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={_next_reqid()}&rt=c"
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    if load_cookie()[0] and not CONFIG.get("xsrf_token"):
        refresh_auth()
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    if ticket:
        headers[TICKET_HEADER] = ticket

    proxy = CONFIG.get("proxy")

    if not HAS_HTTPX:
        # Fallback: non-streaming with urllib
        raw = gemini_stream_generate(prompt, model_id, think_mode, file_refs)
        text = extract_response_text(raw)
        if text:
            yield text
        return

    prev_text = ""
    transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
    # Connection: close disables keep-alive reuse — the proxy/upstream kills
    # idle connections, and reusing a dead one fails mid-stream with
    # "[SSL] record layer failure", which corrupts streaming responses.
    with httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True,
                      headers={"Connection": "close"}) as client:
        try:
            with client.stream("POST", url, content=body, headers=headers) as resp:
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        import re as _re
                        m = _re.search(r'BardErrorInfo"?,?\s*\[(\d+)\]', buf)
                        if m:
                            code = int(m.group(1))
                            hint = {1060: "IP temporarily blocked or region not supported - use a proxy/different network or wait",
                                    1037: "usage limit exceeded",
                                    1013: "temporary upstream error, retry later",
                                    1185: "upstream rejected request"}.get(code, "upstream rejected request")
                            raise RuntimeError(f"Gemini upstream error [{code}]: {hint}")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        current = best_main_answer(line)
                        if not current:
                            continue
                        if current == prev_text or prev_text.startswith(current):
                            continue  # duplicate or older update
                        if not current.startswith(prev_text):
                            if prev_text:
                                log("Stream answer replaced mid-flight; keeping already-emitted text")
                                continue
                            prev_text = ""  # first frame was noise; adopt the real answer
                        delta = clean_gemini_text(current[len(prev_text):], strip=False)
                        prev_text = current
                        if delta:
                            yield delta
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", 0)
            if HAS_HTTPX and status in (400, 405):
                if update_bl_if_needed():
                    log("BL/XSRF refreshed, falling back to non-streaming for this request")
                    raw = gemini_stream_generate(prompt, model_id, think_mode, file_refs)
                    text = extract_response_text(raw)
                    if text:
                        yield text
                    return
            raise


def clean_gemini_text(text: str, strip: bool = True) -> str:
    """Remove internal code execution artifacts."""
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    return text.strip() if strip else text


def iter_frames(line: str):
    """Yield the parsed inner payload of every wrb.fr frame in one response line.

    A single StreamGenerate chunk line can carry several frames; the answer,
    thought summaries, alternative drafts, follow-up chips and image-agent
    updates each arrive as separate frames.
    """
    if '"wrb.fr"' not in line or len(line) < 40:
        return
    try:
        frames = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not (isinstance(frame, list) and len(frame) > 2 and frame[0] == "wrb.fr"):
            continue
        payload = frame[2]
        if not (isinstance(payload, str) and len(payload) >= 20):
            continue
        try:
            inner = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(inner, list) and len(inner) > 4 and inner[4]:
            yield inner


def candidate_texts(inner: list) -> list:
    """Return [(candidate_index, joined_text)] for one frame payload.

    Each candidate's text is a *list* of segments in the wire format and must
    be joined; treating segments as standalone answers returned fragments.
    """
    out = []
    for idx, cand in enumerate(inner[4]):
        if isinstance(cand, list) and len(cand) > 1 and isinstance(cand[1], list):
            text = "".join(t for t in cand[1] if isinstance(t, str))
            if text:
                out.append((idx, text))
    return out


def is_answer_frame(inner: list) -> bool:
    """True for the main answer frame, which carries the new conversation id
    (inner[1], e.g. "c_...") and response id (inner[2], e.g. "r_...").

    Thought summaries, follow-up chips, search and image-agent frames do not
    carry these ids -- that is what tells them apart from the real answer.
    """
    if len(inner) > 2:
        conv_id, resp_id = inner[1], inner[2]
        return (isinstance(conv_id, str) and bool(conv_id)
                and isinstance(resp_id, str) and bool(resp_id))
    return False


def best_main_answer(line: str):
    """Best primary-candidate answer text found in one line, or None."""
    best = ""
    for inner in iter_frames(line):
        if not is_answer_frame(inner):
            continue
        for idx, text in candidate_texts(inner):
            if idx == 0 and len(text) > len(best):
                best = text
    return best or None


def extract_response_text(raw: str) -> str:
    """Parse StreamGenerate response to extract the final answer text.

    Selection order:
      1. Primary candidate (index 0) of the main answer frame -- the frame
         carrying conversation/response ids. Longest update wins (streaming
         re-sends this frame as the answer grows).
      2. Longest joined candidate from any frame (protocol drift fallback).
    Never the thought/draft/chip frames, which previously won the "longest
    text anywhere" heuristic and produced unrelated answers.
    """
    import re as _re
    bard_err = _re.search(r'BardErrorInfo"?,?\s*\[(\d+)\]', raw)
    if bard_err:
        code = int(bard_err.group(1))
        hint = {1060: "IP temporarily blocked or region not supported - use a proxy/different network or wait",
                1037: "usage limit exceeded",
                1013: "temporary upstream error, retry later",
                1185: "upstream rejected request"}.get(code, "upstream rejected request")
        raise RuntimeError(f"Gemini upstream error [{code}]: {hint}")
    main_best = ""
    any_best = ""
    for line in raw.split("\n"):
        for inner in iter_frames(line):
            cands = candidate_texts(inner)
            if not cands:
                continue
            is_main = is_answer_frame(inner)
            for idx, text in cands:
                if len(text) > len(any_best):
                    any_best = text
                if is_main and idx == 0 and len(text) > len(main_best):
                    main_best = text
    return clean_gemini_text(main_best or any_best)


def upstream_echo(raw: str):
    """Return (label, family, variant) echoed by upstream, or None.

    The StreamGenerate response echoes the model that actually served the
    request; comparing it against the requested family/variant detects
    ignored model selection (e.g. expired model ticket).
    """
    for line in raw.split("\n"):
        if '"wrb.fr"' not in line or len(line) < 200:
            continue
        try:
            meta = json.loads(json.loads(line)[0][2])
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
        if isinstance(meta, list) and len(meta) >= 60:
            return meta[42], meta[58], meta[59]
    return None


def check_routing(raw: str, model_id: int, extra_fields: dict = None, ticket: str = None) -> None:
    """Log a warning when upstream served a different model than requested.

    When a ticket is used it wins over the body fields, so expectations are
    read from the ticket's embedded (family, variant).
    """
    echo = upstream_echo(raw)
    if not echo:
        return
    _, fam, var = echo
    if ticket:
        try:
            t = json.loads(ticket)
            want_fam, want_var = t[14], t[15]
        except (json.JSONDecodeError, IndexError, TypeError):
            want_fam, want_var = model_id, (extra_fields or {}).get(80)
    else:
        want_fam, want_var = model_id, (extra_fields or {}).get(80)
    if fam != want_fam or (want_var is not None and var != want_var):
        log(f"Routing mismatch: requested family={want_fam} variant={want_var} "
            f"but upstream served {echo[0]!r} (family={fam} variant={var}); "
            f"the model ticket in CONFIG['model_tickets'] may be expired — "
            f"refresh it from a fresh browser capture")


# ─── OpenAI Format Helpers ───────────────────────────────────────────────────

PROMPT_MAX_BYTES = 60000


def decode_data_url(url: str):
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url, re.DOTALL)
    if not match:
        return None
    mime = match.group(1) or "image/png"
    is_base64 = bool(match.group(2))
    data = match.group(3)
    try:
        if is_base64:
            return base64.b64decode(data, validate=True), mime
        return urllib.parse.unquote_to_bytes(data), mime
    except (ValueError, TypeError, binascii.Error):
        return None


def image_from_url(url: str, mime: str = None):
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        return decode_data_url(url)
    return url, mime or "image/png"


def image_from_part(part: dict):
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url", {})
        if isinstance(image_url, dict):
            return image_from_url(image_url.get("url"), image_url.get("mime_type"))
        return image_from_url(image_url)
    if part_type in ("input_image", "image"):
        image_url = part.get("image_url") or part.get("url")
        if isinstance(image_url, dict):
            return image_from_url(image_url.get("url"), image_url.get("mime_type"))
        if image_url:
            return image_from_url(image_url, part.get("mime_type"))
        image_data = part.get("data") or part.get("base64")
        if isinstance(image_data, str):
            mime = part.get("mime_type") or part.get("media_type") or "image/png"
            if image_data.startswith("data:"):
                return decode_data_url(image_data)
            try:
                return base64.b64decode(image_data, validate=True), mime
            except (ValueError, TypeError, binascii.Error):
                return None
    return None


def messages_to_prompt(messages: list, tools: list = None) -> tuple:
    """Convert OpenAI messages to (prompt_str, images_list)."""
    parts = []
    images = []
    if tools:
        tool_defs = []
        for tool in tools:
            fn = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_defs.append({
                "name": fn.get("name", tool.get("name", "")),
                "description": fn.get("description", tool.get("description", "")),
                "parameters": fn.get("parameters", tool.get("parameters", {})),
            })
        if tool_defs:
            tools_json = json.dumps(tool_defs, indent=2)
            if len(tools_json) > PROMPT_MAX_BYTES // 2:
                slim_defs = [{"name": t["name"], "description": t["description"]} for t in tool_defs]
                tools_json = json.dumps(slim_defs, indent=2)
                log(f"Tools block too large ({len(tool_defs)} tools), stripped parameters")
            parts.append(
                "[System instruction]: You have access to tools. "
                "To call a tool, respond with:\n"
                '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
                "Only use tool_call blocks when needed.\n\n"
                f"Available tools:\n{tools_json}"
            )
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if c.get("type") in ("text", "input_text", "output_text"):
                    text_parts.append(c.get("text", ""))
                else:
                    image = image_from_part(c)
                    if image:
                        images.append(image)
                        text_parts.append("[Image attached]")
            content = " ".join(text_parts)
        if role == "system":
            parts.append(f"[System instruction]: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc_strs = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tc_strs.append(
                        f'```tool_call\n{{"name": "{fn.get("name")}", '
                        f'"arguments": {fn.get("arguments", "{}")}}}\n```'
                    )
                parts.append(f"[Assistant]: {content or ''}\n" + "\n".join(tc_strs))
            else:
                parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            parts.append(f"[Tool result for {msg.get('name', '')}]: {content}")
        else:
            parts.append(content if content else "")
    return "\n\n".join(p for p in parts if p), images


def google_contents_to_prompt(req: dict) -> tuple:
    """Convert Google API contents to (prompt_str, images_list)."""
    parts = []
    images = []

    sys_inst = req.get("systemInstruction")
    if sys_inst:
        sys_text = " ".join(
            part.get("text", "") for part in sys_inst.get("parts", []) if part.get("text")
        )
        if sys_text:
            parts.append(f"[System instruction]: {sys_text}")

    for content in req.get("contents", []):
        role = content.get("role", "user")
        text_parts = []
        for part in content.get("parts", []):
            if part.get("text"):
                text_parts.append(part["text"])
            elif part.get("inlineData"):
                data = part["inlineData"]
                try:
                    images.append((
                        base64.b64decode(data["data"], validate=True),
                        data.get("mimeType", "image/png"),
                    ))
                    text_parts.append("[Image attached]")
                except (KeyError, ValueError, TypeError, binascii.Error):
                    pass
        text = " ".join(text_parts)
        if role == "model":
            parts.append(f"[Assistant]: {text}")
        else:
            parts.append(text)

    return "\n\n".join(part for part in parts if part), images


def tool_names(tools: list) -> set:
    """Extract declared function names from an OpenAI tools list."""
    names = set()
    for tool in tools or []:
        fn = tool.get("function", tool) if tool.get("type") == "function" else tool
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.add(name)
    return names


def _safe_json_loads(raw: str):
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None


def _coerce_tool_data(data):
    """Validate a parsed candidate as {"name": ..., "arguments": {...}}."""
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not name or not isinstance(name, str):
        return None
    args = data.get("arguments", data.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def _parse_bracket_args(raw: str):
    """Parse bracket-shorthand args, tolerating a trailing extra brace."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    if raw.rstrip().endswith("}"):
        try:
            return json.loads(raw.rstrip()[:-1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def parse_tool_calls(text: str, valid_names: set = None) -> tuple:
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list).

    Accepts the formats models emit in practice:
    1. ```tool_call\\n{"name": ..., "arguments": {...}}\\n``` (canonical)
    2. ```function_call\\n{...}\\n``` (common variant)
    3. ```json\\n{"name": ..., "arguments": {...}}\\n``` (bare JSON fence)
    4. [tool_call: name {...}] (bracket shorthand)
    5. Raw {"name": ..., "arguments"/"args": {...}} object

    Fences that do not parse as a tool call (e.g. a legit ```json code
    sample) are left untouched. When valid_names is given, calls to
    undeclared tools are dropped so clients don't choke on hallucinated
    tool names.
    """
    spans = []  # (start, end, {"name":..., "arguments":...})

    def _collect(pattern, group=1):
        for m in re.finditer(pattern, text, re.DOTALL):
            data = _coerce_tool_data(_safe_json_loads(m.group(group)))
            if data:
                spans.append((m.start(), m.end(), data))

    _collect(r'```tool_call\s*\n(.*?)\n```')
    _collect(r'```function_call\s*\n(.*?)\n```')
    _collect(r'```json\s*\n(.*?)\n```')

    for m in re.finditer(r'\[tool_call\s*:\s*([A-Za-z0-9_.\-]+)\s*(\{.*\})\s*\]',
                         text, re.DOTALL):
        args = _parse_bracket_args(m.group(2).strip())
        if args is not None:
            data = _coerce_tool_data({"name": m.group(1), "arguments": args})
            if data:
                spans.append((m.start(), m.end(), data))

    spans.sort()
    # Drop overlapping spans (keep the earliest match).
    merged = []
    for span in spans:
        if merged and span[0] < merged[-1][1]:
            continue
        merged.append(span)

    clean_parts = []
    last_end = 0
    tool_calls = []
    for start, end, data in merged:
        clean_parts.append(text[last_end:start])
        last_end = end
        if valid_names is not None and data["name"] not in valid_names:
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": data["name"],
                "arguments": json.dumps(data["arguments"], ensure_ascii=False),
            },
        })
    clean_parts.append(text[last_end:])

    if not tool_calls:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            data = _coerce_tool_data(_safe_json_loads(stripped))
            if data and (valid_names is None or data["name"] in valid_names):
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data["arguments"], ensure_ascii=False),
                    },
                })
                return "", tool_calls

    return "".join(clean_parts).strip(), tool_calls


# ─── HTTP Handler ────────────────────────────────────────────────────────────

class GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        client_ip = self.client_address[0] if self.client_address else "-"
        log(f"{client_ip} {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        keys = CONFIG.get("api_keys") or []
        if not keys:
            return True
        # Authorization: Bearer <key>
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] in keys:
            return True
        # header keys (OpenAI x-api-key / Google x-goog-api-key)
        for h in ("x-api-key", "x-goog-api-key"):
            if self.headers.get(h, "") in keys:
                return True
        # query param ?key= (Gemini CLI native style)
        if "?" in self.path:
            for pair in self.path.split("?", 1)[1].split("&"):
                if pair.startswith("key=") and pair[4:] in keys:
                    return True
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            if self.path == "/v1/models":
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in MODELS.items()
                ]})
            elif self.path.startswith("/v1beta/models"):
                self._handle_google_models_list()
            elif self.path == "/":
                self.send_json({"status": "ok", "version": __version__,
                                "models": list(MODELS.keys())})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"GET error: {e}")

    def do_POST(self):
        try:
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            body = self._read_request_body()
            if self.path == "/v1/chat/completions":
                self.handle_chat(body)
            elif self.path == "/v1/responses":
                self.handle_responses(body)
            elif ":streamGenerateContent" in self.path:
                self._handle_google_generate(body, stream=True)
            elif ":generateContent" in self.path:
                self._handle_google_generate(body, stream=False)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except:
                pass

    def _read_request_body(self) -> bytes:
        transfer_encoding = self.headers.get("Transfer-Encoding", "")
        if "chunked" in transfer_encoding.lower():
            chunks = []
            while True:
                size_line = self.rfile.readline()
                if not size_line:
                    break
                size_text = size_line.split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError:
                    raise ValueError("invalid chunked request body")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)

        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _resolve_model(self, model_name):
        think_override = None
        if "@think=" in model_name:
            model_name, think_str = model_name.rsplit("@think=", 1)
            think_override = int(think_str)
        cfg = MODELS.get(model_name)
        if not cfg:
            return None, None, None, f"Unknown model: {model_name}", None
        extra = dict(cfg.get("extra") or {})
        if "variant" in cfg and 80 not in extra:
            extra[80] = cfg["variant"]
        return model_name, cfg["mode"], (think_override if think_override is not None else cfg["think"]), None, extra or None

    def _call_gemini(self, prompt, model_id, think_mode, tools, file_refs=None, extra_fields=None, ticket=None):
        raw = gemini_stream_generate(prompt, model_id, think_mode, file_refs, extra_fields, ticket)
        text = extract_response_text(raw)
        tool_calls = None
        if tools and text:
            text, tool_calls = parse_tool_calls(text, tool_names(tools))
        return text or "", tool_calls

    def handle_chat(self, body: bytes):
        req = json.loads(body)
        model_name, model_id, think_mode, err, extra_fields = self._resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return
        ticket = ticket_for(model_name)

        tools = req.get("tools")
        prompt, images = messages_to_prompt(req.get("messages", []), tools)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty prompt"}}, 400)
            return

        stream = req.get("stream", False)
        log(f"Chat completions: stream={stream}, tools={len(tools) if tools else 0}, model={model_name}")
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        try:
            file_refs = upload_images(images)
        except RuntimeError as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        if stream and not tools:
            # True streaming: forward chunks as they arrive
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                first_chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                               "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(first_chunk)}\n\n".encode())
                for delta_text in gemini_stream_generate_iter(prompt, model_id, think_mode, file_refs, extra_fields, ticket):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                # Final chunk
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
                # Emit finish chunk so clients don't hang on a dropped stream
                try:
                    err_chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                 "model": model_name, "choices": [{"index": 0, "delta": {"content": f"[error] {e}"}, "finish_reason": "stop"}]}
                    self.wfile.write(f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            return

        if stream:
            # Tools present + streaming: forward prose deltas in real time,
            # but hold back ```tool_call fenced blocks (including a fence start
            # straddling delta boundaries) so they can be parsed into OpenAI
            # tool_calls at end of stream instead of leaking into chat text.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            first_chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(first_chunk)}\n\n".encode())
            self.wfile.flush()

            def send_delta(content=None, tool_calls=None, finish=None):
                delta = {}
                if content is not None:
                    delta["content"] = content
                if tool_calls:
                    delta["tool_calls"] = tool_calls
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            full_text = ""
            emitted = 0
            finish = "stop"
            try:
                for delta_text in gemini_stream_generate_iter(prompt, model_id, think_mode, file_refs, extra_fields, ticket):
                    full_text += delta_text
                    marker_pos = full_text.find(TOOL_CALL_MARKER)
                    # Without a marker, hold back the last len(marker)-1 chars so a
                    # fence start split across deltas is not forwarded prematurely.
                    limit = marker_pos if marker_pos != -1 else len(full_text) - len(TOOL_CALL_MARKER) + 1
                    if limit > emitted:
                        send_delta(content=full_text[emitted:limit])
                        emitted = limit
                clean, tool_calls = parse_tool_calls(full_text, tool_names(tools))
                if tool_calls:
                    log(f"Chat tool-fenced streaming: parsed {len(tool_calls)} tool call(s)")
                    if len(clean) > emitted:
                        send_delta(content=clean[emitted:])
                    for i, tc in enumerate(tool_calls):
                        send_delta(tool_calls=[{
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }])
                    finish = "tool_calls"
                else:
                    # No (or malformed) tool call: forward whatever is still
                    # buffered, raw, so nothing disappears silently.
                    if len(full_text) > emitted:
                        send_delta(content=full_text[emitted:])
                send_delta(finish=finish)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
                # Emit finish chunk so clients don't hang on a dropped stream
                try:
                    err_chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                 "model": model_name, "choices": [{"index": 0, "delta": {"content": f"[error] {e}"}, "finish_reason": "stop"}]}
                    self.wfile.write(f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            return

        # Non-streaming (or tool calling which needs full response)
        try:
            text, tool_calls = self._call_gemini(prompt, model_id, think_mode, tools, file_refs, extra_fields, ticket)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        self.send_json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": len(text)//4,
                          "total_tokens": (len(prompt)+len(text))//4},
            })

    def handle_responses(self, body: bytes):
        """OpenAI Responses API for Codex CLI compatibility."""
        req = json.loads(body)
        model_name, model_id, think_mode, err, extra_fields = self._resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return
        ticket = ticket_for(model_name)

        input_items = req.get("input", [])
        tools = req.get("tools")

        messages = []
        if req.get("instructions"):
            messages.append({"role": "system", "content": req["instructions"]})
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                         "name": item.get("name", ""), "content": item.get("output", "")})
                    elif item.get("type") in ("input_text", "input_image", "image"):
                        messages.append({"role": "user", "content": [item]})
                    elif item.get("role") == "assistant" or (item.get("type") == "message" and item.get("role") == "assistant"):
                        cp = item.get("content", [])
                        text_acc, tc_list = "", []
                        if isinstance(cp, list):
                            for c in cp:
                                if isinstance(c, dict):
                                    if c.get("type") == "output_text": text_acc += c.get("text", "")
                                    elif c.get("type") == "function_call": tc_list.append(c)
                        elif isinstance(cp, str):
                            text_acc = cp
                        m = {"role": "assistant", "content": text_acc or None}
                        if tc_list:
                            m["tool_calls"] = [{"id": tc.get("call_id", f"call_{i}"), "type": "function",
                                                "function": {"name": tc.get("name",""), "arguments": tc.get("arguments","{}")}}
                                               for i, tc in enumerate(tc_list)]
                        messages.append(m)
                    else:
                        role = item.get("role", "user")
                        messages.append({"role": role, "content": item.get("content", "")})

        if tools:
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                     if t.get("type") == "function" and "function" not in t else t for t in tools]

        prompt, images = messages_to_prompt(messages, tools)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty input"}}, 400)
            return

        try:
            file_refs = upload_images(images)
            text, tool_calls = self._call_gemini(prompt, model_id, think_mode, tools, file_refs, extra_fields, ticket)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        output = []
        if tool_calls:
            for tc in tool_calls:
                output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                               "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "status": "completed"})
        if text or not tool_calls:
            output.append({"type": "message", "id": mid, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": text or "", "annotations": []}]})

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            seq = [0]

            def emit(ev_type, **fields):
                seq[0] += 1
                ev = {"type": ev_type, "sequence_number": seq[0], **fields}
                self.wfile.write(f"event: {ev_type}\ndata: {json.dumps(ev)}\n\n".encode())

            usage = {"input_tokens": len(prompt)//4, "output_tokens": len(text)//4, "total_tokens": (len(prompt)+len(text))//4}
            base_resp = {"id": rid, "object": "response", "created_at": int(time.time()), "model": model_name}
            emit("response.created", response={**base_resp, "status": "in_progress", "output": [], "usage": None})
            emit("response.in_progress", response={**base_resp, "status": "in_progress", "output": [], "usage": None})
            for oi, item in enumerate(output):
                if item["type"] == "function_call":
                    pending = {"type": "function_call", "id": item["id"], "call_id": item["call_id"],
                               "name": item["name"], "arguments": "", "status": "in_progress"}
                    emit("response.output_item.added", output_index=oi, item=pending)
                    emit("response.function_call_arguments.delta", item_id=item["id"], output_index=oi, delta=item["arguments"])
                    emit("response.function_call_arguments.done", item_id=item["id"], output_index=oi, arguments=item["arguments"])
                    emit("response.output_item.done", output_index=oi, item=item)
                elif item["type"] == "message":
                    pending = {"type": "message", "id": item["id"], "role": "assistant", "status": "in_progress", "content": []}
                    emit("response.output_item.added", output_index=oi, item=pending)
                    for ci, cp in enumerate(item["content"]):
                        emit("response.content_part.added", item_id=item["id"], output_index=oi, content_index=ci,
                             part={"type": "output_text", "text": "", "annotations": []})
                        emit("response.output_text.delta", item_id=item["id"], output_index=oi, content_index=ci, delta=cp["text"])
                        emit("response.output_text.done", item_id=item["id"], output_index=oi, content_index=ci, text=cp["text"])
                        emit("response.content_part.done", item_id=item["id"], output_index=oi, content_index=ci, part=cp)
                    emit("response.output_item.done", output_index=oi, item=item)
            emit("response.completed", response={**base_resp, "status": "completed", "output": output, "usage": usage})
            self.wfile.flush()
        else:
            self.send_json({"id": rid, "object": "response", "created_at": int(time.time()), "status": "completed",
                            "model": model_name, "output": output,
                            "usage": {"input_tokens": len(prompt)//4, "output_tokens": len(text)//4, "total_tokens": (len(prompt)+len(text))//4}})


    # ─── Google Native API (Gemini CLI compatible) ────────────────────────────

    def _parse_google_model_from_path(self):
        """Extract model name from /v1beta/models/{model}:method path."""
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        if m:
            return m.group(1)
        return None

    def _handle_google_models_list(self):
        """GET /v1beta/models — Google AI format model list."""
        models = []
        for name, cfg in MODELS.items():
            models.append({
                "name": f"models/{name}",
                "displayName": name,
                "description": cfg["desc"],
                "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            })
        self.send_json({"models": models})

    def _handle_google_generate(self, body: bytes, stream: bool):
        """Handle Google native generateContent / streamGenerateContent."""
        req = json.loads(body)
        model_name = self._parse_google_model_from_path()
        if not model_name:
            self.send_json({"error": {"message": "model not specified in path"}}, 400)
            return

        model_name, model_id, think_mode, err, extra_fields = self._resolve_model(model_name)
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return
        ticket = ticket_for(model_name)

        prompt, images = google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty content"}}, 400)
            return

        try:
            file_refs = upload_images(images)
            text, _ = self._call_gemini(prompt, model_id, think_mode, None, file_refs, extra_fields, ticket)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        candidate = {
            "content": {"parts": [{"text": text or ""}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": len(prompt) // 4,
            "candidatesTokenCount": len(text) // 4,
            "totalTokenCount": (len(prompt) + len(text)) // 4,
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(response_obj)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json(response_obj)


# ─── Main ────────────────────────────────────────────────────────────────────

def load_config(path: str):
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
        log(f"Config loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Web to OpenAI API")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cookie-file", type=str, default=None, help="Path to cookie file")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--version", action="version", version=f"gemini-web2api {__version__}")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("GEMINI_WEB2API_CONFIG")
    if not config_path:
        for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
            if os.path.exists(p):
                config_path = p
                break
    load_config(config_path)

    if args.port:
        CONFIG["port"] = args.port
    if args.cookie_file:
        CONFIG["cookie_file"] = args.cookie_file
    if args.proxy:
        CONFIG["proxy"] = args.proxy

    new_bl = fetch_latest_bl()
    if new_bl:
        CONFIG["gemini_bl"] = new_bl

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    port = CONFIG["port"]
    server = ThreadedServer((CONFIG["host"], port), GeminiHandler)
    print(f"gemini-web2api v{__version__}")
    print(f"  Listening: http://0.0.0.0:{port}")
    print(f"  Base URL:  http://localhost:{port}/v1")
    print(f"  Models:    {', '.join(MODELS.keys())}")
    print(f"  Cookie:    {'yes (' + CONFIG['cookie_file'] + ')' if CONFIG.get('cookie_file') else 'none (anonymous)'}")
    print(f"  Proxy:     {CONFIG.get('proxy') or 'none (uses system env HTTP_PROXY/HTTPS_PROXY)'}")
    print(f"  Retry:     {CONFIG['retry_attempts']}x / {CONFIG['retry_delay_sec']}s")
    print(f"  BL:        {CONFIG['gemini_bl']}")
    print(f"  Temporary: {'yes' if CONFIG.get('temporary_chats', False) else 'no'}")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
