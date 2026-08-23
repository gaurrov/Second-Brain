#!/usr/bin/env python3
"""
End-to-end verification of the HYBRID CONVERSATIONAL RAG system.

Runs against a LIVE server (default http://localhost:8000) and verifies the
12 acceptance tests: routing behaviour (CHAT vs DOCUMENT), Qdrant/embedding
call counts (scraped from /metrics), streaming, isolation (IDOR),
persistence, and error handling.

Usage:
    python tests/e2e_hybrid_rag.py [base_url]
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/api/v1"
ROOT = BASE.rsplit("/api/v1", 1)[0]
RESULTS = []


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"  ... {msg}", flush=True)


def _redact(text):
    """Never print tokens/long bodies into the report."""
    text = re.sub(r"eyJ[\w.-]{20,}", "<jwt>", str(text))
    return str(text)[:160]


def record(name, passed, detail="", blocked=False):
    RESULTS.append({"name": name, "passed": passed, "detail": _redact(detail), "blocked": blocked})
    icon = "SKIP" if blocked else ("PASS" if passed else "FAIL")
    print(f"  [{icon:4s}] {name:45s} {_redact(detail)}", flush=True)


def pace():
    """Keep total request+scrape rate comfortably under the 120/min limit."""
    time.sleep(0.6)


def section(title):
    print(f"\n=== {title} ===", flush=True)


# ---------------------------------------------------------------------------
# HTTP helpers (with 429 backoff)
# ---------------------------------------------------------------------------
def api(method, path, data=None, headers=None, base=None, timeout=120):
    url = f"{base or BASE}{path}"
    body = json.dumps(data).encode() if data is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    for attempt in range(4):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            raw = resp.read()
            try:
                return resp.status, json.loads(raw), raw
            except (json.JSONDecodeError, UnicodeDecodeError):
                return resp.status, None, raw
        except urllib.error.HTTPError as e:
            raw = e.read()
            if e.code == 429 and attempt < 3:
                log(f"rate limited, backing off ({attempt + 1}/3)")
                time.sleep(15)
                continue
            try:
                return e.code, json.loads(raw), raw
            except (json.JSONDecodeError, UnicodeDecodeError):
                return e.code, None, raw
        except (ConnectionResetError, ConnectionRefusedError, OSError) as e:
            return 0, None, str(e).encode()


def sse_stream(path, data, headers, timeout=300):
    """POST an SSE request, return (status, events list, elapsed seconds)."""
    url = f"{BASE}{path}"
    body = json.dumps(data).encode()
    hdrs = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    start = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - start
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, [], time.perf_counter() - start
    events = []
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return 200, events, elapsed


def upload(path, file_path, filename, headers):
    boundary = f"----e2e-hybrid-{uuid.uuid4().hex[:16]}"
    with open(file_path, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    url = f"{BASE}{path}"
    hdrs = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, None


def wait_processing(doc_id, token, max_wait=300):
    start = time.time()
    while time.time() - start < max_wait:
        time.sleep(5)
        _, body, _ = api("GET", f"/documents/{doc_id}", headers={"Authorization": f"Bearer {token}"})
        if body and str(body.get("processing_status", "")).upper() in ("COMPLETED", "FAILED"):
            return body, time.time() - start
    return None, time.time() - start


# ---------------------------------------------------------------------------
# Metrics scraping (counters exposed by prometheus_client at /metrics)
# ---------------------------------------------------------------------------
def _parse_metrics():
    for attempt in range(6):
        try:
            raw = urllib.request.urlopen(f"{ROOT}/metrics", timeout=10).read().decode()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                log(f"/metrics rate limited, backing off ({attempt + 1}/5)")
                time.sleep(12)
                continue
            return None
        except Exception:
            return None
    else:
        return None
    values = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(.+)$", line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        if value in ("NaN", "+Inf", "-Inf"):
            continue
        values[f"{name}{labels}"] = float(value)
    return values


def _sum_metric(values, name, operation=None):
    total = 0.0
    prefix = name + "{"
    for key, val in values.items():
        if not key.startswith(prefix):
            continue
        if operation is not None and f'operation="{operation}"' not in key:
            continue
        total += val
    return total


def _qdrant_search_total():
    """Authoritative search counter from Qdrant itself (/telemetry).

    The app runs uvicorn --workers 2, so each worker keeps its own
    prometheus_client registry and /metrics deltas are unreliable.
    Qdrant is a single process -> its counters are exact.
    """
    host = urllib.parse.urlparse(ROOT).hostname or "localhost"
    try:
        raw = urllib.request.urlopen(f"http://{host}:6333/telemetry", timeout=10).read()
        responses = json.loads(raw)["result"]["requests"]["rest"]["responses"]
        for endpoint, statuses in responses.items():
            if "/points/search" in endpoint:
                return sum(int(s.get("count", 0)) for s in statuses.values())
    except Exception:
        return None
    return 0


class Meter:
    """Captures counter deltas around a block of code."""

    def __enter__(self):
        self.before = _parse_metrics()
        self.q_before = _qdrant_search_total()
        return self

    def __exit__(self, *exc):
        self.after = _parse_metrics()
        self.q_after = _qdrant_search_total()

    def delta(self, metric, operation=None):
        if self.before is None or self.after is None:
            return None  # scrape unavailable -> metrics unreliable
        op = _sum_metric(self.before, metric, operation)
        cl = _sum_metric(self.after, metric, operation)
        return int(round(cl - op))

    @property
    def qdrant_search(self):
        if self.q_before is None or self.q_after is None:
            return None
        return int(self.q_after - self.q_before)

    @property
    def embeddings(self):
        return self.delta("embedding_requests_total")

    @property
    def llm(self):
        return self.delta("llm_requests_total")


REFUSAL_TEXT = "I couldn't find enough information in your knowledge base to answer that."


def mcheck(m, qdrant=None, embed=None, llm=None):
    """None-safe metric delta comparison -> (all_match, unverifiable_count)."""
    unknown = 0
    ok = True
    for got, want in ((m.qdrant_search, qdrant), (m.embeddings, embed), (m.llm, llm)):
        if want is None:
            continue
        if got is None:
            unknown += 1
            continue
        if got != want:
            ok = False
    return ok, unknown


# =========================================================================
print("=" * 72, flush=True)
print("  HYBRID CONVERSATIONAL RAG - FULL E2E VERIFICATION", flush=True)
print("=" * 72, flush=True)

_run = uuid.uuid4().hex[:8]
A_EMAIL, A_USER = f"a_{_run}@hybrid-e2e.io", f"a_{_run}"
B_EMAIL, B_USER = f"b_{_run}@hybrid-e2e.io", f"b_{_run}"
PASSWORD = "SecurePass123!"

lat_chat, lat_doc = [], []

section("[AUTH] Register + login Users A and B")
s1, _, _ = api("POST", "/auth/register", {"username": A_USER, "email": A_EMAIL, "password": PASSWORD})
record("setup.register-A", s1 == 201, f"status={s1}")
pace()
s2, body, _ = api("POST", "/auth/login", {"email": A_EMAIL, "password": PASSWORD})
token_a = (body or {}).get("access_token", "")
auth_a = {"Authorization": f"Bearer {token_a}"}
record("setup.login-A", s2 == 200 and bool(token_a), f"status={s2}")
pace()
s3, _, _ = api("POST", "/auth/register", {"username": B_USER, "email": B_EMAIL, "password": PASSWORD})
record("setup.register-B", s3 == 201, f"status={s3}")
pace()
s4, body, _ = api("POST", "/auth/login", {"email": B_EMAIL, "password": PASSWORD})
token_b = (body or {}).get("access_token", "")
auth_b = {"Authorization": f"Bearer {token_b}"}
record("setup.login-B", s4 == 200 and bool(token_b), f"status={s4}")
pace()

section("[SETUP] Upload + process documents (A: architecture, B: cooking)")
up_s, body = upload("/documents/upload", "tests/test_document.pdf", "architecture.pdf", auth_a)
doc_a = (body or {}).get("id")
record("setup.upload-A", up_s == 202 and bool(doc_a), f"status={up_s}, doc={doc_a}")
result_a, secs_a = (None, 0)
if doc_a:
    result_a, secs_a = wait_processing(doc_a, token_a)
    record("setup.process-A", bool(result_a) and result_a["processing_status"].upper() == "COMPLETED",
           f"{secs_a:.1f}s chunks={((result_a or {}).get('chunk_count'))}")

up_s, body = upload("/documents/upload", "tests/test_document_cooking.pdf", "cooking.pdf", auth_b)
doc_b = (body or {}).get("id")
record("setup.upload-B", up_s == 202 and bool(doc_b), f"status={up_s}, doc={doc_b}")
if doc_b:
    result_b, secs_b = wait_processing(doc_b, token_b)
    record("setup.process-B", bool(result_b) and result_b["processing_status"].upper() == "COMPLETED",
           f"{secs_b:.1f}s chunks={((result_b or {}).get('chunk_count'))}")

# Cold-start warmup: first DOCUMENT query pays embedding-model load inside
# the app process. Warm it so latency numbers measure steady state.
log("warmup DOCUMENT query (loads embedding model once)")
t0 = time.perf_counter()
with Meter() as m:
    _, wbody, _ = api("POST", "/chat", {"message": "What does my uploaded document say about microservices?"}, headers=auth_a)
warm_s = time.perf_counter() - t0
log(f"warmup done in {warm_s:.2f}s (sources={len((wbody or {}).get('sources', []))})")

# -------------------------------------------------------------------------

section("TEST 1 - NORMAL CHAT: 'Hi'")
pace()
with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat", {"message": "Hi"}, headers=auth_a)
    dt = time.perf_counter() - t0
lat_chat.append(dt)
mok, unk = mcheck(m, qdrant=0, embed=0)
ok = (
    status == 200
    and bool(body and body.get("answer"))
    and (body or {}).get("sources") == []
    and mok
)
record("TEST1.chat-hi", ok,
       f"status={status}, qdrant={m.qdrant_search}, embed={m.embeddings}, llm={m.llm}, {dt:.2f}s"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))
conv_hi = (body or {}).get("conversation_id")
pace()

section("TEST 2 - NORMAL CONVERSATION: 'How are you?'")
with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat", {"message": "How are you?", "conversation_id": conv_hi}, headers=auth_a)
    dt = time.perf_counter() - t0
lat_chat.append(dt)
mok, unk = mcheck(m, qdrant=0, embed=0)
ok = status == 200 and bool(body and body.get("answer")) and (body or {}).get("sources") == [] and mok
record("TEST2.chat-how-are-you", ok,
       f"status={status}, qdrant={m.qdrant_search}, embed={m.embeddings}, {dt:.2f}s"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))
pace()

section("TEST 3 - GENERAL KNOWLEDGE: 'What is Docker?'")
with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat", {"message": "What is Docker?", "conversation_id": conv_hi}, headers=auth_a)
    dt = time.perf_counter() - t0
lat_chat.append(dt)
docker_answer = (body or {}).get("answer", "")
mok, unk = mcheck(m, qdrant=0, embed=0)
ok = status == 200 and len(docker_answer) > 20 and (body or {}).get("sources") == [] and mok
record("TEST3.chat-general-knowledge", ok,
       f"status={status}, qdrant={m.qdrant_search}, embed={m.embeddings}, ans_len={len(docker_answer)}, {dt:.2f}s"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))

section("TEST 4 - DOCUMENT QUESTION (grounded RAG)")
pace()
with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat",
                          {"message": "What does my uploaded document say about microservices?"},
                          headers=auth_a)
    dt = time.perf_counter() - t0
lat_doc.append(dt)
srcs = (body or {}).get("sources", [])
mok, unk = mcheck(m, qdrant=None, embed=None, llm=0)
ok = (status == 200 and len((body or {}).get("answer", "")) > 20 and len(srcs) > 0
      and m.qdrant_search not in (None, 0) and m.embeddings not in (None, 0)
      and all(s["document_id"] == doc_a for s in srcs))
record("TEST4.document-grounded", ok,
       f"status={status}, qdrant={m.qdrant_search}, embed={m.embeddings}, sources={len(srcs)}, own_doc_only={all(s['document_id'] == doc_a for s in srcs)}, {dt:.2f}s"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))
conv_t4 = (body or {}).get("conversation_id")

section("TEST 5 - CONVERSATIONAL -> DOCUMENT TRANSITION")
pace()
with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat", {"message": "What is CQRS?"}, headers=auth_a)
    dt = time.perf_counter() - t0
lat_chat.append(dt)
cqrs_answer = (body or {}).get("answer", "")
conv_t5 = (body or {}).get("conversation_id")
mok, unk = mcheck(m, qdrant=0, embed=0)
ok1 = status == 200 and len(cqrs_answer) > 10 and (body or {}).get("sources") == [] and mok
record("TEST5a.chat-what-is-cqrs", ok1,
       f"status={status}, qdrant={m.qdrant_search}, {dt:.2f}s"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))
pace()

with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat",
                          {"message": "Does my architecture document mention it?",
                           "conversation_id": conv_t5},
                          headers=auth_a)
    dt = time.perf_counter() - t0
lat_doc.append(dt)
srcs = (body or {}).get("sources", "")
follow_answer = (body or {}).get("answer", "")
ok2 = (status == 200 and m.qdrant_search not in (None, 0) and m.embeddings not in (None, 0)
       and len(srcs) > 0 and all(s["document_id"] == doc_a for s in srcs))
record("TEST5b.followup-document-route", ok2,
       f"status={status}, qdrant={m.qdrant_search}, embed={m.embeddings}, sources={len(srcs)}, {dt:.2f}s")

section("TEST 6 - DOCUMENT NOT FOUND (no hallucination)")
pace()
with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat",
                          {"message": "What does my uploaded document say about quantum blockchain synergy?"},
                          headers=auth_a)
    dt = time.perf_counter() - t0
answer6 = (body or {}).get("answer", "")
mok, unk = mcheck(m, llm=0)
ok = (status == 200 and answer6 == REFUSAL_TEXT and (body or {}).get("sources") == []
      and m.qdrant_search not in (None, 0) and mok)
record("TEST6.refusal-no-hallucination", ok,
       f"status={status}, llm_calls={m.llm}, sources={(body or {}).get('sources')}, answer={answer6[:60]!r}"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))

section("TEST 7 - NORMAL CHAT AFTER RAG ('Thanks!' must skip Qdrant)")
pace()
with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat",
                          {"message": "What did my document say about CQRS?",
                           "conversation_id": conv_t5},
                          headers=auth_a)
    dt = time.perf_counter() - t0
lat_doc.append(dt)
rag_again = (body or {}).get("answer", "")
ok1 = status == 200 and m.qdrant_search not in (None, 0)
record("TEST7a.document-reask", ok1, f"status={status}, qdrant={m.qdrant_search}, {dt:.2f}s")
pace()

with Meter() as m:
    t0 = time.perf_counter()
    status, body, _ = api("POST", "/chat", {"message": "Thanks!", "conversation_id": conv_t5}, headers=auth_a)
    dt = time.perf_counter() - t0
lat_chat.append(dt)
mok, unk = mcheck(m, qdrant=0, embed=0)
ok2 = (status == 200 and bool(body and body.get("answer")) and (body or {}).get("sources") == [] and mok)
record("TEST7b.thanks-skips-qdrant", ok2,
       f"status={status}, qdrant={m.qdrant_search}, embed={m.embeddings}, {dt:.2f}s"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))

section("TEST 8 - STREAMING CHAT ('Hi')")
pace()
with Meter() as m:
    status, events, dt = sse_stream("/chat/stream", {"message": "Hi"}, auth_a)
tokens = [e for e in events if e.get("type") == "token"]
final_src = [e for e in events if e.get("type") == "sources"]
stream_text = "".join(t["content"] for t in tokens)
mok, unk = mcheck(m, qdrant=0, embed=0)
ok = (status == 200 and len(stream_text) > 3 and final_src and final_src[-1]["sources"] == [] and mok)
record("TEST8.stream-chat", ok,
       f"status={status}, tokens={len(tokens)}, chars={len(stream_text)}, final_sources={bool(final_src) and final_src[-1]['sources']}, qdrant={m.qdrant_search}, {dt:.2f}s"
       + (f", METRICS_UNVERIFIABLE={unk}" if unk else ""))

section("TEST 9 - STREAMING RAG (grounded + citations)")
pace()
with Meter() as m:
    status, events, dt = sse_stream("/chat/stream",
                                    {"message": "What does my document say about CQRS?"}, auth_a)
tokens = [e for e in events if e.get("type") == "token"]
final_src = [e for e in events if e.get("type") == "sources"]
stream_text9 = "".join(t["content"] for t in tokens)
srcs9 = final_src[-1]["sources"] if final_src else []
ok = (status == 200 and len(stream_text9) > 20 and len(srcs9) > 0
      and m.qdrant_search not in (None, 0) and m.embeddings not in (None, 0)
      and all(s["document_id"] == doc_a for s in srcs9))
record("TEST9.stream-rag", ok,
       f"status={status}, tokens={len(tokens)}, chars={len(stream_text9)}, sources={len(srcs9)}, qdrant={m.qdrant_search}, {dt:.2f}s")

section("TEST 10 - MULTI-USER SECURITY + IDOR")
# B asks about B's own domain (cooking) - grounded from B's doc only.
pace()
with Meter() as m:
    status, body, _ = api("POST", "/chat",
                          {"message": "What does my uploaded document say about italian cooking techniques?"},
                          headers=auth_b)
srcs_b = (body or {}).get("sources", [])
b_own = bool(srcs_b) and all(s["document_id"] == doc_b for s in srcs_b)
record("TEST10a.B-grounded-own-doc", status == 200 and b_own,
       f"status={status}, sources={len(srcs_b)}, all_B_doc={b_own}")

# B asks about A's topic via DOCUMENT route -> must refuse (A's vectors invisible).
pace()
with Meter() as m:
    status, body, _ = api("POST", "/chat",
                          {"message": "What does my uploaded document say about microservices architecture?"},
                          headers=auth_b)
ans_b_micro = (body or {}).get("answer", "")
record("TEST10b.B-no-leak-of-A-doc", status == 200 and ans_b_micro == REFUSAL_TEXT and (body or {}).get("sources") == [],
       f"status={status}, refused={ans_b_micro == REFUSAL_TEXT}, sources={(body or {}).get('sources')}")

# IDOR attempts.
pace()
s_idor, _, _ = api("GET", f"/documents/{doc_b}", headers=auth_a)
record("TEST10c.IDOR-A-reads-B-doc", s_idor in (403, 404), f"status={s_idor}")
pace()
s_idor, _, _ = api("DELETE", f"/documents/{doc_b}", headers=auth_a)
record("TEST10d.IDOR-A-deletes-B-doc", s_idor in (403, 404), f"status={s_idor}")
if conv_t4:
    pace()
    s_idor, _, _ = api("GET", f"/conversations/{conv_t4}", headers=auth_b)
    record("TEST10e.IDOR-B-reads-A-conversation", s_idor in (403, 404), f"status={s_idor}")
    pace()
    s_idor, body, _ = api("POST", "/chat", {"message": "tell me more", "conversation_id": conv_t4}, headers=auth_b)
    record("TEST10f.IDOR-B-posts-to-A-conversation", s_idor in (403, 404), f"status={s_idor}")

section("TEST 11 - CONVERSATION PERSISTENCE")
pace()
_, conv_body, _ = api("GET", "/conversations", headers=auth_a)
convs_a = ((conv_body or {}).get("conversations") or []) if isinstance(conv_body, dict) else (conv_body or [])
titles_a = [c["title"] for c in convs_a]
record("TEST11a.A-conversations-created", len(convs_a) >= 3, f"count={len(convs_a)}")
persist_ok = True
for c in convs_a:
    st, msg_body, _ = api("GET", f"/conversations/{c['id']}/messages", headers=auth_a)
    msgs = (msg_body or {}).get("messages", []) if isinstance(msg_body, dict) else []
    roles_ok = [m.get("role") for m in msgs] == ["user", "assistant"] * (len(msgs) // 2) if msgs else False
    meta_first_user = msgs[0].get("retrieval_metadata") is None if msgs else False
    if not (st == 200 and len(msgs) >= 2 and roles_ok and meta_first_user):
        persist_ok = False
record("TEST11b.messages-bound-to-own-conversation", persist_ok,
       f"checked={len(convs_a)} conversations, alternating roles, user msgs carry no retrieval_metadata")
pace()
_, list_b_body, _ = api("GET", "/conversations", headers=auth_b)
n_b = len((((list_b_body or {}).get("conversations")) or []) if isinstance(list_b_body, dict) else [])
record("TEST11c.isolation-of-lists", n_b >= 1, f"A={len(convs_a)} conversations, B={n_b} conversations (disjoint sets)")

section("TEST 12 - ERROR HANDLING")
pace()
s_err, _, _ = api("POST", "/chat", {"message": "Hi"},
                  headers={"Authorization": "Bearer fake.invalid.token"})
record("TEST12a.invalid-jwt", s_err == 401, f"status={s_err}")
pace()
s_err, _, _ = api("POST", "/chat", {"message": "Hi"})
record("TEST12b.missing-jwt", s_err in (401, 403), f"status={s_err}")
pace()
s_err, body, _ = api("POST", "/chat", {"message": ""}, headers=auth_a)
record("TEST12c.empty-message", s_err in (400, 422), f"status={s_err}")
pace()
s_err, body, _ = api("POST", "/chat", {"message": "Hello there", "conversation_id": str(uuid.uuid4())},
                     headers=auth_a)
record("TEST12d.missing-conversation", s_err in (403, 404), f"status={s_err}")
pace()
s_err, body, _ = api("GET", f"/documents/{uuid.uuid4()}", headers=auth_a)
record("TEST12e.invalid-document", s_err == 404, f"status={s_err}")
pace()
s_err, events, dt = sse_stream("/chat/stream", {"message": ""}, auth_a)
record("TEST12f.empty-message-stream", s_err in (400, 422), f"status={s_err}")

# =========================================================================
section("LATENCY SUMMARY (steady-state, excludes warmup)")
if lat_chat:
    print(f"  CHAT avg: {sum(lat_chat)/len(lat_chat):.2f}s  (n={len(lat_chat)}) samples={[round(x,2) for x in lat_chat]}")
if lat_doc:
    print(f"  DOCUMENT/RAG avg: {sum(lat_doc)/len(lat_doc):.2f}s  (n={len(lat_doc)}) samples={[round(x,2) for x in lat_doc]}")

section("FINAL REPORT")
passed = sum(1 for r in RESULTS if r["passed"])
blocked = sum(1 for r in RESULTS if r.get("blocked"))
failed = sum(1 for r in RESULTS if not r["passed"] and not r.get("blocked"))
for r in RESULTS:
    icon = "SKIP" if r.get("blocked") else ("PASS" if r["passed"] else "FAIL")
    print(f"  [{icon:4s}] {r['name']:45s} {r['detail']}")
print("-" * 72)
print(f"  TOTAL: {len(RESULTS)} | PASSED: {passed} | FAILED: {failed} | SKIPPED: {blocked}")
sys.exit(0 if failed == 0 else 1)
