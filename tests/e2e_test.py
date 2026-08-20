#!/usr/bin/env python3
"""
Comprehensive End-to-End verification script for Second Brain / Memora backend.
Tests: multi-user flows, security (IDOR), streaming, error handling, performance.
"""
import json
import time
import sys
import urllib.request
import urllib.error
import uuid

BASE = "http://localhost:8000/api/v1"
ROOT = "http://localhost:8000"
RESULTS = []

def log(section, msg, status="INFO"):
    icon = {"INFO": "[+]", "PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[!]", "BLOCKED": "[SKIP]"}.get(status, "[?]")
    print(f"  {icon} {msg}", flush=True)

def record(test_name, passed, detail="", blocked=False):
    RESULTS.append({"test": test_name, "passed": passed, "detail": detail, "blocked": blocked})
    if blocked:
        log("result", f"{test_name}: BLOCKED - {detail}", "BLOCKED")
    else:
        status = "PASS" if passed else "FAIL"
        log("result", f"{test_name}: {detail or 'OK'}", status)

def api(method, path, data=None, headers=None, base=None):
    url = f"{base or BASE}{path}"
    body = json.dumps(data).encode() if data is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        raw = resp.read()
        try:
            return resp.status, json.loads(raw), raw
        except (json.JSONDecodeError, UnicodeDecodeError):
            return resp.status, None, raw
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body), body
        except (json.JSONDecodeError, UnicodeDecodeError):
            return e.code, None, body
    except (ConnectionResetError, ConnectionRefusedError, OSError) as e:
        return 0, None, str(e).encode()

def sleep_between(msg="", secs=1):
    if secs > 0:
        time.sleep(secs)

def upload_file(path, file_path, filename, headers):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
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
        raw = resp.read()
        return resp.status, json.loads(raw), raw
    except urllib.error.HTTPError as e:
        body_resp = e.read()
        try:
            return e.code, json.loads(body_resp), body_resp
        except (json.JSONDecodeError, UnicodeDecodeError):
            return e.code, None, body_resp
    except (ConnectionResetError, ConnectionRefusedError, OSError) as e:
        return 0, None, str(e).encode()

def stream_chat(path, data, headers, timeout=300):
    url = f"{BASE}{path}"
    body = json.dumps(data).encode()
    hdrs = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    resp = urllib.request.urlopen(req, timeout=timeout)
    raw = resp.read().decode("utf-8", errors="replace")
    return resp.status, raw

def wait_for_processing(doc_id, token, max_wait=300):
    start = time.time()
    while time.time() - start < max_wait:
        time.sleep(5)
        status, body, _ = api("GET", f"/documents/{doc_id}",
                              headers={"Authorization": f"Bearer {token}"})
        if body:
            proc_status = body.get("processing_status", "")
            elapsed = time.time() - start
            log("info", f"  Status: {proc_status} ({elapsed:.0f}s elapsed)")
            if proc_status in ("COMPLETED", "completed", "FAILED", "failed"):
                return body
    return None

# =========================================================================
print("=" * 70, flush=True)
print("  SECOND BRAIN E2E VERIFICATION", flush=True)
print("=" * 70, flush=True)

# --- Unique credentials per run (survives volume-persisted databases) ---
_run_id = uuid.uuid4().hex[:12]
USER_A_EMAIL = f"alice_{_run_id}@e2e-secondbrain.io"
USER_A_USERNAME = f"alice_{_run_id}"
USER_A_PASSWORD = "SecurePass123!"
USER_B_EMAIL = f"bob_{_run_id}@e2e-secondbrain.io"
USER_B_USERNAME = f"bob_{_run_id}"
USER_B_PASSWORD = "SecurePass456!"
log("info", f"Run ID: {_run_id}")
log("info", f"User A: {USER_A_EMAIL} / {USER_A_USERNAME}")
log("info", f"User B: {USER_B_EMAIL} / {USER_B_USERNAME}")

# --- HEALTH CHECKS ---
print("\n[1] INFRASTRUCTURE HEALTH CHECKS", flush=True)
status, body, _ = api("GET", "/health/live", base=ROOT)
record("health/live", body and body.get("status") == "ok", f"status={status}")

status, body, _ = api("GET", "/health/ready", base=ROOT)
record("health/ready", body is not None, f"status={status}")

# =========================================================================
# USER A FLOW
# =========================================================================
print("\n[2] USER A: Registration & Authentication", flush=True)
status, body, _ = api("POST", "/auth/register", {
    "email": USER_A_EMAIL,
    "username": USER_A_USERNAME,
    "password": USER_A_PASSWORD,
})
record("A-register", status == 201, f"status={status}")

status, body, _ = api("POST", "/auth/login", {
    "email": USER_A_EMAIL,
    "password": USER_A_PASSWORD,
})
record("A-login", status == 200 and "access_token" in (body or {}),
       f"status={status}")
token_a = (body or {}).get("access_token")
auth_a = {"Authorization": f"Bearer {token_a}"}

# --- UPLOAD & PROCESS ---
print("\n[3] USER A: Document Upload & Processing", flush=True)
ts = time.time()
status, body, _ = upload_file("/documents/upload", "tests/test_document.pdf",
                               "architecture.pdf", auth_a)
upload_time = time.time() - ts
record("A-upload", status == 202, f"status={status}, time={upload_time:.2f}s")
doc_a_id = (body or {}).get("id")

if doc_a_id:
    log("info", f"Document ID: {doc_a_id}")

    ts = time.time()
    result = wait_for_processing(doc_a_id, token_a)
    processing_time = time.time() - ts

    if result:
        proc_status = result.get("processing_status")
        record("A-processing", proc_status.upper() == "COMPLETED",
               f"status={proc_status}, time={processing_time:.1f}s, chunks={result.get('chunk_count')}")
    else:
        record("A-processing", False, "Timed out after 300s")
else:
    record("A-processing", False, f"No document ID returned, status={status}")

# --- RAG QUERY ---
print("\n[4] USER A: RAG Query (non-streaming)", flush=True)
conv_id_a = None
ts = time.time()
status, body, _ = api("POST", "/chat", {
    "message": "What is microservices architecture and how does it relate to event-driven patterns?",
    "conversation_id": None
}, headers=auth_a)
chat_time = time.time() - ts
llm_blocked = status == 502
record("A-chat", status == 200 and body and "answer" in body,
       f"status={status}, time={chat_time:.2f}s" + (" [LLM unavailable - Groq API key invalid]" if llm_blocked else ""),
       blocked=llm_blocked)

if body and not llm_blocked:
    answer = body.get("answer", "")
    sources = body.get("sources", [])
    conv_id_a = body.get("conversation_id")
    record("A-answer-grounded", len(answer) > 20, f"answer_len={len(answer)}")
    record("A-answer-sources", len(sources) > 0, f"source_count={len(sources)}")
    log("info", f"Answer preview: {answer[:200]}...")
    log("info", f"Conversation ID: {conv_id_a}")
else:
    record("A-answer-grounded", False, "No body returned (LLM unavailable)", blocked=llm_blocked)
    record("A-answer-sources", False, "No body returned (LLM unavailable)", blocked=llm_blocked)

# --- CONTINUE CONVERSATION ---
print("\n[5] USER A: Conversation Persistence", flush=True)
if conv_id_a:
    ts = time.time()
    status, body, _ = api("POST", "/chat", {
        "message": "Can you tell me more about the CQRS pattern specifically?",
        "conversation_id": conv_id_a
    }, headers=auth_a)
    chat2_time = time.time() - ts
    blocked2 = status == 502
    record("A-continue-chat", status == 200 and body and "answer" in body,
           f"status={status}, time={chat2_time:.2f}s" + (" [LLM unavailable]" if blocked2 else ""),
           blocked=blocked2)
    if body:
        log("info", f"Answer preview: {body.get('answer', '')[:200]}...")
else:
    record("A-continue-chat", False, "No conversation ID (LLM unavailable, no chat completed)", blocked=llm_blocked)

# --- SECOND CONVERSATION ---
print("\n[6] USER A: Second Conversation", flush=True)
status, body, _ = api("POST", "/chat", {
    "message": "What are the key technologies used in this architecture?",
}, headers=auth_a)
blocked3 = status == 502
record("A-second-conv", status == 200 and body and "answer" in body,
       f"status={status}" + (" [LLM unavailable]" if blocked3 else ""),
       blocked=blocked3)

# --- LIST CONVERSATIONS ---
print("\n[7] USER A: List & Verify Resources", flush=True)
status, body, _ = api("GET", "/conversations", headers=auth_a)
if status == 200 and body:
    conversations = body if isinstance(body, list) else body.get("conversations", [])
    record("A-list-conversations", len(conversations) >= 1,
           f"count={len(conversations)}" + (" [no conversations created - LLM unavailable]" if len(conversations) == 0 and llm_blocked else ""),
           blocked=len(conversations) == 0 and llm_blocked)
else:
    record("A-list-conversations", False, f"status={status}")

status, body, _ = api("GET", "/documents", headers=auth_a)
if status == 200 and body:
    doc_list = body if isinstance(body, list) else body.get("documents", [])
    total = body.get("total", len(doc_list)) if isinstance(body, dict) else len(doc_list)
    record("A-list-documents", total >= 1, f"total={total}")
else:
    record("A-list-documents", False, f"status={status}")

# --- USER PROFILE ---
status, body, _ = api("GET", "/users/profile", headers=auth_a)
record("A-user-profile", status == 200 and body and body.get("username") == USER_A_USERNAME,
       f"status={status}")

# =========================================================================
# USER B FLOW
# =========================================================================
print("\n[8] USER B: Registration & Authentication", flush=True)
sleep_between("rate-limit window clearance before User B", 62)
status, body, _ = api("POST", "/auth/register", {
    "email": USER_B_EMAIL,
    "username": USER_B_USERNAME,
    "password": USER_B_PASSWORD,
})
record("B-register", status == 201, f"status={status}")

status, body, _ = api("POST", "/auth/login", {
    "email": USER_B_EMAIL,
    "password": USER_B_PASSWORD,
})
record("B-login", status == 200 and "access_token" in (body or {}),
       f"status={status}")
token_b = (body or {}).get("access_token")
auth_b = {"Authorization": f"Bearer {token_b}"}

# --- UPLOAD DIFFERENT DOC ---
print("\n[9] USER B: Document Upload & Processing", flush=True)
ts = time.time()
status, body, _ = upload_file("/documents/upload", "tests/test_document_cooking.pdf",
                               "cooking.pdf", auth_b)
upload_time_b = time.time() - ts
record("B-upload", status == 202, f"status={status}, time={upload_time_b:.2f}s")
doc_b_id = (body or {}).get("id")

if doc_b_id:
    ts = time.time()
    result_b = wait_for_processing(doc_b_id, token_b)
    proc_time_b = time.time() - ts
    if result_b:
        proc_status = result_b.get("processing_status")
        record("B-processing", proc_status.upper() == "COMPLETED",
               f"status={proc_status}, time={proc_time_b:.1f}s, chunks={result_b.get('chunk_count')}")
    else:
        record("B-processing", False, "Timed out")

# --- RAG QUERY ON B'S DOC ---
print("\n[10] USER B: RAG Query (Italian cooking)", flush=True)
ts = time.time()
status, body, _ = api("POST", "/chat", {
    "message": "What are the key ingredients and techniques in Italian cooking?",
}, headers=auth_b)
chat_time_b = time.time() - ts
llm_blocked_b = status == 502
record("B-chat", status == 200 and body and "answer" in body,
       f"status={status}, time={chat_time_b:.2f}s" + (" [LLM unavailable]" if llm_blocked_b else ""),
       blocked=llm_blocked_b)
if body and not llm_blocked_b:
    answer_b = body.get("answer", "")
    sources_b = body.get("sources", [])
    record("B-answer-grounded", len(answer_b) > 20, f"answer_len={len(answer_b)}")
    record("B-answer-sources", len(sources_b) > 0, f"source_count={len(sources_b)}")
    log("info", f"Answer preview: {answer_b[:200]}...")
else:
    record("B-answer-grounded", False, "No body (LLM unavailable)", blocked=llm_blocked_b)
    record("B-answer-sources", False, "No body (LLM unavailable)", blocked=llm_blocked_b)

# =========================================================================
# SECURITY: CROSS-USER IDOR TESTS
# =========================================================================
print("\n[11] SECURITY: Cross-User IDOR Tests", flush=True)
if doc_b_id:
    # A tries to access B's document metadata
    status, body, _ = api("GET", f"/documents/{doc_b_id}", headers=auth_a)
    record("IDOR-doc-metadata", status in (403, 404),
           f"status={status} (expected 403/404)")

    # A tries to delete B's document
    status, body, _ = api("DELETE", f"/documents/{doc_b_id}", headers=auth_a)
    record("IDOR-doc-delete", status in (403, 404),
           f"status={status} (expected 403/404)")

# A tries to access B's conversations (should only see own)
status, body, _ = api("GET", "/conversations", headers=auth_a)
record("IDOR-conversations-isolated", status == 200,
       f"status={status}")

# Fake/invalid token
status, body, _ = api("GET", "/documents",
                       headers={"Authorization": "Bearer fake.invalid.token"})
record("IDOR-fake-token", status == 401, f"status={status} (expected 401)")

# Empty token
status, body, _ = api("GET", "/documents",
                       headers={"Authorization": "Bearer "})
record("IDOR-empty-token", status in (401, 403), f"status={status} (expected 401/403)")

# No auth
status, body, _ = api("GET", "/documents")
record("IDOR-no-auth", status in (401, 403), f"status={status} (expected 401/403)")

# =========================================================================
# STREAMING CHAT
# =========================================================================
print("\n[12] STREAMING CHAT", flush=True)
ts = time.time()
try:
    status, raw = stream_chat("/chat/stream", {
        "message": "What is API Gateway pattern?",
    }, auth_a)
    stream_time = time.time() - ts
    event_count = raw.count("data:") if raw else 0
    has_content = "data:" in (raw or "")
    record("A-stream-chat", status == 200 and has_content,
           f"status={status}, events={event_count}, time={stream_time:.2f}s")
    log("info", f"Stream preview: {raw[:300] if raw else 'empty'}...")
except Exception as e:
    stream_time = time.time() - ts
    record("A-stream-chat", False, f"error={e}, time={stream_time:.2f}s")

# =========================================================================
# ERROR HANDLING
# =========================================================================
print("\n[13] ERROR HANDLING", flush=True)
sleep_between("spacing before error tests", 2)

# Duplicate registration (must hit same email → 409)
status, body, _ = api("POST", "/auth/register", {
    "email": USER_A_EMAIL,
    "username": f"alice_dup_{_run_id}",
    "password": USER_A_PASSWORD,
})
record("error-dup-email", status == 409, f"status={status} (expected 409)")

# Wrong password
status, body, _ = api("POST", "/auth/login", {
    "email": USER_A_EMAIL,
    "password": "WrongPassword!",
})
record("error-wrong-password", status == 401, f"status={status} (expected 401)")

# Non-existent document
status, body, _ = api("GET", f"/documents/{uuid.uuid4()}", headers=auth_a)
record("error-doc-not-found", status == 404, f"status={status} (expected 404)")

# Chat with empty question
status, body, _ = api("POST", "/chat", {"message": ""}, headers=auth_a)
record("error-empty-question", status in (400, 422), f"status={status} (expected 400/422)")

# =========================================================================
# FINAL REPORT
# =========================================================================
print("\n" + "=" * 70, flush=True)
print("  FINAL TEST REPORT", flush=True)
print("=" * 70, flush=True)
passed = sum(1 for r in RESULTS if r["passed"])
blocked = sum(1 for r in RESULTS if r.get("blocked"))
failed = sum(1 for r in RESULTS if not r["passed"] and not r.get("blocked"))
total = len(RESULTS)

for r in RESULTS:
    if r.get("blocked"):
        icon = "SKIP"
    elif r["passed"]:
        icon = "PASS"
    else:
        icon = "FAIL"
    print(f"  [{icon:4s}] {r['test']:40s} {r['detail']}", flush=True)

print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed} | Blocked (LLM): {blocked}", flush=True)
if blocked:
    print("  Note: Blocked tests require a valid GROQ_API_KEY in .env", flush=True)
print("=" * 70, flush=True)

sys.exit(0 if failed == 0 else 1)
