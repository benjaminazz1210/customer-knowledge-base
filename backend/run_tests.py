#!/usr/bin/env python3
"""
run_tests.py — NexusAI strict feature verification

This script executes all `type=auto` checks declared in feature_list.json and
writes pass/fail status back to that file.

Usage:
    cd backend
    python run_tests.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8001")
REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_LIST_PATH = REPO_ROOT / "feature_list.json"
FRONTEND_DIR = REPO_ROOT / "frontend"
FILES_PAGE_PATH = FRONTEND_DIR / "src" / "app" / "files" / "page.js"
CHAT_PAGE_PATH = FRONTEND_DIR / "src" / "app" / "page.js"
HEADER_PATH = FRONTEND_DIR / "src" / "components" / "Header.jsx"

TEST_FILENAME_TXT = None
TEST_FILENAME_MD = None
TEST_QUERY_TOKEN = None


def log_pass(feature_id: str, msg: str) -> None:
    print(f"  ✅ [{feature_id}] {msg}")


def log_fail(feature_id: str, msg: str) -> None:
    print(f"  ❌ [{feature_id}] {msg}")


def load_feature_list():
    with FEATURE_LIST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_feature_list(features) -> None:
    with FEATURE_LIST_PATH.open("w", encoding="utf-8") as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
        f.write("\n")


def _create_temp_file(suffix: str, content: str):
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        prefix="nexusai_test_",
        encoding="utf-8",
        delete=False,
    ) as f:
        f.write(content)
        return Path(f.name), Path(f.name).name


def _upload_file(path: Path, filename: str, mime_type: str = "text/plain"):
    with path.open("rb") as fh:
        return requests.post(
            f"{BASE_URL}/api/upload",
            files={"file": (filename, fh, mime_type)},
            timeout=300,
        )


def _get_files_list():
    resp = requests.get(f"{BASE_URL}/api/files", timeout=15)
    assert resp.status_code == 200, f"GET /api/files expected 200, got {resp.status_code}"
    data = resp.json()
    assert isinstance(data, list), f"/api/files should return list, got {type(data)}"
    return data


def _safe_delete_file(filename: str) -> None:
    if not filename:
        return
    try:
        requests.delete(f"{BASE_URL}/api/files/{filename}", timeout=10)
    except Exception:
        pass


def _safe_clear_history() -> None:
    try:
        requests.delete(f"{BASE_URL}/api/history", timeout=10)
    except Exception:
        pass


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_cmd(cmd, cwd: Path, timeout: int = 900) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-40:])
        stdout_tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise AssertionError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout (tail) ---\n{stdout_tail}\n"
            f"--- stderr (tail) ---\n{stderr_tail}"
        )


def test_infra_001():
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(f"{BASE_URL}/api/health", timeout=30)
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            body = resp.json()
            assert body.get("status") == "ok", f"Expected status='ok', got {body}"
            assert body.get("qdrant") is True, f"Expected qdrant=true, got {body}"
            return
        except Exception as err:
            last_err = err
            if attempt < 2:
                print(f"    ⏳ Retrying health check ({attempt + 1}/3)...")
                time.sleep(2)
    raise last_err


def test_infra_002():
    resp = requests.get(f"{BASE_URL}/", timeout=10)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    assert body.get("message") == "NexusAI API is running", f"Unexpected root response: {body}"


def test_infra_003():
    resp = requests.options(
        f"{BASE_URL}/api/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=10,
    )
    assert resp.status_code in (200, 204), f"Expected 200/204, got {resp.status_code}"
    allow_origin = resp.headers.get("access-control-allow-origin", "")
    assert allow_origin in ("*", "http://localhost:3000"), f"Unexpected allow-origin: {allow_origin}"
    allow_methods = resp.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods.upper(), f"POST missing in allow-methods: {allow_methods}"


def test_backend_001():
    global TEST_FILENAME_TXT, TEST_QUERY_TOKEN
    TEST_QUERY_TOKEN = f"nexusai-token-{uuid.uuid4().hex[:12]}"
    content = (
        "NexusAI strict upload test document\n\n"
        f"Unique lookup token: {TEST_QUERY_TOKEN}\n"
        "This document validates parsing, chunking, embedding, and vector persistence.\n"
    )
    tmp_path, filename = _create_temp_file(".txt", content)
    TEST_FILENAME_TXT = filename
    try:
        resp = _upload_file(tmp_path, filename, "text/plain")
    finally:
        tmp_path.unlink(missing_ok=True)

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("status") == "Ready", f"Expected status='Ready', got {body}"
    assert body.get("filename") == filename, f"Filename mismatch: {body}"
    assert body.get("chunks_count", 0) > 0, f"Expected chunks_count > 0, got {body}"


def test_backend_002():
    assert TEST_FILENAME_TXT, "No uploaded txt filename captured from backend-001"
    files = _get_files_list()
    filenames = []
    for item in files:
        assert isinstance(item, dict), f"Each file entry must be object, got {type(item)}"
        assert "filename" in item and "status" in item, f"Missing filename/status: {item}"
        filenames.append(item.get("filename"))
    assert TEST_FILENAME_TXT in filenames, f"Uploaded file '{TEST_FILENAME_TXT}' missing in {filenames}"


def test_backend_003():
    assert TEST_QUERY_TOKEN, "No unique query token captured from backend-001"
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": f"请复述这个唯一标识：{TEST_QUERY_TOKEN}"},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    content_type = resp.headers.get("content-type", "")
    assert "text/event-stream" in content_type, f"Expected SSE content-type, got {content_type}"

    has_sources = False
    has_token = False
    has_done = False
    sources_payload = None

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data_str = raw_line[6:].strip()
        if data_str == "[DONE]":
            has_done = True
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if "sources" in data:
            has_sources = True
            sources_payload = data.get("sources")
        if data.get("token"):
            has_token = True

    assert has_sources, "SSE stream missing sources event"
    assert has_token, "SSE stream missing token payload"
    assert has_done, "SSE stream missing [DONE] terminator"
    assert isinstance(sources_payload, list), f"sources should be list, got {type(sources_payload)}"
    if sources_payload:
        first = sources_payload[0]
        assert isinstance(first, dict), f"sources[0] should be object, got {type(first)}"
        assert "source_file" in first and "content" in first and "score" in first, f"Invalid source shape: {first}"


def test_backend_004():
    assert TEST_FILENAME_TXT, "No uploaded txt filename captured from backend-001"
    resp = requests.delete(f"{BASE_URL}/api/files/{TEST_FILENAME_TXT}", timeout=20)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("status") == "Deleted", f"Expected status='Deleted', got {body}"

    time.sleep(0.5)
    files = _get_files_list()
    filenames = [item.get("filename") for item in files if isinstance(item, dict)]
    assert TEST_FILENAME_TXT not in filenames, f"Deleted file still present: {TEST_FILENAME_TXT}"


def test_backend_005():
    _safe_clear_history()
    test_msgs = [
        {"text": f"Hello AI {uuid.uuid4().hex[:6]}", "isAi": False},
        {"text": "Hello User!", "isAi": True},
    ]

    resp = requests.post(f"{BASE_URL}/api/history", json={"messages": test_msgs}, timeout=15)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.json().get("status") == "success", f"Unexpected POST response: {resp.json()}"

    resp = requests.get(f"{BASE_URL}/api/history", timeout=15)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data == test_msgs, f"History mismatch. expected={test_msgs} actual={data}"

    resp = requests.delete(f"{BASE_URL}/api/history", timeout=15)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert resp.json().get("status") == "success", f"Unexpected DELETE response: {resp.json()}"

    resp = requests.get(f"{BASE_URL}/api/history", timeout=15)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data == [], f"Expected empty list after clear, got {data}"


def test_backend_006():
    tmp_path, filename = _create_temp_file(".txt", "")
    try:
        resp = _upload_file(tmp_path, filename, "text/plain")
    finally:
        tmp_path.unlink(missing_ok=True)
    assert resp.status_code == 400, f"Expected 400 for empty file, got {resp.status_code}: {resp.text}"
    detail = str(resp.json().get("detail", "")).lower()
    assert "empty" in detail or "parsed" in detail, f"Unexpected error detail: {detail}"


def test_backend_007():
    tmp_path, filename = _create_temp_file(".exe", "not executable, just test payload")
    try:
        resp = _upload_file(tmp_path, filename, "application/octet-stream")
    finally:
        tmp_path.unlink(missing_ok=True)
    assert resp.status_code == 400, f"Expected 400 for unsupported format, got {resp.status_code}: {resp.text}"
    detail = str(resp.json().get("detail", ""))
    assert "Unsupported file format" in detail, f"Unexpected detail: {detail}"


def test_backend_008():
    resp = requests.post(f"{BASE_URL}/api/chat", json={}, timeout=15)
    assert resp.status_code == 422, f"Expected 422 for invalid payload, got {resp.status_code}: {resp.text}"


def test_backend_009():
    resp = requests.post(f"{BASE_URL}/api/history", json={"messages": "not-an-array"}, timeout=15)
    assert resp.status_code == 422, f"Expected 422 for invalid payload, got {resp.status_code}: {resp.text}"


def test_backend_010():
    global TEST_FILENAME_MD
    md_content = (
        "# NexusAI Markdown Test\n\n"
        f"- token: {uuid.uuid4().hex[:10]}\n"
        "- this file validates md upload lifecycle\n"
    )
    tmp_path, filename = _create_temp_file(".md", md_content)
    TEST_FILENAME_MD = filename
    try:
        resp = _upload_file(tmp_path, filename, "text/markdown")
    finally:
        tmp_path.unlink(missing_ok=True)

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("status") == "Ready", f"Unexpected upload body: {body}"

    files = _get_files_list()
    filenames = [item.get("filename") for item in files if isinstance(item, dict)]
    assert filename in filenames, f"Uploaded md file missing in list: {filenames}"

    resp = requests.delete(f"{BASE_URL}/api/files/{filename}", timeout=20)
    assert resp.status_code == 200, f"Expected 200 on delete, got {resp.status_code}: {resp.text}"
    assert resp.json().get("status") == "Deleted", f"Unexpected delete body: {resp.json()}"

    files_after = _get_files_list()
    after_names = [item.get("filename") for item in files_after if isinstance(item, dict)]
    assert filename not in after_names, f"Deleted md file still exists: {after_names}"


def test_backend_011():
    files = _get_files_list()
    for item in files:
        assert isinstance(item, dict), f"File item should be object, got {type(item)}"
        filename = item.get("filename")
        status = item.get("status")
        assert isinstance(filename, str) and filename.strip(), f"Invalid filename in {item}"
        assert status == "Ready", f"Expected status='Ready', got {status} in {item}"


def test_backend_012():
    from app.services.document_parser import DocumentParser

    txt = DocumentParser.parse("hello txt".encode("utf-8"), "a.txt")
    md = DocumentParser.parse("# hello md".encode("utf-8"), "a.md")
    assert txt == "hello txt", f"Unexpected txt parse result: {txt}"
    assert md == "# hello md", f"Unexpected md parse result: {md}"

    try:
        DocumentParser.parse(b"bad", "x.exe")
    except ValueError as exc:
        assert "Unsupported file format" in str(exc), f"Unexpected ValueError: {exc}"
    else:
        raise AssertionError("Expected ValueError for unsupported extension")


def test_backend_013():
    from app.services.text_chunker import TextChunker

    chunks = TextChunker.chunk("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"], f"Unexpected chunks: {chunks}"
    assert TextChunker.chunk("", chunk_size=4, overlap=1) == [], "Empty text should return []"


def test_frontend_001():
    _run_cmd(["npm", "run", "lint"], FRONTEND_DIR, timeout=900)


def test_frontend_002():
    _run_cmd(["npm", "run", "build"], FRONTEND_DIR, timeout=1200)


def test_frontend_003():
    content = _read_text(FILES_PAGE_PATH)
    assert "confirmingDelete" in content, "Inline delete confirm state missing"
    assert "确认删除" in content, "Confirm delete text missing"
    assert "取消" in content, "Cancel text missing"
    assert "window.confirm" not in content, "window.confirm should not be used"


def test_frontend_004():
    content = _read_text(CHAT_PAGE_PATH)
    assert "showNewChatConfirm" in content, "New chat confirm state missing"
    assert "确认清空？" in content, "New chat confirm text missing"
    assert "/api/history" in content and 'method: "DELETE"' in content, "History clear API call missing"


def test_frontend_005():
    content = _read_text(CHAT_PAGE_PATH)
    assert "data: " in content, "SSE prefix parsing missing"
    assert "[DONE]" in content, "SSE done marker handling missing"
    assert "if (data.token)" in content, "Token handling branch missing"
    assert "else if (data.sources)" in content, "Sources handling branch missing"


def test_frontend_006():
    content = _read_text(HEADER_PATH)
    assert 'path: "/"' in content, "Chat route missing from header nav"
    assert 'path: "/files"' in content, "Files route missing from header nav"


def test_frontend_007():
    content = _read_text(FILES_PAGE_PATH)
    assert "100 * 1024 * 1024" in content, "100MB size cap missing"
    assert ".pdf,.txt,.docx,.md,.pptx" in content, "Accepted extension list mismatch"


TEST_MAP = {
    "infra-001": test_infra_001,
    "infra-002": test_infra_002,
    "infra-003": test_infra_003,
    "backend-001": test_backend_001,
    "backend-002": test_backend_002,
    "backend-003": test_backend_003,
    "backend-004": test_backend_004,
    "backend-005": test_backend_005,
    "backend-006": test_backend_006,
    "backend-007": test_backend_007,
    "backend-008": test_backend_008,
    "backend-009": test_backend_009,
    "backend-010": test_backend_010,
    "backend-011": test_backend_011,
    "backend-012": test_backend_012,
    "backend-013": test_backend_013,
    "frontend-001": test_frontend_001,
    "frontend-002": test_frontend_002,
    "frontend-003": test_frontend_003,
    "frontend-004": test_frontend_004,
    "frontend-005": test_frontend_005,
    "frontend-006": test_frontend_006,
    "frontend-007": test_frontend_007,
}


def main() -> int:
    print("🚀 NexusAI Strict Feature Verification Started...\n")
    features = load_feature_list()

    all_passed = True
    passed_count = 0
    total_auto = 0

    try:
        for feature in features:
            fid = feature["id"]
            steps = feature.get("steps", [])
            is_auto = any(step.get("type") == "auto" for step in steps)
            if not is_auto:
                continue

            total_auto += 1
            desc = feature.get("description", "")
            print(f"▶ Testing: {desc}")
            test_fn = TEST_MAP.get(fid)

            if test_fn is None:
                feature["passes"] = False
                log_fail(fid, "FAILED: no test function mapped")
                all_passed = False
                continue

            try:
                test_fn()
                feature["passes"] = True
                log_pass(fid, "PASSED")
                passed_count += 1
            except AssertionError as err:
                feature["passes"] = False
                log_fail(fid, f"FAILED: {err}")
                all_passed = False
            except Exception as err:
                feature["passes"] = False
                log_fail(fid, f"ERROR: {err}")
                all_passed = False
    finally:
        _safe_delete_file(TEST_FILENAME_TXT)
        _safe_delete_file(TEST_FILENAME_MD)
        _safe_clear_history()
        save_feature_list(features)

    print(f"\n{'=' * 70}")
    print(f"Results: {passed_count}/{total_auto} auto tests passed")
    if all_passed:
        print("🌟 All automated feature tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above.")
    print(f"📄 Updated: {FEATURE_LIST_PATH.resolve()}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
