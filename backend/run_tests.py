#!/usr/bin/env python3
"""
run_tests.py — NexusAI Automated Feature Verification

Runs through all 'auto' steps in feature_list.json and marks passes.
Requires: backend running on localhost:8001, Qdrant on localhost:6333.

Usage:
    cd backend
    python run_tests.py
"""

import json
import os
import sys
import time
import tempfile
import requests

BASE_URL = "http://localhost:8001"
FEATURE_LIST_PATH = os.path.join(os.path.dirname(__file__), "..", "feature_list.json")

# ── Helpers ─────────────────────────────────────────────────────────────────

def log_pass(feature_id, msg):
    print(f"  ✅ [{feature_id}] {msg}")

def log_fail(feature_id, msg):
    print(f"  ❌ [{feature_id}] {msg}")

def load_feature_list():
    with open(FEATURE_LIST_PATH, "r") as f:
        return json.load(f)

def save_feature_list(features):
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
        f.write("\n")

# ── Test Functions ──────────────────────────────────────────────────────────

TEST_FILENAME = None  # Will be set during upload test

def test_infra_001():
    """Health Check: Backend API and Qdrant are alive."""
    # Retry in case server is restarting (e.g., due to --reload)
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(f"{BASE_URL}/api/health", timeout=30)
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            body = resp.json()
            assert body.get("status") == "ok", f"Expected status='ok', got {body}"
            assert body.get("qdrant") is True, f"Qdrant not reachable: {body}"
            return True
        except Exception as e:
            last_err = e
            if attempt < 2:
                print(f"    ⏳ Retrying health check ({attempt+1}/3)...")
                time.sleep(5)
    raise last_err

def test_backend_001():
    """File Upload & Processing."""
    global TEST_FILENAME
    # Create a real test file
    test_content = (
        "NexusAI Test Document\n\n"
        "This is a test document for the NexusAI knowledge base system.\n"
        "It contains information about artificial intelligence and machine learning.\n"
        "The purpose of this document is to verify the upload, parsing, chunking,\n"
        "embedding, and vector storage pipeline.\n\n"
        "Key Topics:\n"
        "- Natural Language Processing (NLP)\n"
        "- Retrieval-Augmented Generation (RAG)\n"
        "- Vector Databases like Qdrant\n"
        "- Large Language Models (LLMs)\n"
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="nexusai_test_", delete=False) as f:
        f.write(test_content)
        tmp_path = f.name
    
    TEST_FILENAME = os.path.basename(tmp_path)
    
    try:
        with open(tmp_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/api/upload",
                files={"file": (TEST_FILENAME, f, "text/plain")},
                timeout=120  # Embedding can be slow on first load
            )
        
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body.get("status") == "Ready", f"Expected status='Ready', got {body}"
        assert body.get("chunks_count", 0) > 0, f"Expected chunks_count > 0, got {body}"
        assert body.get("filename") == TEST_FILENAME, f"Filename mismatch: {body}"
        return True
    finally:
        os.unlink(tmp_path)

def test_backend_002():
    """Knowledge Base List."""
    resp = requests.get(f"{BASE_URL}/api/files", timeout=10)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    assert isinstance(body, list), f"Expected array, got {type(body)}"
    filenames = [item.get("filename") if isinstance(item, dict) else item for item in body]
    assert TEST_FILENAME in filenames, f"Uploaded file '{TEST_FILENAME}' not found in {filenames}"
    return True

def test_backend_003():
    """RAG Chat with SSE streaming."""
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "What topics are covered in the uploaded document?"},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300  # DeepSeek Reasoner can be slow to start
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    
    content_type = resp.headers.get("content-type", "")
    assert "text/event-stream" in content_type, f"Expected text/event-stream, got {content_type}"
    
    has_sources = False
    has_token = False
    has_done = False
    
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            has_done = True
            break
        try:
            data = json.loads(data_str)
            if "sources" in data:
                has_sources = True
            if "token" in data:
                has_token = True
        except json.JSONDecodeError:
            pass
    
    assert has_sources, "SSE stream did not contain 'sources' event"
    assert has_token, "SSE stream did not contain any 'token' events"
    assert has_done, "SSE stream did not end with [DONE]"
    return True

def test_backend_004():
    """File Deletion Cascade."""
    # Step 1: Delete the file
    resp = requests.delete(f"{BASE_URL}/api/files/{TEST_FILENAME}", timeout=10)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    assert body.get("status") == "Deleted", f"Expected status='Deleted', got {body}"
    
    # Step 2: Verify it's gone
    time.sleep(0.5)  # Small delay for Qdrant to propagate
    resp = requests.get(f"{BASE_URL}/api/files", timeout=10)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    filenames = [item.get("filename") if isinstance(item, dict) else item for item in body]
    assert TEST_FILENAME not in filenames, f"Deleted file '{TEST_FILENAME}' still in list: {filenames}"
    return True

# ── Runner ──────────────────────────────────────────────────────────────────

TEST_MAP = {
    "infra-001": test_infra_001,
    "backend-001": test_backend_001,
    "backend-002": test_backend_002,
    "backend-003": test_backend_003,
    "backend-004": test_backend_004,
}

def main():
    print("🚀 NexusAI Feature Verification Started...\n")
    
    features = load_feature_list()
    all_passed = True
    passed_count = 0
    total_auto = 0
    
    for feature in features:
        fid = feature["id"]
        test_fn = TEST_MAP.get(fid)
        
        if test_fn is None:
            # Skip non-auto tests (browser tests)
            continue
        
        total_auto += 1
        desc = feature["description"]
        print(f"▶ Testing: {desc}")
        
        try:
            test_fn()
            feature["passes"] = True
            log_pass(fid, "PASSED")
            passed_count += 1
        except AssertionError as e:
            feature["passes"] = False
            log_fail(fid, f"FAILED: {e}")
            all_passed = False
        except Exception as e:
            feature["passes"] = False
            log_fail(fid, f"ERROR: {e}")
            all_passed = False
    
    # Save updated results
    save_feature_list(features)
    
    print(f"\n{'='*60}")
    print(f"Results: {passed_count}/{total_auto} auto tests passed")
    
    if all_passed:
        print("🌟 All automated feature tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above.")
    
    print(f"📄 Updated: {os.path.abspath(FEATURE_LIST_PATH)}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
