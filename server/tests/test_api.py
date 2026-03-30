"""
Test: network access to the LLM API and basic chat functionality.

Usage:
    python3 server/tests/test_api.py [BASE_URL]

Default BASE_URL: http://localhost:8000
Requires the server to be running (docker compose up).
"""

import sys
import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
API_KEY = "changeme"
PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name} — {detail}")


def test_health():
    print("\n--- Health Check ---")
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    check("health status 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    check("health has status field", "status" in data)
    check("health has model field", "model" in data)


def test_chat():
    print("\n--- Chat Completion ---")
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "temperature": 0.1,
            "max_tokens": 32,
        },
        timeout=120,
    )
    check("chat status 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    check("has choices", len(data.get("choices", [])) > 0)
    if data.get("choices"):
        msg = data["choices"][0]["message"]["content"]
        check("response not empty", len(msg) > 0, f"got empty response")
        print(f"  Model replied: {msg[:100]}")
    check("has usage", "usage" in data)


def test_auth_required():
    print("\n--- Auth Required ---")
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
        timeout=10,
    )
    check("no auth → 403", r.status_code == 403, f"got {r.status_code}")

    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": "Bearer wrong-key"},
        json={"messages": [{"role": "user", "content": "test"}]},
        timeout=10,
    )
    check("bad key → 401", r.status_code == 401, f"got {r.status_code}")


def test_context_limit():
    print("\n--- Context Limit ---")
    long_msg = "word " * 20000  # ~20k tokens
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"messages": [{"role": "user", "content": long_msg}]},
        timeout=10,
    )
    check("too long → 400", r.status_code == 400, f"got {r.status_code}")


def main():
    print(f"Testing LLM API at {BASE_URL}")
    test_health()
    test_auth_required()
    test_context_limit()
    test_chat()
    print(f"\n{'='*40}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
