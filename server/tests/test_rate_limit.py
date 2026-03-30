"""
Test: rate limiting works correctly.

Usage:
    python3 server/tests/test_rate_limit.py [BASE_URL]

Sends RATE_LIMIT + 5 requests rapidly and checks that excess gets 429.
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


def test_rate_limit():
    print("\n--- Rate Limit Test ---")

    # First, check what the limit is via a small burst
    # Send 35 requests rapidly (default limit is 30/min)
    total = 35
    statuses = []
    for i in range(total):
        r = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
            timeout=120,
        )
        statuses.append(r.status_code)

    ok_count = statuses.count(200)
    limited_count = statuses.count(429)

    print(f"  Sent {total} requests: {ok_count} ok, {limited_count} rate-limited")
    check("some requests succeeded", ok_count > 0, f"got {ok_count}")
    check("some requests rate-limited", limited_count > 0, f"got {limited_count}")
    check("rate limit returns 429", 429 in statuses, f"statuses: {set(statuses)}")


def main():
    print(f"Testing rate limits at {BASE_URL}")
    test_rate_limit()
    print(f"\n{'='*40}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
