"""
Test: stability under concurrent requests.

Usage:
    python3 server/tests/test_stability.py [BASE_URL] [NUM_REQUESTS]

Default: 10 concurrent requests to http://localhost:8000
Requires the server to be running (docker compose up).
"""

import sys
import time
import asyncio
import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
NUM_REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
API_KEY = "changeme"


async def send_request(client: httpx.AsyncClient, idx: int) -> dict:
    start = time.time()
    try:
        r = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "messages": [{"role": "user", "content": f"Reply with the number {idx}"}],
                "temperature": 0.1,
                "max_tokens": 32,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        return {
            "idx": idx,
            "status": r.status_code,
            "elapsed": elapsed,
            "reply": r.json().get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
            if r.status_code == 200
            else r.text[:100],
        }
    except Exception as e:
        return {"idx": idx, "status": -1, "elapsed": time.time() - start, "reply": str(e)}


async def run():
    print(f"Sending {NUM_REQUESTS} concurrent requests to {BASE_URL}\n")
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)

    ok = sum(1 for r in results if r["status"] == 200)
    rate_limited = sum(1 for r in results if r["status"] == 429)
    errors = sum(1 for r in results if r["status"] not in (200, 429))
    times = [r["elapsed"] for r in results if r["status"] == 200]

    print(f"{'idx':>4} {'status':>6} {'time':>7}  reply")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["idx"]):
        print(f"{r['idx']:>4} {r['status']:>6} {r['elapsed']:>6.1f}s  {r['reply']}")

    print(f"\n{'='*40}")
    print(f"Success:      {ok}/{NUM_REQUESTS}")
    print(f"Rate limited: {rate_limited}/{NUM_REQUESTS}")
    print(f"Errors:       {errors}/{NUM_REQUESTS}")
    if times:
        print(f"Avg time:     {sum(times)/len(times):.1f}s")
        print(f"Max time:     {max(times):.1f}s")

    if errors > 0:
        print("\nFAIL: some requests errored")
        sys.exit(1)
    else:
        print("\nPASS: all requests handled (200 or 429)")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
