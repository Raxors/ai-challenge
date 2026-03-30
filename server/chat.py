"""
Interactive chat client for the private LLM service.

Usage:
    python3 server/chat.py
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

BASE_URL = "http://193.124.22.28:28228"
API_KEY = os.getenv("API_KEYS", "").split(",")[0].strip()


def chat():
    history = []
    print("Private LLM Chat")
    print(f"Server: {BASE_URL}")
    print("Type 'exit' to quit, 'clear' to reset history\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("History cleared.\n")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            r = httpx.post(
                f"{BASE_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {},
                json={
                    "messages": history,
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
                timeout=120,
            )

            if r.status_code != 200:
                print(f"Error {r.status_code}: {r.text}\n")
                history.pop()
                continue

            reply = r.json()["choices"][0]["message"]["content"]
            history.append({"role": "assistant", "content": reply})
            print(f"AI: {reply}\n")

        except httpx.ConnectError:
            print("Error: cannot connect to server\n")
            history.pop()
        except httpx.ReadTimeout:
            print("Error: server timeout\n")
            history.pop()


if __name__ == "__main__":
    chat()
