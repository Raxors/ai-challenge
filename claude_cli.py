import os
import sys
import textwrap
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

load_dotenv()


def main():
    client = OpenAI()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Enter your prompt: ")

    system_prompt = (
        "You are a helpful CLI assistant. "
        "Respond in plain text format, without markdown. "
        "Keep your answers concise — no more than 500 words. "
        "When the answer is complete, stop immediately without adding extra commentary."
    )

    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    try:
        # Ответ 1 — с ограничениями
        response_limited = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=512,
            stop=["\n\n\n"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        print("=" * width)
        print("ОТВЕТ 1 (с ограничениями):")
        print("=" * width)
        for paragraph in response_limited.choices[0].message.content.split("\n"):
            if paragraph:
                print(textwrap.fill(paragraph, width=width))
            else:
                print()

        # Ответ 2 — без ограничений
        response_full = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        print()
        print("=" * width)
        print("ОТВЕТ 2 (без ограничений):")
        print("=" * width)
        for paragraph in response_full.choices[0].message.content.split("\n"):
            if paragraph:
                print(textwrap.fill(paragraph, width=width))
            else:
                print()

    except RateLimitError:
        print("Error: You exceeded your OpenAI quota. Please check your plan and billing at https://platform.openai.com/settings/organization/billing")
    except AuthenticationError:
        print("Error: Invalid API key. Please check your OPENAI_API_KEY in .env file.")
    except APIError as e:
        print(f"Error: OpenAI API error - {e.message}")


if __name__ == "__main__":
    main()
