import os
import sys
import textwrap
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

load_dotenv()


def print_wrapped(text, width):
    for paragraph in text.split("\n"):
        if paragraph:
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def print_header(title, width):
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def main():
    client = OpenAI()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Enter your prompt: ")

    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    try:
        # --- Ответ 1: Температура 0 ---
        print_header("ОТВЕТ 1: Температура 0", width)
        r1 = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        print_wrapped(r1.choices[0].message.content, width)

        # --- Ответ 2: Температура 0.7 ---
        print_header("ОТВЕТ 2: Температура 0.7", width)
        r2 = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        print_wrapped(r2.choices[0].message.content, width)

        # --- Ответ 3: Температура 1.2 ---
        print_header("ОТВЕТ 3: Температура 1.2", width)
        r3 = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.2
        )
        print_wrapped(r3.choices[0].message.content, width)

        print_wrapped("Выводы:\n"
                      "Чем меньше температура, тем более точны и менее креативны ответы.\n"
                      "Использовать низкую температуру для задач требующих точности, например математика, код итд.\n"
                      "Использовать высокую температуру для творческих задач, например написать рассказ, придумать идею.\n",
                      width)

    except RateLimitError:
        print(
            "Error: You exceeded your OpenAI quota. Please check your plan and billing at https://platform.openai.com/settings/organization/billing")
    except AuthenticationError:
        print("Error: Invalid API key. Please check your OPENAI_API_KEY in .env file.")
    except APIError as e:
        print(f"Error: OpenAI API error - {e.message}")


if __name__ == "__main__":
    main()
