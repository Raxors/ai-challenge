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
        # --- Ответ 1: Прямой ответ без дополнительных инструкций ---
        print_header("ОТВЕТ 1: Прямой ответ", width)
        r1 = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        print_wrapped(r1.choices[0].message.content, width)

        # --- Ответ 2: Пошаговое решение ---
        print_header("ОТВЕТ 2: Пошаговое решение", width)
        r2 = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "Решай любую задачу пошагово. Разбей решение на чёткие нумерованные шаги. В конце дай итоговый ответ.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        print_wrapped(r2.choices[0].message.content, width)

        # --- Ответ 3: Модель сначала составляет промпт, затем решает ---
        print_header("ОТВЕТ 3: Самостоятельно составленный промпт + решение", width)
        r3_meta = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": "Ты — эксперт по prompt engineering. Пользователь даст тебе задачу. Составь идеальный промпт для решения этой задачи. Верни ТОЛЬКО текст промпта, без пояснений.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        generated_prompt = r3_meta.choices[0].message.content
        print(f"[Сгенерированный промпт]: {generated_prompt}")
        print("-" * width)

        r3 = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": generated_prompt}],
        )
        print_wrapped(r3.choices[0].message.content, width)

        # --- Ответ 4: Группа экспертов ---
        print_header("ОТВЕТ 4: Группа экспертов (Аналитик, Инженер, Критик)", width)
        r4 = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты симулируешь группу из трёх экспертов, которые обсуждают задачу пользователя. "
                        "Каждый эксперт даёт свой ответ:\n\n"
                        "1. АНАЛИТИК — разбирает задачу, выделяет ключевые аспекты и данные.\n"
                        "2. ИНЖЕНЕР — предлагает практическое решение и реализацию.\n"
                        "3. КРИТИК — находит слабые места в предложенных решениях и предлагает улучшения.\n\n"
                        "Формат ответа:\n"
                        "[Аналитик]: ...\n"
                        "[Инженер]: ...\n"
                        "[Критик]: ...\n"
                        "[Итог]: общий вывод на основе мнений всех экспертов."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        print_wrapped(r4.choices[0].message.content, width)

    except RateLimitError:
        print("Error: You exceeded your OpenAI quota. Please check your plan and billing at https://platform.openai.com/settings/organization/billing")
    except AuthenticationError:
        print("Error: Invalid API key. Please check your OPENAI_API_KEY in .env file.")
    except APIError as e:
        print(f"Error: OpenAI API error - {e.message}")


if __name__ == "__main__":
    main()
