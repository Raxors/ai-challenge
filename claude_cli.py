import os
import sys
import time
import textwrap
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# Модели: слабая, средняя, сильная
MODELS = [
    {
        "name": "gpt-4o-mini",
        "label": "Слабая модель",
        "input_price": 0.15,   # $ за 1M input токенов
        "output_price": 0.60,  # $ за 1M output токенов
    },
    {
        "name": "gpt-4o",
        "label": "Средняя модель",
        "input_price": 2.50,
        "output_price": 10.00,
    },
    {
        "name": "o1",
        "label": "Сильная модель",
        "input_price": 15.00,
        "output_price": 60.00,
    },
]


def print_wrapped(text, width):
    for paragraph in text.split("\n"):
        if paragraph:
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def calc_cost(model_info, input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * model_info["input_price"]
    output_cost = (output_tokens / 1_000_000) * model_info["output_price"]
    return input_cost + output_cost


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

    results = []

    for model_info in MODELS:
        print()
        print("=" * width)
        print(f"{model_info['label']} ({model_info['name']})")
        print("=" * width)

        try:
            start_time = time.time()
            params = {
                "model": model_info["name"],
                "messages": [{"role": "user", "content": prompt}],
            }
            if model_info["name"].startswith("o"):
                params["max_completion_tokens"] = 16384
            else:
                params["max_tokens"] = 1024
            response = client.chat.completions.create(**params)
            elapsed = time.time() - start_time

            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            cost = calc_cost(model_info, input_tokens, output_tokens)

            print_wrapped(text, width)
            print()
            print(f"  Время ответа:    {elapsed:.2f} сек")
            print(f"  Токены (in/out):  {input_tokens} / {output_tokens} (всего: {total_tokens})")
            print(f"  Стоимость:       ${cost:.6f}")

            results.append({
                "label": model_info["label"],
                "name": model_info["name"],
                "time": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
            })

        except RateLimitError:
            print("Error: Quota exceeded for this model.")
        except AuthenticationError:
            print("Error: Invalid API key.")
        except APIError as e:
            print(f"Error: {e.message}")

    # --- Сравнительная таблица ---
    if len(results) > 1:
        print()
        print("=" * width)
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * width)
        print(f"{'Модель':<25} {'Время':>10} {'Токены':>10} {'Стоимость':>12}")
        print("-" * 60)
        for r in results:
            print(f"{r['name']:<25} {r['time']:>8.2f}s {r['total_tokens']:>10} ${r['cost']:>11.6f}")

        fastest = min(results, key=lambda x: x["time"])
        cheapest = min(results, key=lambda x: x["cost"])
        most_expensive = max(results, key=lambda x: x["cost"])
        slowest = max(results, key=lambda x: x["time"])
        print()
        print(f"  Самая быстрая:  {fastest['name']} ({fastest['time']:.2f}s)")
        print(f"  Самая медленная: {slowest['name']} ({slowest['time']:.2f}s)")
        print(f"  Самая дешёвая:  {cheapest['name']} (${cheapest['cost']:.6f})")
        print(f"  Самая дорогая:  {most_expensive['name']} (${most_expensive['cost']:.6f})")

        if len(results) >= 2:
            speed_diff = slowest["time"] / fastest["time"]
            cost_diff = most_expensive["cost"] / cheapest["cost"] if cheapest["cost"] > 0 else 0
            print()
            print(f"  Разница в скорости:   x{speed_diff:.1f}")
            print(f"  Разница в стоимости:  x{cost_diff:.1f}")

        # --- Вывод и рекомендации ---
        print()
        print("=" * width)
        print("ВЫВОД И РЕКОМЕНДАЦИИ")
        print("=" * width)
        print()
        recommendations = [
            (
                "gpt-4o-mini (слабая)",
                "Самая дешёвая. Подходит для простых задач: "
                "классификация текста, извлечение данных, простые вопросы, "
                "чат-боты с большим потоком запросов, прототипирование. "
                "Не подходит для сложных рассуждений и анализа."
            ),
            (
                "gpt-4o (средняя)",
                "Оптимальный баланс цены и качества. Подходит для большинства задач: "
                "генерация кода, написание текстов, суммаризация, анализ документов, "
                "работа с изображениями."
            ),
            (
                "o1 (сильная)",
                "Самая мощная, но медленная и дорогая. Подходит для сложных задач: "
                "математика, логика, научный анализ, стратегическое планирование, "
                "задачи требующие глубокого reasoning. "
                "Не оправдана для простых запросов."
            ),
        ]
        for name, desc in recommendations:
            print(f"  {name}:")
            print_wrapped(f"    {desc}", width)
            print()


if __name__ == "__main__":
    main()
