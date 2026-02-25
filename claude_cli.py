import os
import textwrap

from dotenv import load_dotenv
from openai_model import OpenAIModel
from agent import Agent
from llm_interface import LLMError

load_dotenv()

MODEL_NAME = "gpt-4o"


def print_wrapped(text, width):
    for paragraph in text.split("\n"):
        if paragraph:
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def get_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def print_metrics(metrics, width):
    """Выводит статистику токенов после каждого запроса."""
    m = metrics
    bar_len = 30
    filled = int(bar_len * m["usage_percent"] / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    # Цвет предупреждения
    if m["usage_percent"] > 90:
        status = "!! CRITICAL"
    elif m["usage_percent"] > 70:
        status = "! WARNING"
    else:
        status = "OK"

    print("-" * width)
    print(f"  Запрос #{m['request_number']}:")
    print(f"    Input tokens:    {m['request_input_tokens']}")
    print(f"    Output tokens:   {m['request_output_tokens']}")
    print(f"    Стоимость:       ${m['request_cost']:.6f}")
    print()
    print(f"  История диалога:   {m['history_tokens']} / {m['context_limit']} токенов")
    print(f"    [{bar}] {m['usage_percent']:.1f}% {status}")
    print()
    print(f"  Итого за сессию:")
    print(f"    Input:           {m['session_total_input']} токенов")
    print(f"    Output:          {m['session_total_output']} токенов")
    print(f"    Стоимость:       ${m['session_total_cost']:.6f}")
    print("-" * width)


def main():
    llm = OpenAIModel(model=MODEL_NAME)
    agent = Agent(
        model=llm,
        model_name=MODEL_NAME,
        max_tokens=1024,
        system_prompt="You are a helpful assistant. Answer concisely and clearly.",
    )

    width = get_width()
    msg_count = agent.get_message_count()
    if msg_count > 0:
        print(f"Агент запущен. Загружена история ({msg_count} сообщений).")
    else:
        print("Агент запущен. Новый диалог.")
    print("Команды: 'exit' — выход, 'reset' — сброс, 'stats' — статистика.")
    print("=" * width)

    try:
        while True:
            try:
                user_input = input("\nВы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nВыход.")
                break

            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("Выход.")
                break
            if user_input.lower() == "reset":
                agent.reset()
                print("[История диалога сброшена. Счётчики обнулены.]")
                continue
            if user_input.lower() == "stats":
                print(f"\n  Запросов за сессию: {agent.request_number}")
                print(f"  Input токенов:      {agent.session_input_tokens}")
                print(f"  Output токенов:     {agent.session_output_tokens}")
                print(f"  Общая стоимость:    ${agent.session_cost:.6f}")
                continue

            try:
                result = agent.ask(user_input)
            except LLMError as e:
                print(f"\nОшибка: {e}")
                continue

            width = get_width()
            print()
            print_wrapped(result["text"], width)
            print()
            print_metrics(result["metrics"], width)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
