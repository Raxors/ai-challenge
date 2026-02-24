import os
import textwrap
from dotenv import load_dotenv
from openai_model import OpenAIModel
from agent import Agent
from llm_interface import LLMError

load_dotenv()


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


def main():
    llm = OpenAIModel(model="gpt-4o")
    agent = Agent(
        model=llm,
        max_tokens=1024,
        system_prompt="You are a helpful assistant. Answer concisely and clearly.",
    )

    width = get_width()
    msg_count = agent.get_message_count()
    if msg_count > 0:
        print(f"Агент запущен. Загружена история ({msg_count} сообщений).")
    else:
        print("Агент запущен. Новый диалог.")
    print("Команды: 'exit' — выход, 'reset' — сброс диалога.")
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
                print("[История диалога сброшена]")
                continue

            try:
                result = agent.ask(user_input)
            except LLMError as e:
                print(f"\nОшибка: {e}")
                continue

            width = get_width()
            print()
            print_wrapped(result["text"], width)
            print(f"\n  [токены: {result['input_tokens']} in / {result['output_tokens']} out]")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
