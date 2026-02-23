import os
import textwrap
from dotenv import load_dotenv
from openai_model import OpenAIModel
from agent import Agent

load_dotenv()


def print_wrapped(text, width):
    for paragraph in text.split("\n"):
        if paragraph:
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def main():
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    llm = OpenAIModel(model="gpt-4o")
    agent = Agent(
        model=llm,
        max_tokens=1024,
        system_prompt="You are a helpful assistant. Answer concisely and clearly.",
    )

    print("Агент запущен. Введите запрос (или 'exit' для выхода, 'reset' для сброса диалога).")
    print("=" * width)

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

        result = agent.ask(user_input)

        if "error" in result:
            print(f"\nОшибка: {result['error']}")
        else:
            print()
            print_wrapped(result["text"], width)
            print(f"\n  [токены: {result['input_tokens']} in / {result['output_tokens']} out]")


if __name__ == "__main__":
    main()
