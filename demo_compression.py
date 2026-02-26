"""
Демонстрация компрессии контекста.
Сравнение: обычный агент vs агент с компрессией.
"""

import textwrap
from dotenv import load_dotenv
from openai_model import OpenAIModel
from agent import Agent
from llm_interface import LLMError

load_dotenv()

MODEL = "gpt-4o"
SYSTEM_PROMPT = "Ты — полезный помощник. Отвечай подробно на русском языке."

QUESTIONS = [
    "Что такое машинное обучение? Объясни основные типы: supervised, unsupervised, reinforcement learning.",
    "Расскажи про нейронные сети. Что такое перцептрон, слои, функции активации, обратное распространение ошибки?",
    "Чем отличаются CNN от RNN? Где применяется каждая архитектура?",
    "Что такое трансформер? Объясни механизм внимания (attention) и self-attention.",
    "Как работает GPT? Чем он отличается от BERT?",
    "Что такое fine-tuning и transfer learning? Приведи примеры.",
    "Расскажи про RAG (Retrieval-Augmented Generation). Зачем это нужно?",
    "Какие проблемы есть у LLM: галлюцинации, bias, безопасность?",
    "Суммаризируй ВСЁ что мы обсудили в этом диалоге. Проверим помнишь ли ты начало разговора.",
]


def print_wrapped(text, width=90, indent="    "):
    for paragraph in text.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph, width=width, initial_indent=indent, subsequent_indent=indent))
        else:
            print()


def run_agent(agent, label):
    """Запускает серию вопросов и собирает метрики."""
    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}")

    metrics_log = []

    for i, q in enumerate(QUESTIONS, 1):
        try:
            result = agent.ask(q)
            m = result["metrics"]
            metrics_log.append(m)

            compressed_mark = " [СЖАТИЕ]" if m.get("compressed") else ""
            print(f"\n  --- Запрос #{i}{compressed_mark} ---")
            print(f"  Вопрос: {q[:80]}...")
            print(f"  Ответ (начало): {result['text'][:150]}...")
            print(f"  Input: {m['request_input_tokens']}  Output: {m['request_output_tokens']}  "
                  f"История: {m['history_tokens']} токенов  "
                  f"({m['usage_percent']:.1f}%)")

            if m.get("compressed") and m.get("summary"):
                print(f"  Summary: {m['summary'][:100]}...")

        except LLMError as e:
            print(f"\n  !!! Ошибка на запросе #{i}: {e}")
            break

    return metrics_log


def print_comparison(metrics_normal, metrics_compressed):
    """Выводит сравнительную таблицу."""
    print(f"\n{'=' * 100}")
    print("  СРАВНЕНИЕ: БЕЗ СЖАТИЯ vs СО СЖАТИЕМ")
    print(f"{'=' * 100}")

    print(f"\n  {'Запрос':<8} {'--- Без сжатия ---':>30}    {'--- Со сжатием ---':>30}    {'Экономия':>10}")
    print(f"  {'#':<8} {'Input':>10} {'History':>10} {'Cost':>10}    {'Input':>10} {'History':>10} {'Cost':>10}    {'Tokens':>10}")
    print("  " + "-" * 96)

    for i in range(min(len(metrics_normal), len(metrics_compressed))):
        mn = metrics_normal[i]
        mc = metrics_compressed[i]
        saved = mn["request_input_tokens"] - mc["request_input_tokens"]
        print(
            f"  {i+1:<8} "
            f"{mn['request_input_tokens']:>10} {mn['history_tokens']:>10} ${mn['request_cost']:>9.6f}    "
            f"{mc['request_input_tokens']:>10} {mc['history_tokens']:>10} ${mc['request_cost']:>9.6f}    "
            f"{saved:>+10}"
        )

    # Итоги
    if metrics_normal and metrics_compressed:
        total_input_n = sum(m["request_input_tokens"] for m in metrics_normal)
        total_input_c = sum(m["request_input_tokens"] for m in metrics_compressed)
        total_cost_n = metrics_normal[-1]["session_total_cost"]
        total_cost_c = metrics_compressed[-1]["session_total_cost"]
        compression_overhead = metrics_compressed[-1].get("compression_tokens", 0)

        print("  " + "-" * 96)
        print(f"\n  ИТОГО:")
        print(f"    Без сжатия:    {total_input_n:>8} input токенов   ${total_cost_n:.6f}")
        print(f"    Со сжатием:    {total_input_c:>8} input токенов   ${total_cost_c:.6f}")
        print(f"    Overhead сжатия: {compression_overhead} токенов (потрачено на генерацию summary)")
        print()

        if total_input_n > 0:
            saved_pct = (1 - total_input_c / total_input_n) * 100
            print(f"    Экономия input токенов: {saved_pct:.1f}%")
        if total_cost_n > 0:
            cost_saved_pct = (1 - total_cost_c / total_cost_n) * 100
            print(f"    Экономия стоимости:     {cost_saved_pct:.1f}%")

    # Качество
    print(f"\n  КАЧЕСТВО (последний вопрос — проверка памяти):")
    if metrics_normal:
        print(f"    Без сжатия:  агент помнит ВСЮ историю дословно")
    if metrics_compressed:
        print(f"    Со сжатием:  агент помнит СУТЬ через summary, детали могут теряться")
    print()


def main():
    llm = OpenAIModel(model=MODEL)

    # --- Агент БЕЗ сжатия ---
    agent_normal = Agent(
        model=llm, model_name=MODEL, max_tokens=1024,
        system_prompt=SYSTEM_PROMPT,
        db_path="demo_normal.db",
        compress=False,
    )
    agent_normal.reset()
    metrics_normal = run_agent(agent_normal, "АГЕНТ БЕЗ СЖАТИЯ (полная история)")
    agent_normal.close()

    # --- Агент СО сжатием ---
    agent_compressed = Agent(
        model=llm, model_name=MODEL, max_tokens=1024,
        system_prompt=SYSTEM_PROMPT,
        db_path="demo_compressed.db",
        compress=True,
        keep_last=6,  # последние 6 сообщений как есть, остальное -> summary
    )
    agent_compressed.reset()
    metrics_compressed = run_agent(agent_compressed, "АГЕНТ СО СЖАТИЕМ (summary + последние 6 сообщений)")
    agent_compressed.close()

    # --- Сравнение ---
    print_comparison(metrics_normal, metrics_compressed)

    # Удаляем демо-базы
    import os
    for f in ["demo_normal.db", "demo_compressed.db"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
