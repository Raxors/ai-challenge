"""
Тестовый сценарий: собираем ТЗ на мобильное приложение (12 сообщений).
Прогоняем через все 3 стратегии и сравниваем результаты.
"""

import os
import json
import time
from dotenv import load_dotenv
from openai_model import OpenAIModel
from agent import Agent
from context_strategies import BranchingStrategy
from llm_interface import LLMError

load_dotenv()

MODEL_NAME = "gpt-4o"

SYSTEM_PROMPT = (
    "Ты — бизнес-аналитик. Помогаешь заказчику составить техническое задание "
    "на мобильное приложение. Задавай уточняющие вопросы. Запоминай все детали. "
    "Отвечай кратко, по делу."
)

# Сценарий: 12 сообщений, собираем ТЗ на приложение для доставки еды
SCENARIO = [
    "Привет! Мне нужно мобильное приложение для доставки еды.",
    "Название проекта — FoodFly. Целевая аудитория — жители Москвы 20-45 лет.",
    "Бюджет — 3 миллиона рублей. Срок — 4 месяца. Нужны iOS и Android.",
    "Основные функции: каталог ресторанов, корзина, оплата онлайн, отслеживание курьера на карте.",
    "Оплата через Сбербанк и Тинькофф. Ещё нужна поддержка Apple Pay и Google Pay.",
    "Для авторизации — вход по номеру телефона через SMS. Без паролей.",
    "Дизайн — минималистичный, светлая тема, основной цвет — оранжевый (#FF6B00).",
    "Нужна админ-панель для ресторанов: управление меню, заказами, статистика.",
    "Уведомления push — о статусе заказа, акциях. Интеграция с Firebase.",
    "Курьерское приложение отдельное: принятие заказов, навигация, чат с клиентом.",
    "Важное ограничение: не используем React Native, только нативная разработка. Swift + Kotlin.",
    # Финальный вопрос — проверка памяти
    "Напомни мне: какой бюджет проекта, какие платёжные системы мы обсуждали, и какие технологии для разработки выбрали? Перечисли все детали.",
]

DIVIDER = "=" * 80


def run_sliding_window(llm):
    """Прогон с Sliding Window (N=6 — маленькое окно, чтобы увидеть потери)."""
    print(f"\n{DIVIDER}")
    print("  СТРАТЕГИЯ 1: Sliding Window (window_size=6)")
    print(DIVIDER)

    agent = Agent(
        model=llm, model_name=MODEL_NAME, max_tokens=512,
        system_prompt=SYSTEM_PROMPT, db_path=":memory:",
        strategy_name="sliding_window", window_size=6,
    )

    results = []
    for i, msg in enumerate(SCENARIO, 1):
        print(f"\n[{i}/{len(SCENARIO)}] User: {msg[:80]}...")
        try:
            result = agent.ask(msg)
            results.append(result)
            print(f"  Assistant: {result['text'][:120]}...")
            m = result["metrics"]
            print(f"  [sent={m['sent_tokens']}tok, in={m['request_input_tokens']}, out={m['request_output_tokens']}]")
        except LLMError as e:
            print(f"  ERROR: {e}")
            results.append(None)

    agent.close()
    return results


def run_sticky_facts(llm):
    """Прогон с Sticky Facts (window_size=4 — ещё меньше окно, но есть факты)."""
    print(f"\n{DIVIDER}")
    print("  СТРАТЕГИЯ 2: Sticky Facts (window_size=4)")
    print(DIVIDER)

    agent = Agent(
        model=llm, model_name=MODEL_NAME, max_tokens=512,
        system_prompt=SYSTEM_PROMPT, db_path=":memory:",
        strategy_name="sticky_facts", window_size=4,
    )

    results = []
    for i, msg in enumerate(SCENARIO, 1):
        print(f"\n[{i}/{len(SCENARIO)}] User: {msg[:80]}...")
        try:
            result = agent.ask(msg)
            results.append(result)
            print(f"  Assistant: {result['text'][:120]}...")
            m = result["metrics"]
            si = m.get("strategy_info", {})
            print(f"  [sent={m['sent_tokens']}tok, in={m['request_input_tokens']}, out={m['request_output_tokens']}, facts={si.get('facts_count', 0)}]")
        except LLMError as e:
            print(f"  ERROR: {e}")
            results.append(None)

    # Показываем финальные факты
    print("\n  Финальные факты:")
    si = agent.strategy.get_state_info()
    for k, v in si.get("facts", {}).items():
        print(f"    {k}: {v}")

    agent.close()
    return results


def run_branching(llm):
    """Прогон с Branching — создаём чекпоинт посередине и две ветки."""
    print(f"\n{DIVIDER}")
    print("  СТРАТЕГИЯ 3: Branching (checkpoint + 2 branches)")
    print(DIVIDER)

    agent = Agent(
        model=llm, model_name=MODEL_NAME, max_tokens=512,
        system_prompt=SYSTEM_PROMPT, db_path=":memory:",
        strategy_name="branching", window_size=10,
    )

    strategy: BranchingStrategy = agent.strategy
    results_main = []

    # Отправляем первые 6 сообщений (основная ветка)
    for i, msg in enumerate(SCENARIO[:6], 1):
        print(f"\n[main {i}/6] User: {msg[:80]}...")
        try:
            result = agent.ask(msg)
            results_main.append(result)
            print(f"  Assistant: {result['text'][:120]}...")
        except LLMError as e:
            print(f"  ERROR: {e}")
            results_main.append(None)

    # Создаём чекпоинт после 6 сообщений
    strategy.create_checkpoint("after_6_msgs", agent.history)
    print(f"\n  [CHECKPOINT 'after_6_msgs' создан — {len(agent.history)} сообщений]")

    # Ветка A: продолжаем с дизайном
    strategy.create_branch("branch_A", checkpoint_name="after_6_msgs")
    strategy.switch_branch("branch_A")
    print("\n  --- Переключение на branch_A (дизайн) ---")

    branch_a_msgs = [
        "Дизайн — минималистичный, тёмная тема, основной цвет — синий (#0066FF).",
        "Напомни: какой бюджет проекта и какие платёжные системы?",
    ]
    results_a = []
    for i, msg in enumerate(branch_a_msgs, 1):
        print(f"\n[branch_A {i}/{len(branch_a_msgs)}] User: {msg[:80]}...")
        try:
            result = agent.ask(msg)
            results_a.append(result)
            print(f"  Assistant: {result['text'][:120]}...")
        except LLMError as e:
            print(f"  ERROR: {e}")
            results_a.append(None)

    # Ветка B: продолжаем с другими требованиями
    strategy.create_branch("branch_B", checkpoint_name="after_6_msgs")
    strategy.switch_branch("branch_B")
    print("\n  --- Переключение на branch_B (технологии) ---")

    branch_b_msgs = [
        "Используем Flutter для кроссплатформенной разработки.",
        "Напомни: какой бюджет проекта и какие платёжные системы?",
    ]
    results_b = []
    for i, msg in enumerate(branch_b_msgs, 1):
        print(f"\n[branch_B {i}/{len(branch_b_msgs)}] User: {msg[:80]}...")
        try:
            result = agent.ask(msg)
            results_b.append(result)
            print(f"  Assistant: {result['text'][:120]}...")
        except LLMError as e:
            print(f"  ERROR: {e}")
            results_b.append(None)

    # Показываем ветки
    print("\n  Ветки:")
    for name, info in strategy.list_branches().items():
        marker = " <--" if name == (strategy.current_branch or "main") else ""
        print(f"    {name}: {info}{marker}")

    agent.close()
    return results_main, results_a, results_b


def analyze_results(sw_results, sf_results, br_main, br_a, br_b):
    """Анализ и сравнение результатов."""
    print(f"\n\n{'#' * 80}")
    print("  СРАВНИТЕЛЬНЫЙ АНАЛИЗ СТРАТЕГИЙ")
    print('#' * 80)

    def calc_totals(results):
        total_in = sum(r["metrics"]["request_input_tokens"] for r in results if r)
        total_out = sum(r["metrics"]["request_output_tokens"] for r in results if r)
        total_cost = sum(r["metrics"]["request_cost"] for r in results if r)
        return total_in, total_out, total_cost

    # Sliding Window
    sw_in, sw_out, sw_cost = calc_totals(sw_results)
    sw_last = sw_results[-1] if sw_results[-1] else None
    sw_last_text = sw_last["text"] if sw_last else "ERROR"

    # Sticky Facts
    sf_in, sf_out, sf_cost = calc_totals(sf_results)
    sf_extra = 0
    if sf_results[-1]:
        si = sf_results[-1]["metrics"].get("strategy_info", {})
        sf_extra = si.get("facts_extraction_tokens", 0)
    sf_last = sf_results[-1] if sf_results[-1] else None
    sf_last_text = sf_last["text"] if sf_last else "ERROR"

    # Branching
    br_all = br_main + br_a + br_b
    br_in, br_out, br_cost = calc_totals(br_all)

    # Таблица сравнения
    print(f"""
┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Метрика             │ Sliding Window   │ Sticky Facts     │ Branching        │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Input токены        │ {sw_in:>16} │ {sf_in:>16} │ {br_in:>16} │
│ Output токены       │ {sw_out:>16} │ {sf_out:>16} │ {br_out:>16} │
│ Доп. токены (факты) │ {0:>16} │ {sf_extra:>16} │ {0:>16} │
│ Всего токенов       │ {sw_in+sw_out:>16} │ {sf_in+sf_out+sf_extra:>16} │ {br_in+br_out:>16} │
│ Стоимость ($)       │ {sw_cost:>16.6f} │ {sf_cost+sf_extra*2.5/1e6:>16.6f} │ {br_cost:>16.6f} │
│ Запросов к LLM      │ {len(sw_results):>16} │ {len(sf_results):>16} │ {len(br_all):>16} │
└─────────────────────┴──────────────────┴──────────────────┴──────────────────┘
""")

    # Проверка памяти — последний вопрос
    print("=" * 80)
    print("  ПРОВЕРКА ПАМЯТИ (финальный ответ)")
    print("=" * 80)

    check_items = ["3 миллион", "бюджет", "Сбербанк", "Тинькофф", "Apple Pay", "Google Pay",
                   "Swift", "Kotlin", "натив"]

    print(f"\n--- Sliding Window ---")
    print(f"Ответ: {sw_last_text[:500]}")
    sw_found = sum(1 for item in check_items if item.lower() in sw_last_text.lower())
    print(f"Найдено ключевых деталей: {sw_found}/{len(check_items)}")

    print(f"\n--- Sticky Facts ---")
    print(f"Ответ: {sf_last_text[:500]}")
    sf_found = sum(1 for item in check_items if item.lower() in sf_last_text.lower())
    print(f"Найдено ключевых деталей: {sf_found}/{len(check_items)}")

    # Branching — проверяем ответы двух веток
    br_a_last = br_a[-1]["text"] if br_a and br_a[-1] else "N/A"
    br_b_last = br_b[-1]["text"] if br_b and br_b[-1] else "N/A"

    check_items_br = ["3 миллион", "бюджет", "Сбербанк", "Тинькофф", "Apple Pay", "Google Pay"]

    print(f"\n--- Branching: branch_A (дизайн=синий) ---")
    print(f"Ответ: {br_a_last[:500]}")
    br_a_found = sum(1 for item in check_items_br if item.lower() in br_a_last.lower())
    print(f"Найдено ключевых деталей: {br_a_found}/{len(check_items_br)}")
    has_blue = "синий" in br_a_last.lower() or "0066FF" in br_a_last.upper()
    has_orange = "оранжев" in br_a_last.lower() or "FF6B00" in br_a_last.upper()
    print(f"Упоминает синий (branch_A): {'Да' if has_blue else 'Нет'}")
    print(f"Упоминает оранжевый (не из этой ветки): {'Да (ОШИБКА)' if has_orange else 'Нет (OK)'}")

    print(f"\n--- Branching: branch_B (технологии=Flutter) ---")
    print(f"Ответ: {br_b_last[:500]}")
    br_b_found = sum(1 for item in check_items_br if item.lower() in br_b_last.lower())
    print(f"Найдено ключевых деталей: {br_b_found}/{len(check_items_br)}")
    has_flutter = "flutter" in br_b_last.lower()
    has_swift = "swift" in br_b_last.lower()
    print(f"Упоминает Flutter (branch_B): {'Да' if has_flutter else 'Нет'}")
    print(f"Упоминает Swift (не из этой ветки): {'Да (ОШИБКА)' if has_swift else 'Нет (OK)'}")

    # Итоговая оценка
    print(f"\n{'=' * 80}")
    print("  ИТОГОВАЯ ОЦЕНКА")
    print('=' * 80)
    print(f"""
┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Критерий            │ Sliding Window   │ Sticky Facts     │ Branching        │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Память деталей      │ {sw_found}/{len(check_items)} деталей    │ {sf_found}/{len(check_items)} деталей    │ {br_a_found}/{len(check_items_br)} + {br_b_found}/{len(check_items_br)} дет.  │
│ Расход токенов      │ {'Низкий' if sw_in < sf_in else 'Высокий':>16} │ {'Средний':>16} │ {'Зависит':>16} │
│ Стабильность        │ {'Теряет':>16} │ {'Стабильно':>16} │ {'Стабильно':>16} │
│ Ветвление           │ {'Нет':>16} │ {'Нет':>16} │ {'Да':>16} │
│ Рекомендация        │ {'Короткие':>16} │ {'Длинные':>16} │ {'Исследование':>16} │
│                     │ {'диалоги':>16} │ {'диалоги':>16} │ {'вариантов':>16} │
└─────────────────────┴──────────────────┴──────────────────┴──────────────────┘

Выводы:
  1. Sliding Window — самый экономичный, но теряет ранние детали при малом окне.
     Подходит для коротких диалогов или задач без долгой памяти.

  2. Sticky Facts — лучший баланс: извлекает и хранит ключевые факты,
     не теряет важное даже при маленьком окне сообщений.
     Дополнительные затраты на извлечение фактов окупаются качеством.

  3. Branching — уникальная возможность исследовать варианты.
     Каждая ветка изолирована, решения в одной не влияют на другую.
     Идеален для A/B сравнения, параллельных гипотез.
""")


def main():
    print(f"{'#' * 80}")
    print("  ТЕСТ СТРАТЕГИЙ УПРАВЛЕНИЯ КОНТЕКСТОМ")
    print(f"  Сценарий: сбор ТЗ на мобильное приложение ({len(SCENARIO)} сообщений)")
    print(f"  Модель: {MODEL_NAME}")
    print(f"{'#' * 80}")

    llm = OpenAIModel(model=MODEL_NAME)

    # Запуск всех трёх стратегий
    print("\n[1/3] Запуск Sliding Window...")
    sw_results = run_sliding_window(llm)

    print("\n[2/3] Запуск Sticky Facts...")
    sf_results = run_sticky_facts(llm)

    print("\n[3/3] Запуск Branching...")
    br_main, br_a, br_b = run_branching(llm)

    # Анализ
    analyze_results(sw_results, sf_results, br_main, br_a, br_b)


if __name__ == "__main__":
    main()
