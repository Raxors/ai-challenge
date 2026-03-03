"""
Автоматический тест 3-слойной системы памяти.

Проверяет ДВЕ вещи:
  1. Какие данные попадают в каждый слой (short-term, working, long-term)
  2. Как это влияет на ответы ассистента (recall, изоляция, контекст)

Фазы:
  1. Профиль пользователя  → long-term по категориям
  2. Задача A (API design)  → working memory с релевантным контентом
  3. Задача B (dashboard)   → изоляция working + long-term recall в ответах
  4. Возврат к задаче A     → восстановление working + контекст в ответах
"""

import os
import sys
import json

from dotenv import load_dotenv

load_dotenv()

from llm import OpenAIModel
from agent import Agent
from strategies.memory_layers import MemoryLayerStrategy
from storage import LONG_TERM_CATEGORIES
from core import LLMError


MODEL_NAME = "gpt-4o"

# ── Тестовые сценарии ─────────────────────────────

PHASE_1_PROFILE = [
    "Привет! Меня зовут Алексей, я senior backend разработчик в компании TechFlow.",
    "Я предпочитаю Python и FastAPI, использую PostgreSQL. Код пишу в PyCharm.",
    "Мы работаем по Scrum, спринты по 2 недели. Тимлид — Марина.",
]

PHASE_2_TASK_A = [
    "Давай спроектируем REST API для системы управления задачами (task manager). Нужны CRUD для задач и проектов.",
    "Задачи должны иметь статусы: todo, in_progress, review, done. Приоритет: low, medium, high, critical.",
    "Нужна фильтрация по статусу и приоритету, пагинация по 20 элементов. Авторизация через JWT.",
    "Добавим эндпоинт для статистики: количество задач по статусам за период.",
    "Какие модели данных ты предлагаешь для Task и Project?",
]

PHASE_3_TASK_B = [
    "Теперь другая задача — нужен дашборд для мониторинга серверов.",
    "Дашборд должен показывать CPU, RAM, диск и сеть в реальном времени. Обновление каждые 5 секунд.",
    "Используем WebSocket для push-обновлений. Бэкенд собирает метрики через psutil.",
    "Нужны алерты: если CPU > 90% или RAM > 85% — отправить уведомление в Telegram.",
]

PHASE_4_RETURN_A = [
    "Вернёмся к API task manager. Напомни, какие эндпоинты мы обсуждали?",
    "Добавим bulk-операции: массовое обновление статуса задач.",
]

W = 60  # print width


def header(title):
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def section(title):
    print(f"\n  {'─' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'─' * (W - 4)}")


def check(label, passed):
    mark = "✓" if passed else "✗"
    print(f"  {mark} {label}: {passed}")
    return passed


def dump_short_term(agent, strategy):
    """Инспектирует prepare_messages() — что реально уходит в LLM."""
    msgs = strategy.prepare_messages(agent.history)
    system_blocks = [m for m in msgs if m["role"] == "system"]
    non_system = [m for m in msgs if m["role"] != "system"]

    print(f"\n    System-блоки ({len(system_blocks)}):")
    for i, m in enumerate(system_blocks):
        tag = m["content"][:70].replace("\n", " ")
        print(f"      [{i}] {tag}...")

    print(f"    Non-system сообщений: {len(non_system)} (window_size={strategy.window_size})")
    for m in non_system[-3:]:
        role = "User" if m["role"] == "user" else "Asst"
        text = m["content"][:60].replace("\n", " ")
        print(f"      {role}: {text}...")

    return msgs, system_blocks, non_system


def dump_working_memory(strategy):
    wm = dict(strategy.working_memory)
    print(f"\n    Working memory (задача: {strategy.current_task_id}), ключей: {len(wm)}")
    for k, v in wm.items():
        print(f"      {k}: {str(v)[:70]}")
    return wm


def dump_long_term(strategy):
    lt = strategy.store.get_long_term_dict()
    total = sum(len(items) for items in lt.values())
    print(f"\n    Long-term memory, категорий: {len(lt)}, фактов: {total}")
    for cat, items in lt.items():
        print(f"      [{cat}] ({len(items)} фактов)")
        for k, v in items.items():
            print(f"        {k}: {str(v)[:60]}")
    return lt


def run_test():
    header("ТЕСТ: 3-слойная система памяти (Memory Layers)")

    llm = OpenAIModel(model=MODEL_NAME)
    agent = Agent(
        model=llm,
        model_name=MODEL_NAME,
        max_tokens=512,
        system_prompt="You are a helpful senior developer assistant. Answer concisely in Russian.",
        strategy_name="memory_layers",
        window_size=6,
    )

    strategy = agent.strategy
    assert isinstance(strategy, MemoryLayerStrategy)

    checks_all = []
    token_stats = {"input": 0, "output": 0}

    def send(msg):
        """Отправляет одно сообщение, возвращает result или None."""
        try:
            result = agent.ask(msg)
            m = result["metrics"]
            token_stats["input"] += m["request_input_tokens"]
            token_stats["output"] += m["request_output_tokens"]
            return result
        except LLMError as e:
            print(f"    ОШИБКА: {e}")
            return None

    def send_phase(phase_name, messages, task_id=None):
        section(f"Фаза: {phase_name}")
        if task_id:
            strategy.switch_task(task_id)
            print(f"  [Задача → {task_id}]")
        results = []
        for i, msg in enumerate(messages, 1):
            print(f"\n  [{i}/{len(messages)}] User: {msg[:55]}...")
            r = send(msg)
            if r:
                print(f"    → {r['text'][:75]}...")
            results.append(r)
        return results

    # ╔═══════════════════════════════════════════════╗
    # ║  ФАЗА 1: Профиль → Long-term по категориям   ║
    # ╚═══════════════════════════════════════════════╝

    send_phase("1. Профиль пользователя", PHASE_1_PROFILE)

    section("ПРОВЕРКА СЛОЁВ после фазы 1")

    # Short-term
    print("  [Short-term]")
    msgs_1, sys_1, nonsys_1 = dump_short_term(agent, strategy)
    checks_all.append(check(
        "Short-term: non-system <= window_size",
        len(nonsys_1) <= strategy.window_size,
    ))

    # Working memory — для default task должна быть пустой или минимальной
    print("  [Working memory]")
    wm_1 = dump_working_memory(strategy)

    # Long-term — главная проверка фазы 1
    print("  [Long-term]")
    lt_1 = dump_long_term(strategy)

    lt_1_all_values = " ".join(
        str(v).lower() for items in lt_1.values() for v in items.values()
    )
    lt_1_all_keys = " ".join(
        k.lower() for items in lt_1.values() for k in items.keys()
    )
    lt_1_text = lt_1_all_values + " " + lt_1_all_keys

    checks_all.append(check(
        "Long-term: имя (алексей/alexe) сохранено",
        "алексей" in lt_1_text or "alexe" in lt_1_text,
    ))
    checks_all.append(check(
        "Long-term: компания (techflow) сохранена",
        "techflow" in lt_1_text,
    ))
    checks_all.append(check(
        "Long-term: стек (python/fastapi/postgres) сохранён",
        any(kw in lt_1_text for kw in ["python", "fastapi", "postgres"]),
    ))
    checks_all.append(check(
        "Long-term: контакт (марина/marina) сохранён",
        "марин" in lt_1_text or "marina" in lt_1_text,
    ))
    checks_all.append(check(
        "Long-term: >= 3 фактов итого",
        sum(len(items) for items in lt_1.values()) >= 3,
    ))

    # Проверяем что long-term инжектится в prepare_messages
    lt_injected = any(
        "долгосрочная память" in m["content"].lower() or "long" in m["content"].lower()
        for m in sys_1
    )
    checks_all.append(check(
        "Short-term: long-term блок инжектирован в system",
        lt_injected,
    ))

    # ╔═══════════════════════════════════════════════╗
    # ║  ФАЗА 2: Задача A → Working memory           ║
    # ╚═══════════════════════════════════════════════╝

    send_phase("2. Задача A (API task manager)", PHASE_2_TASK_A, task_id="api_task_manager")

    section("ПРОВЕРКА СЛОЁВ после фазы 2")

    # Short-term
    print("  [Short-term]")
    msgs_2, sys_2, nonsys_2 = dump_short_term(agent, strategy)
    checks_all.append(check(
        "Short-term: окно ограничено (non-system <= window)",
        len(nonsys_2) <= strategy.window_size,
    ))
    # Working memory block injected?
    wm_injected = any(
        "рабочая память" in m["content"].lower() or "working" in m["content"].lower()
        for m in sys_2
    )
    checks_all.append(check(
        "Short-term: working memory блок инжектирован в system",
        wm_injected,
    ))

    # Working memory
    print("  [Working memory]")
    wm_a = dump_working_memory(strategy)
    wm_a_text = json.dumps(wm_a, ensure_ascii=False).lower()

    checks_all.append(check(
        "Working memory A: не пуста",
        len(wm_a) > 0,
    ))
    checks_all.append(check(
        "Working memory A: содержит API-контекст (api/crud/endpoint/task)",
        any(kw in wm_a_text for kw in ["api", "crud", "endpoint", "task", "эндпоинт", "задач"]),
    ))
    checks_all.append(check(
        "Working memory A: содержит статусы или приоритеты",
        any(kw in wm_a_text for kw in ["todo", "in_progress", "done", "status", "статус", "приоритет", "priority", "jwt"]),
    ))

    # Long-term should have grown
    print("  [Long-term]")
    lt_2 = dump_long_term(strategy)
    lt_2_count = sum(len(items) for items in lt_2.values())
    lt_1_count = sum(len(items) for items in lt_1.values())
    checks_all.append(check(
        f"Long-term: сохранён (>= фазы 1: {lt_2_count} >= {lt_1_count})",
        lt_2_count >= lt_1_count,
    ))

    # ╔═══════════════════════════════════════════════╗
    # ║  ФАЗА 3: Задача B → Изоляция + LT recall     ║
    # ╚═══════════════════════════════════════════════╝

    # Сохраняем working memory A для сравнения
    wm_a_snapshot = dict(strategy.working_memory)

    send_phase("3. Задача B (Мониторинг серверов)", PHASE_3_TASK_B, task_id="monitoring_dashboard")

    section("ПРОВЕРКА СЛОЁВ после фазы 3")

    # Short-term
    print("  [Short-term]")
    msgs_3, sys_3, nonsys_3 = dump_short_term(agent, strategy)

    # Working memory B — должна быть про дашборд, не про API
    print("  [Working memory]")
    wm_b = dump_working_memory(strategy)
    wm_b_text = json.dumps(wm_b, ensure_ascii=False).lower()

    checks_all.append(check(
        "Working memory B: не пуста",
        len(wm_b) > 0,
    ))
    checks_all.append(check(
        "Working memory B: содержит dashboard-контекст (cpu/ram/websocket/мониторинг/dashboard)",
        any(kw in wm_b_text for kw in ["cpu", "ram", "websocket", "мониторинг", "dashboard", "дашборд", "psutil", "алерт", "alert"]),
    ))
    checks_all.append(check(
        "Изоляция: working B не содержит crud/endpoint задачи A",
        "crud" not in wm_b_text and "endpoint" not in wm_b_text and "эндпоинт" not in wm_b_text,
    ))

    # Long-term preserved
    print("  [Long-term]")
    lt_3 = dump_long_term(strategy)
    lt_3_text = " ".join(
        str(v).lower() for items in lt_3.values() for v in items.values()
    )
    checks_all.append(check(
        "Long-term: профиль сохранён между задачами (алексей/techflow)",
        ("алексей" in lt_3_text or "alexe" in lt_3_text) and "techflow" in lt_3_text,
    ))

    # Влияние на ответы: последний ответ фазы 3 — ассистент знает стек пользователя?
    section("ВЛИЯНИЕ НА ОТВЕТЫ (фаза 3)")
    print("  Проверяем: ассистент помнит профиль через long-term?")
    probe_result = send("Напомни, какой у меня стек технологий и какую IDE я использую?")
    if probe_result:
        probe_text = probe_result["text"].lower()
        print(f"    Ответ: {probe_result['text'][:120]}...")
        recall_ok = any(kw in probe_text for kw in ["python", "fastapi", "postgres", "pycharm"])
        checks_all.append(check(
            "Ответ в task B: ассистент вспомнил стек из long-term (python/fastapi/pycharm)",
            recall_ok,
        ))
    else:
        checks_all.append(check("Ответ в task B: запрос не удался", False))

    # ╔═══════════════════════════════════════════════╗
    # ║  ФАЗА 4: Возврат к задаче A → Restore + Ответ║
    # ╚═══════════════════════════════════════════════╝

    phase4_results = send_phase("4. Возврат к задаче A", PHASE_4_RETURN_A, task_id="api_task_manager")

    section("ПРОВЕРКА СЛОЁВ после фазы 4")

    # Short-term
    print("  [Short-term]")
    msgs_4, sys_4, nonsys_4 = dump_short_term(agent, strategy)

    # Working memory — должна быть восстановлена
    print("  [Working memory]")
    wm_restored = dump_working_memory(strategy)
    wm_restored_text = json.dumps(wm_restored, ensure_ascii=False).lower()

    checks_all.append(check(
        "Working memory A: восстановлена (не пуста)",
        len(wm_restored) > 0,
    ))
    checks_all.append(check(
        "Working memory A: содержит оригинальный контекст (api/crud/task)",
        any(kw in wm_restored_text for kw in ["api", "crud", "endpoint", "task", "эндпоинт", "задач"]),
    ))

    # Влияние на ответы: ассистент помнит контекст задачи A
    section("ВЛИЯНИЕ НА ОТВЕТЫ (фаза 4)")
    if phase4_results and phase4_results[0]:
        resp = phase4_results[0]["text"].lower()
        print(f"  Ответ на 'Напомни эндпоинты': {phase4_results[0]['text'][:120]}...")
        api_recall = any(kw in resp for kw in [
            "crud", "api", "task", "endpoint", "эндпоинт",
            "статус", "статистик", "jwt", "фильтрац", "пагинац",
        ])
        checks_all.append(check(
            "Ответ: ассистент вспомнил API-контекст задачи A",
            api_recall,
        ))
    else:
        checks_all.append(check("Ответ фазы 4: запрос не удался", False))

    # Проверяем что ответ НЕ путает с task B
    if phase4_results and phase4_results[0]:
        resp = phase4_results[0]["text"].lower()
        no_confusion = not any(kw in resp for kw in ["cpu", "ram", "websocket", "дашборд", "мониторинг", "psutil"])
        checks_all.append(check(
            "Ответ: не путает с задачей B (нет cpu/ram/websocket)",
            no_confusion,
        ))

    # Long-term
    print("  [Long-term]")
    lt_4 = dump_long_term(strategy)

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  ИТОГИ ПО ЗАДАЧАМ                                       ║
    # ╚═══════════════════════════════════════════════════════════╝

    classification_tokens = strategy.classifier.classification_tokens
    total_main = token_stats["input"] + token_stats["output"]
    overhead_pct = (classification_tokens / total_main * 100) if total_main > 0 else 0
    lt_final = lt_4
    lt_final_total = sum(len(items) for items in lt_final.values())
    lt_3_count = sum(len(i) for i in lt_3.values())

    # ══════════════════════════════════════════════════
    # ЗАДАНИЕ 1: Разделите информацию минимум на 3 типа
    # ══════════════════════════════════════════════════

    header("ЗАДАНИЕ 1: Информация разделена на 3 типа")

    # --- Тип 1: Краткосрочная ---
    section("Тип 1: Краткосрочная (текущий диалог)")
    print(f"""
  Что это:   Последние N сообщений из history (sliding window).
  Где хранится: list в памяти процесса (Agent.history).
  Источник:  Каждое сообщение user/assistant попадает напрямую.
  Время жизни: Только текущая сессия, обрезается по window_size={strategy.window_size}.

  Содержимое по фазам (non-system сообщений в окне):
    Фаза 1 (профиль):     {len(nonsys_1)} из {len(PHASE_1_PROFILE)*2} (user+assistant)
    Фаза 2 (task A):      {len(nonsys_2)} из {len(PHASE_2_TASK_A)*2}
    Фаза 3 (task B):      {len(nonsys_3)} из {len(PHASE_3_TASK_B)*2}
    Фаза 4 (возврат A):   {len(nonsys_4)} из {len(PHASE_4_RETURN_A)*2}""")
    all_bounded = all(
        n <= strategy.window_size
        for n in [len(nonsys_1), len(nonsys_2), len(nonsys_3), len(nonsys_4)]
    )
    print(f"\n  ✓ Окно ограничено ({strategy.window_size}): {all_bounded}")

    # --- Тип 2: Рабочая ---
    section("Тип 2: Рабочая (данные текущей задачи)")
    print(f"""
  Что это:   key-value словарь с контекстом ТЕКУЩЕЙ задачи.
  Где хранится: SQLite таблица working_memory (по task_id).
  Источник:  MemoryClassifier выбирает, что сохранить в working_memory.
  Время жизни: Пока задача активна. При switch_task() — сохраняется и
               загружается другая. При reset() — очищается.

  Задача A (api_task_manager) — {len(wm_a)} ключей:""")
    for k, v in wm_a.items():
        print(f"    {k}: {str(v)[:65]}")

    print(f"\n  Задача B (monitoring_dashboard) — {len(wm_b)} ключей:")
    for k, v in wm_b.items():
        print(f"    {k}: {str(v)[:65]}")

    print(f"\n  Задача A (восстановлена после возврата) — {len(wm_restored)} ключей:")
    for k, v in wm_restored.items():
        print(f"    {k}: {str(v)[:65]}")

    wm_a_keys = set(wm_a.keys())
    wm_restored_keys = set(wm_restored.keys())
    preserved_keys = wm_a_keys & wm_restored_keys
    print(f"\n  Ключи из фазы 2, сохранившиеся после round-trip: "
          f"{len(preserved_keys)}/{len(wm_a_keys)}")

    isolation_leaked = [kw for kw in ["crud", "endpoint", "эндпоинт", "jwt", "пагинац"]
                        if kw in wm_b_text]
    print(f"  Изоляция (task A не утёк в task B): "
          f"{'✓ чисто' if not isolation_leaked else '✗ утечка: ' + str(isolation_leaked)}")

    # --- Тип 3: Долговременная ---
    section("Тип 3: Долговременная (профиль, решения, знания)")
    print(f"""
  Что это:   Факты о пользователе, полезные МЕЖДУ задачами.
  Где хранится: SQLite таблица long_term_memory (category + key + value).
  Источник:  MemoryClassifier выбирает, что попадает в long_term_updates.
             Валидация: category должна быть из {', '.join(LONG_TERM_CATEGORIES)}.
  Время жизни: Перманентно. Переживает reset(), switch_task(), перезапуск.

  Рост по фазам: {lt_1_count} → {lt_2_count} → {lt_3_count} → {lt_final_total} фактов""")

    for cat, items in lt_final.items():
        print(f"\n  [{cat}] ({len(items)} фактов):")
        for k, v in items.items():
            print(f"    {k}: {str(v)[:65]}")

    # ══════════════════════════════════════════════════
    # ЗАДАНИЕ 2: Хранятся отдельно + явный выбор
    # ══════════════════════════════════════════════════

    header("ЗАДАНИЕ 2: Раздельное хранение + явный выбор")

    section("Раздельное хранение")
    print(f"""
  Слой               Хранилище                   Таблица/структура
  ──────────────     ─────────────────────       ────────────────────
  Краткосрочная      RAM (Agent.history)          list[dict]
  Рабочая            SQLite (memory.db)           working_memory (task_id -> JSON)
  Долговременная     SQLite (memory.db)           long_term_memory (category+key -> value)

  Три слоя никогда не смешиваются:
    - history — это полная лента диалога, хранится в HistoryStore (chat_history.db)
    - working_memory — dict по task_id, изолирован между задачами
    - long_term_memory — category/key/value, общий для всех задач""")

    section("Явный выбор: кто решает что куда")
    print(f"""
  После КАЖДОГО ответа ассистента вызывается:
    MemoryClassifier.classify(recent_msgs, working_memory, long_term)

  Классификатор (LLM) получает:
    - последние 4 сообщения
    - текущее состояние working_memory
    - текущее состояние long_term_memory

  И возвращает JSON с ЯВНЫМ разделением:
    {{
      "working_memory": {{...}},        ← что обновить в рабочей памяти
      "long_term_updates": [...]         ← что добавить в долговременную
      "task_change_detected": bool       ← сменилась ли задача
    }}

  Валидация перед записью:
    - working_memory: проверка что dict
    - long_term_updates: каждый элемент должен содержать category/key/value,
      category должна быть из допустимых: {', '.join(LONG_TERM_CATEGORIES)}

  Токены потрачены на классификацию: {classification_tokens}
  Overhead: {overhead_pct:.1f}% от основного трафика""")

    lt_block_in_prompt = any(
        "долгосрочная память" in m["content"].lower() for m in sys_4
    )
    wm_block_in_prompt = any(
        "рабочая память" in m["content"].lower() for m in sys_4
    )
    print(f"\n  Инжекция в prompt (prepare_messages):")
    print(f"    Краткосрочная: последние {strategy.window_size} non-system сообщений")
    print(f"    Рабочая:       {'✓' if wm_block_in_prompt else '✗'} system-блок '[Рабочая память — текущая задача: ...]'")
    print(f"    Долговременная: {'✓' if lt_block_in_prompt else '✗'} system-блок '[Долгосрочная память — устойчивые факты...]'")

    # ══════════════════════════════════════════════════
    # ЗАДАНИЕ 3: Проверка данных + влияние на ответы
    # ══════════════════════════════════════════════════

    header("ЗАДАНИЕ 3: Проверка данных + влияние на ответы")

    # --- 3а: Какие данные попадают в каждый слой ---
    section("3а. Какие данные попадают в каждый слой")

    print(f"\n  Краткосрочная:")
    print(f"    Попадает: каждое user/assistant сообщение напрямую")
    print(f"    Окно: {strategy.window_size} последних, старые отбрасываются")
    print(f"    Пример (последние из фазы 4):")
    for m in nonsys_4[-2:]:
        role = "User" if m["role"] == "user" else "Asst"
        print(f"      {role}: {m['content'][:60]}...")

    print(f"\n  Рабочая:")
    print(f"    Попадает: контекст текущей задачи (цели, решения, требования)")
    print(f"    Классификатор извлёк для задачи A: {list(wm_a.keys())}")
    print(f"    Классификатор извлёк для задачи B: {list(wm_b.keys())}")
    wm_a_has_api = any(kw in wm_a_text for kw in ["api", "crud", "endpoint", "task", "эндпоинт", "задач"])
    wm_b_has_dash = any(kw in wm_b_text for kw in ["cpu", "ram", "websocket", "мониторинг", "dashboard", "дашборд", "psutil", "алерт", "alert"])
    print(f"    Task A содержит API-контекст: {'✓' if wm_a_has_api else '✗'}")
    print(f"    Task B содержит dashboard-контекст: {'✓' if wm_b_has_dash else '✗'}")
    print(f"    Task B НЕ содержит данных task A: {'✓' if not isolation_leaked else '✗'}")

    print(f"\n  Долговременная:")
    print(f"    Попадает: профиль, предпочтения, контакты, знания, решения")
    lt_1_text_final = " ".join(
        str(v).lower() for items in lt_final.values() for v in items.values()
    ) + " " + " ".join(
        k.lower() for items in lt_final.values() for k in items.keys()
    )
    has_name = "алексей" in lt_1_text_final or "alexe" in lt_1_text_final
    has_company = "techflow" in lt_1_text_final
    has_stack = any(kw in lt_1_text_final for kw in ["python", "fastapi", "postgres"])
    has_contact = "марин" in lt_1_text_final or "marina" in lt_1_text_final
    print(f"    Имя (Алексей): {'✓' if has_name else '✗'}")
    print(f"    Компания (TechFlow): {'✓' if has_company else '✗'}")
    print(f"    Стек (Python/FastAPI/PostgreSQL): {'✓' if has_stack else '✗'}")
    print(f"    Контакт (Марина): {'✓' if has_contact else '✗'}")
    print(f"    Категории: {list(lt_final.keys())}")

    # --- 3б: Как это влияет на ответы ассистента ---
    section("3б. Как это влияет на ответы ассистента")

    # Тест 1: Long-term recall из другой задачи
    print(f"\n  Тест 1: Long-term recall в чужой задаче")
    print(f"    Ситуация: ассистент в задаче B (мониторинг),")
    print(f"              профиль установлен в фазе 1 (другой контекст)")
    print(f"    Вопрос: 'Напомни, какой у меня стек и какую IDE я использую?'")
    if probe_result:
        probe_resp = probe_result["text"]
        found_kw = [kw for kw in ["python", "fastapi", "postgres", "pycharm"]
                    if kw in probe_resp.lower()]
        print(f"    Ответ: {probe_resp[:130]}")
        print(f"    Ключевые слова из long-term в ответе: {found_kw}")
        if found_kw:
            print(f"    ✓ ВЫВОД: Long-term ВЛИЯЕТ — ассистент помнит профиль между задачами")
        else:
            print(f"    ✗ ВЫВОД: Long-term НЕ повлиял на ответ")
    else:
        print(f"    ✗ Запрос не удался")

    # Тест 2: Working memory restore
    print(f"\n  Тест 2: Working memory restore после переключения задач")
    print(f"    Ситуация: ассистент вернулся к задаче A после работы над B,")
    print(f"              working memory A восстановлена из SQLite")
    print(f"    Вопрос: 'Напомни, какие эндпоинты мы обсуждали?'")
    if phase4_results and phase4_results[0]:
        resp_text = phase4_results[0]["text"]
        found_api = [kw for kw in ["crud", "api", "endpoint", "эндпоинт", "статус",
                                    "статистик", "jwt", "фильтрац", "пагинац", "task"]
                     if kw in resp_text.lower()]
        found_b = [kw for kw in ["cpu", "ram", "websocket", "дашборд", "мониторинг"]
                   if kw in resp_text.lower()]
        print(f"    Ответ: {resp_text[:130]}")
        print(f"    Ключевые слова задачи A в ответе: {found_api}")
        print(f"    Ключевые слова задачи B в ответе: {found_b}")
        if found_api and not found_b:
            print(f"    ✓ ВЫВОД: Working memory ВЛИЯЕТ — контекст задачи A восстановлен,")
            print(f"             задача B не смешивается")
        elif found_api and found_b:
            print(f"    ~ ВЫВОД: Контекст восстановлен, но есть примесь из задачи B")
        else:
            print(f"    ✗ ВЫВОД: Working memory НЕ повлияла на ответ")
    else:
        print(f"    ✗ Запрос не удался")

    # Тест 3: Изоляция — задача B не знает про задачу A
    print(f"\n  Тест 3: Изоляция — ответы задачи B не содержат контекст задачи A")
    print(f"    Ситуация: в working memory задачи B нет данных задачи A")
    print(f"    Данные task A, попавшие в task B: {isolation_leaked if isolation_leaked else 'ничего'}")
    if not isolation_leaked:
        print(f"    ✓ ВЫВОД: Рабочая память изолирована между задачами")
    else:
        print(f"    ✗ ВЫВОД: Изоляция нарушена")

    # ══════════════════════════════════════════════════
    # Итоговая сводка
    # ══════════════════════════════════════════════════

    header("СВОДКА")

    print(f"\n  Статистика:")
    total_msgs = sum(len(m) for m in [PHASE_1_PROFILE, PHASE_2_TASK_A, PHASE_3_TASK_B, PHASE_4_RETURN_A]) + 1
    print(f"    Сообщений:               {total_msgs}")
    print(f"    Input токенов:           {token_stats['input']}")
    print(f"    Output токенов:          {token_stats['output']}")
    print(f"    Токены на классификацию: {classification_tokens} ({overhead_pct:.1f}% overhead)")

    section("Все проверки")
    passed = sum(checks_all)
    total = len(checks_all)

    print(f"\n  Пройдено: {passed}/{total}")
    if passed == total:
        print(f"  ✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print(f"  ✗ Не пройдены: {total - passed} проверок")

    print(f"\n{'═' * W}")

    # Cleanup
    agent.close()
    for f in ("memory.db", "chat_history.db"):
        if os.path.exists(f):
            os.remove(f)

    return passed == total


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
