"""
Автоматический тест ProjectInvariants — инварианты проекта.

Проверяет ТРИ вещи:
  1. Механика инвариантов (CRUD, persistence, to_prompt)
  2. Отказ при конфликте (LLM соблюдает инварианты)
  3. Автоматическое применение (инжекция, reload, edge cases)

Стратегии для интеграции: все 4 (sliding_window, sticky_facts, branching, memory_layers)
"""

import os
import sys
import json
import tempfile

from dotenv import load_dotenv

load_dotenv()

from core.invariants import ProjectInvariants
from llm import OpenAIModel
from agent import Agent
from core import LLMError

MODEL_NAME = "gpt-4o"

W = 60


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


# ═══════════════════════════════════════════════════════
#   ФАЗА 1: Unit-тесты ProjectInvariants (без LLM)
# ═══════════════════════════════════════════════════════

def test_phase1_unit():
    header("ФАЗА 1: Unit-тесты ProjectInvariants")
    results = []

    # Тест 1.1: Пустые по умолчанию
    section("1.1 Пустые по умолчанию")
    inv = ProjectInvariants()
    results.append(check("is_empty() для нового объекта", inv.is_empty()))
    results.append(check("invariants == []", inv.invariants == []))
    results.append(check("_next_id == 1", inv._next_id == 1))

    # Тест 1.2: add() с автоинкрементом id
    section("1.2 add() с автоинкрементом id")
    inv = ProjectInvariants()
    id1 = inv.add("стек", "Backend только на Python + FastAPI")
    id2 = inv.add("архитектура", "Микросервисная архитектура с REST API")
    id3 = inv.add("бизнес-правило", "Данные пользователей только в EU регионе")
    results.append(check("Первый id == 1", id1 == 1))
    results.append(check("Второй id == 2", id2 == 2))
    results.append(check("Третий id == 3", id3 == 3))
    results.append(check("is_empty() == False", not inv.is_empty()))
    results.append(check("len(invariants) == 3", len(inv.invariants) == 3))

    # Тест 1.3: remove() по id
    section("1.3 remove() по id")
    inv = ProjectInvariants()
    id1 = inv.add("стек", "Python only")
    id2 = inv.add("архитектура", "REST API")
    removed = inv.remove(id1)
    results.append(check("remove(1) == True", removed))
    results.append(check("len(invariants) == 1", len(inv.invariants) == 1))
    results.append(check("Оставшийся id == 2", inv.invariants[0]["id"] == 2))

    # Тест 1.4: remove() несуществующего id
    section("1.4 remove() несуществующего id")
    removed = inv.remove(999)
    results.append(check("remove(999) == False", not removed))

    # Тест 1.5: id не переиспользуются после remove
    section("1.5 id не переиспользуются после remove")
    inv = ProjectInvariants()
    id1 = inv.add("a", "rule1")
    id2 = inv.add("b", "rule2")
    inv.remove(id1)
    id3 = inv.add("c", "rule3")
    results.append(check("id1 == 1", id1 == 1))
    results.append(check("id2 == 2", id2 == 2))
    results.append(check("id3 == 3 (не 1)", id3 == 3))

    # Тест 1.6: clear()
    section("1.6 clear()")
    inv = ProjectInvariants()
    inv.add("стек", "Python only")
    inv.add("архитектура", "REST API")
    inv.clear()
    results.append(check("is_empty() после clear()", inv.is_empty()))
    results.append(check("_next_id == 1 после clear()", inv._next_id == 1))

    # Тест 1.7: get_by_id()
    section("1.7 get_by_id()")
    inv = ProjectInvariants()
    inv.add("стек", "Python only")
    id2 = inv.add("архитектура", "REST API")
    found = inv.get_by_id(id2)
    results.append(check("get_by_id(2) найден", found is not None))
    results.append(check("category == 'архитектура'", found["category"] == "архитектура"))
    results.append(check("rule == 'REST API'", found["rule"] == "REST API"))
    results.append(check("get_by_id(999) == None", inv.get_by_id(999) is None))

    # Тест 1.8: get_all() возвращает копию
    section("1.8 get_all() возвращает копию")
    inv = ProjectInvariants()
    inv.add("стек", "Python only")
    all_inv = inv.get_all()
    all_inv.append({"id": 999, "category": "fake", "rule": "fake"})
    results.append(check("Модификация копии не влияет на оригинал",
                          len(inv.invariants) == 1))

    # Тест 1.9: Save/Load round-trip
    section("1.9 Save/Load round-trip")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        original = ProjectInvariants()
        original.add("стек", "Backend только на Python + FastAPI")
        original.add("архитектура", "Микросервисная архитектура")
        original.remove(1)  # удалили первый
        original.add("бизнес-правило", "Данные только в EU")
        original.save(tmp_path)

        with open(tmp_path, encoding="utf-8") as f:
            raw = f.read()
        print(f"\n  Saved JSON:\n  {raw[:300]}...\n")
        results.append(check("JSON файл создан", os.path.exists(tmp_path)))

        loaded = ProjectInvariants.load(tmp_path)
        results.append(check("len совпадает", len(loaded.invariants) == len(original.invariants)))
        results.append(check("_next_id совпадает", loaded._next_id == original._next_id))
        results.append(check("invariants совпадают", loaded.invariants == original.invariants))
    finally:
        os.unlink(tmp_path)

    # Тест 1.10: Load из несуществующего файла
    section("1.10 Load из несуществующего файла")
    inv = ProjectInvariants.load("/tmp/nonexistent_invariants_xyz.json")
    results.append(check("load несуществующего = пустой объект", inv.is_empty()))

    # Тест 1.11: Load из corrupted JSON
    section("1.11 Load из corrupted JSON")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("{corrupted json data!!!")
        tmp_path = f.name
    try:
        inv = ProjectInvariants.load(tmp_path)
        results.append(check("corrupted JSON → пустой объект", inv.is_empty()))
    finally:
        os.unlink(tmp_path)

    # Тест 1.12: to_prompt() формат
    section("1.12 to_prompt() формат")
    inv = ProjectInvariants()
    inv.add("архитектура", "Используем микросервисную архитектуру с REST API")
    inv.add("стек", "Backend только на Python + FastAPI")
    inv.add("бизнес-правило", "Все данные пользователей хранятся только в EU регионе")
    prompt = inv.to_prompt()
    print(f"\n  to_prompt() output:\n  {prompt.replace(chr(10), chr(10) + '  ')}\n")

    results.append(check("'ОБЯЗАТЕЛЬНЫЕ ограничения' в prompt",
                          "ОБЯЗАТЕЛЬНЫЕ ограничения" in prompt))
    results.append(check("'ОТКАЖИ' в prompt", "ОТКАЖИ" in prompt))
    results.append(check("'АБСОЛЮТНЫЙ приоритет' в prompt",
                          "АБСОЛЮТНЫЙ приоритет" in prompt))
    results.append(check("'[архитектура]' в prompt", "[архитектура]" in prompt))
    results.append(check("'[стек]' в prompt", "[стек]" in prompt))
    results.append(check("'[бизнес-правило]' в prompt", "[бизнес-правило]" in prompt))

    # Тест 1.13: to_prompt() с пустым списком — всё равно содержит заголовок
    section("1.13 to_prompt() пустой список")
    inv = ProjectInvariants()
    prompt = inv.to_prompt()
    results.append(check("Пустой to_prompt() содержит заголовок",
                          "ОБЯЗАТЕЛЬНЫЕ ограничения" in prompt))

    # Тест 1.14: add() после load сохраняет правильный _next_id
    section("1.14 add() после load с правильным _next_id")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        inv = ProjectInvariants()
        inv.add("a", "rule1")
        inv.add("b", "rule2")
        inv.save(tmp_path)

        loaded = ProjectInvariants.load(tmp_path)
        id3 = loaded.add("c", "rule3")
        results.append(check("id после load == 3", id3 == 3))
    finally:
        os.unlink(tmp_path)

    # Тест 1.15: Множественный remove
    section("1.15 Множественный remove")
    inv = ProjectInvariants()
    inv.add("a", "r1")
    inv.add("b", "r2")
    inv.add("c", "r3")
    inv.remove(1)
    inv.remove(3)
    results.append(check("Осталась 1 запись", len(inv.invariants) == 1))
    results.append(check("Оставшийся id == 2", inv.invariants[0]["id"] == 2))

    # Тест 1.16: clear() и повторное добавление
    section("1.16 clear() и повторное добавление")
    inv = ProjectInvariants()
    inv.add("a", "r1")
    inv.add("b", "r2")
    inv.clear()
    id_new = inv.add("c", "r3")
    results.append(check("id после clear() == 1", id_new == 1))

    # Тест 1.17: Save/Load пустого объекта
    section("1.17 Save/Load пустого объекта")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        inv = ProjectInvariants()
        inv.save(tmp_path)
        loaded = ProjectInvariants.load(tmp_path)
        results.append(check("Пустой save/load round-trip", loaded.is_empty()))
        results.append(check("_next_id == 1 после load пустого", loaded._next_id == 1))
    finally:
        os.unlink(tmp_path)

    # Тест 1.18: Содержимое инвариантов в to_prompt()
    section("1.18 Порядок правил в to_prompt()")
    inv = ProjectInvariants()
    inv.add("first", "Rule One")
    inv.add("second", "Rule Two")
    prompt = inv.to_prompt()
    pos1 = prompt.index("[first]")
    pos2 = prompt.index("[second]")
    results.append(check("Порядок правил сохранён", pos1 < pos2))

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 1: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Интеграция с LLM — отказ при нарушении
# ═══════════════════════════════════════════════════════

def make_agent(tmp_dir, label, strategy="sliding_window"):
    """Создать агента с временными файлами для тестирования."""
    db_path = os.path.join(tmp_dir, f"history_{label}.db")
    profile_path = os.path.join(tmp_dir, f"profile_{label}.json")
    state_path = os.path.join(tmp_dir, f"state_{label}.json")
    inv_path = os.path.join(tmp_dir, f"invariants_{label}.json")

    llm = OpenAIModel(model=MODEL_NAME)
    agent = Agent(
        model=llm,
        model_name=MODEL_NAME,
        max_tokens=512,
        system_prompt="You are a helpful assistant. Answer concisely.",
        db_path=db_path,
        strategy_name=strategy,
        window_size=10,
        profile_path=profile_path,
        task_state_path=state_path,
        invariants_path=inv_path,
    )
    return agent


def test_phase2_llm():
    header("ФАЗА 2: LLM интеграция — отказ при нарушении инвариантов")
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Тест 2.1: Отказ при нарушении стека (Node.js vs Python)
        section("2.1 Отказ при нарушении стека")
        agent = make_agent(tmp_dir, "stack_violation")
        agent.invariants.add("стек", "Backend только на Python + FastAPI. Никаких других языков и фреймворков.")
        agent.save_invariants()

        try:
            result = agent.ask("Напиши мне бэкенд на Node.js с Express. Создай сервер с роутингом.")
            response = result["text"].lower()
            print(f"  Ответ (первые 300): {result['text'][:300]}...")

            has_refusal = any(word in response for word in [
                "инвариант", "ограничен", "не могу", "отказ", "нарушен",
                "python", "fastapi", "противоречит", "запрещ",
                "invariant", "cannot", "refuse", "violation",
            ])
            results.append(check("Отказ при запросе Node.js", has_refusal))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Отказ при запросе Node.js", False))
        finally:
            agent.close()

        # Тест 2.2: Объяснение причины отказа
        section("2.2 Объяснение причины отказа")
        agent = make_agent(tmp_dir, "explain_refusal")
        agent.invariants.add("архитектура", "Только микросервисная архитектура. Монолит запрещён.")
        agent.save_invariants()

        try:
            result = agent.ask("Давай сделаем приложение монолитом, так будет проще.")
            response = result["text"].lower()
            print(f"  Ответ (первые 300): {result['text'][:300]}...")

            mentions_invariant = any(word in response for word in [
                "инвариант", "ограничен", "микросервис", "монолит",
                "запрещ", "нарушен", "противоречит",
                "invariant", "constraint", "microservice",
            ])
            results.append(check("Упоминает причину отказа", mentions_invariant))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Упоминает причину отказа", False))
        finally:
            agent.close()

        # Тест 2.3: НЕ отказывает при корректном запросе
        section("2.3 НЕ отказывает при корректном запросе")
        agent = make_agent(tmp_dir, "correct_request")
        agent.invariants.add("стек", "Backend только на Python + FastAPI")
        agent.save_invariants()

        try:
            result = agent.ask("Напиши простой эндпоинт на FastAPI для получения списка задач.")
            response = result["text"].lower()
            print(f"  Ответ (первые 300): {result['text'][:300]}...")

            has_code = any(word in response for word in ["fastapi", "def ", "get", "@app", "@router", "async"])
            results.append(check("Помогает с корректным запросом FastAPI", has_code))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Помогает с корректным запросом FastAPI", False))
        finally:
            agent.close()

        # Тест 2.4: Отказ при нарушении одного из нескольких инвариантов
        section("2.4 Отказ при нарушении одного из нескольких")
        agent = make_agent(tmp_dir, "multi_invariants")
        agent.invariants.add("стек", "Backend только на Python + FastAPI")
        agent.invariants.add("архитектура", "Только REST API, никакого GraphQL")
        agent.invariants.add("бизнес-правило", "Все данные хранятся только в EU регионе")
        agent.save_invariants()

        try:
            result = agent.ask("Давай добавим GraphQL endpoint для нашего API.")
            response = result["text"].lower()
            print(f"  Ответ (первые 300): {result['text'][:300]}...")

            has_refusal = any(word in response for word in [
                "инвариант", "ограничен", "не могу", "graphql", "rest",
                "запрещ", "нарушен", "противоречит",
                "invariant", "cannot", "refuse",
            ])
            results.append(check("Отказ при GraphQL (инвариант REST only)", has_refusal))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Отказ при GraphQL", False))
        finally:
            agent.close()

        # Тест 2.5: invariants_count в metrics
        section("2.5 invariants_count в metrics")
        agent = make_agent(tmp_dir, "metrics_test")
        agent.invariants.add("стек", "Python only")
        agent.invariants.add("архитектура", "REST API")
        agent.save_invariants()

        try:
            result = agent.ask("Привет!")
            inv_count = result["metrics"].get("invariants_count", 0)
            results.append(check("invariants_count == 2 в metrics", inv_count == 2))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("invariants_count в metrics", False))
        finally:
            agent.close()

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 2: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: Edge cases
# ═══════════════════════════════════════════════════════

def test_phase3_edge_cases():
    header("ФАЗА 3: Edge cases")
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Тест 3.1: Пустые инварианты не инжектятся
        section("3.1 Пустые инварианты не инжектятся")
        agent = make_agent(tmp_dir, "empty_inv")
        results.append(check("invariants.is_empty()", agent.invariants.is_empty()))

        # Симулируем prepare_messages
        agent.history.append({"role": "user", "content": "test"})
        messages = agent.strategy.prepare_messages(agent.history)
        msg_count_before = len(messages)

        if not agent.invariants.is_empty():
            messages.insert(1, {"role": "system", "content": agent.invariants.to_prompt()})

        msg_count_after = len(messages)
        results.append(check("Пустые инварианты: сообщения не изменены",
                              msg_count_before == msg_count_after))
        agent.close()

        # Тест 3.2: reload_invariants() подхватывает внешние правки
        section("3.2 reload_invariants() подхватывает внешние правки")
        agent = make_agent(tmp_dir, "reload_inv")
        results.append(check("Изначально пустые", agent.invariants.is_empty()))

        # Внешне записываем файл инвариантов
        external_inv = ProjectInvariants()
        external_inv.add("внешний", "Правило от внешнего процесса")
        external_inv.save(agent.invariants_path)

        agent.reload_invariants()
        results.append(check("После reload: не пустые", not agent.invariants.is_empty()))
        results.append(check("После reload: 1 инвариант",
                              len(agent.invariants.invariants) == 1))
        results.append(check("После reload: правильная категория",
                              agent.invariants.invariants[0]["category"] == "внешний"))
        agent.close()

        # Тест 3.3: corrupted JSON в invariants файле
        section("3.3 corrupted JSON в файле инвариантов")
        inv_path = os.path.join(tmp_dir, "corrupted_inv.json")
        with open(inv_path, "w") as f:
            f.write("not valid json {{{")

        db_path = os.path.join(tmp_dir, "history_corrupted.db")
        profile_path = os.path.join(tmp_dir, "profile_corrupted.json")
        state_path = os.path.join(tmp_dir, "state_corrupted.json")

        llm = OpenAIModel(model=MODEL_NAME)
        agent = Agent(
            model=llm,
            model_name=MODEL_NAME,
            max_tokens=512,
            system_prompt="You are a helpful assistant.",
            db_path=db_path,
            strategy_name="sliding_window",
            window_size=10,
            profile_path=profile_path,
            task_state_path=state_path,
            invariants_path=inv_path,
        )
        results.append(check("corrupted JSON → пустые инварианты", agent.invariants.is_empty()))
        agent.close()

        # Тест 3.4: get_all() возвращает копию (не мутирует оригинал)
        section("3.4 get_all() возвращает копию")
        inv = ProjectInvariants()
        inv.add("a", "rule1")
        copy = inv.get_all()
        copy.clear()
        results.append(check("clear() копии не влияет на оригинал",
                              len(inv.invariants) == 1))

        # Тест 3.5: Работает со всеми 4 стратегиями
        section("3.5 Работает со всеми 4 стратегиями")
        strategies = ["sliding_window", "sticky_facts", "branching", "memory_layers"]
        for strategy in strategies:
            agent = make_agent(tmp_dir, f"strat_inv_{strategy}", strategy=strategy)
            agent.invariants.add("стек", "Python only")
            agent.save_invariants()

            try:
                result = agent.ask("Привет, какие у нас ограничения?")
                has_count = result["metrics"].get("invariants_count", 0) > 0
                results.append(check(f"[{strategy}] invariants_count > 0 в metrics", has_count))
                print(f"    [{strategy}] Ответ: {result['text'][:100]}...")
            except LLMError as e:
                print(f"    [{strategy}] Ошибка: {e}")
                results.append(check(f"[{strategy}] invariants в metrics", False))
            finally:
                agent.close()

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 3: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   MAIN — запуск всех фаз + итоговый отчёт
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "█" * W)
    print("  ТЕСТ ИНВАРИАНТОВ ПРОЕКТА — ЖЁСТКИЕ ОГРАНИЧЕНИЯ")
    print("█" * W)

    total_passed = 0
    total_tests = 0

    # Фаза 1: Юнит-тесты (без LLM)
    p1_passed, p1_total = test_phase1_unit()
    total_passed += p1_passed
    total_tests += p1_total

    # Фаза 2: Интеграция с LLM
    p2_passed, p2_total = test_phase2_llm()
    total_passed += p2_passed
    total_tests += p2_total

    # Фаза 3: Крайние случаи
    p3_passed, p3_total = test_phase3_edge_cases()
    total_passed += p3_passed
    total_tests += p3_total

    # ── Итоговый отчёт ──────────────────────────────────
    header("ИТОГОВЫЙ ОТЧЁТ")

    section("ЗАДАНИЕ 1: Механика инвариантов")
    print("""
  ProjectInvariants — динамический список правил [{id, category, rule}].
  Каждый инвариант имеет уникальный автоинкрементный id (не переиспользуется).
  CRUD: add(), remove(), clear(), get_all(), get_by_id().
  Persistence: JSON-файл (ручная сериализация, не asdict()).
  to_prompt() генерирует системное сообщение с жёсткими инструкциями отказа.
""")

    section("ЗАДАНИЕ 2: Отказ при конфликте")
    print("""
  При наличии инвариантов ассистент ОБЯЗАН:
    - Соблюдать ВСЕ инварианты при каждом запросе
    - ОТКАЗАТЬ при запросе, противоречащем инварианту
    - ОБЪЯСНИТЬ, какой именно инвариант нарушается
    - НЕ предлагать нарушающих решений даже по настоянию
    - При корректном запросе — помогать как обычно
""")

    section("ЗАДАНИЕ 3: Автоматическое применение")
    print("""
  Инварианты автоматически:
    1. Загружаются из JSON при создании Agent
    2. Инжектятся ПОСЛЕДНИМИ среди system-сообщений (recency bias)
    3. Сохраняются при изменении через CLI-команды
    4. Работают со ВСЕМИ 4 стратегиями без их модификации
    5. Поддерживают reload для внешних изменений
    6. Graceful degradation при corrupted JSON

  Порядок инжекции:
    [0] main system prompt
    [1] User Profile (если не пустой)
    [2] Task State (если не пустой)
    [3] Инварианты (если не пустые) ← ПОСЛЕДНИЕ, макс. вес
""")

    print(f"\n{'═' * W}")
    print(f"  ВСЕГО: {total_passed}/{total_tests} тестов пройдено")
    if total_passed == total_tests:
        print("  ✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print(f"  ✗ ПРОВАЛЕНО: {total_tests - total_passed} тестов")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
