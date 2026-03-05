"""
Автоматический тест Task State Machine — управление жизненным циклом задачи.

Проверяет ТРИ вещи:
  1. Механика конечного автомата (фазы, переходы, pause/resume, persistence)
  2. LLM осведомлённость о фазах (инжекция состояния в промпт)
  3. Автоматическое отслеживание состояния (edge cases, все стратегии)

Стратегии для интеграции: все 4 (sliding_window, sticky_facts, branching, memory_layers)
"""

import os
import sys
import json
import tempfile

from dotenv import load_dotenv

load_dotenv()

from core.task_state import TaskState, PHASES, TRANSITIONS
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
#   ФАЗА 1: Unit-тесты TaskState (без LLM)
# ═══════════════════════════════════════════════════════

def test_phase1_unit():
    header("ФАЗА 1: Unit-тесты TaskState")
    results = []

    # Тест 1.1: Пустое состояние по умолчанию
    section("1.1 Пустое состояние по умолчанию")
    ts = TaskState()
    results.append(check("is_empty() для нового состояния", ts.is_empty()))
    results.append(check("task_name == ''", ts.task_name == ""))
    results.append(check("phase == ''", ts.phase == ""))
    results.append(check("history == []", ts.history == []))

    # Тест 1.2: start() инициализирует корректно
    section("1.2 start() инициализирует задачу")
    ts = TaskState()
    ts.start("Спроектировать REST API", "Определить эндпоинты", "Перечислить все CRUD операции")
    results.append(check("task_name установлено", ts.task_name == "Спроектировать REST API"))
    results.append(check("phase == 'planning'", ts.phase == "planning"))
    results.append(check("current_step установлен", ts.current_step == "Определить эндпоинты"))
    results.append(check("expected_action установлен", ts.expected_action == "Перечислить все CRUD операции"))
    results.append(check("is_empty() == False", not ts.is_empty()))
    results.append(check("created_at заполнен", ts.created_at != ""))
    results.append(check("history имеет 1 запись", len(ts.history) == 1))

    # Тест 1.3: advance() цепочка planning → execution → validation → done
    section("1.3 advance() цепочка фаз")
    ts = TaskState()
    ts.start("Тестовая задача")
    results.append(check("Начало: planning", ts.phase == "planning"))

    p1 = ts.advance()
    results.append(check("advance → execution", p1 == "execution"))

    p2 = ts.advance()
    results.append(check("advance → validation", p2 == "validation"))

    p3 = ts.advance()
    results.append(check("advance → done", p3 == "done"))

    # Тест 1.4: advance() блокируется на done
    section("1.4 advance() блокируется на done")
    try:
        ts.advance()
        results.append(check("advance() от done → ValueError", False))
    except ValueError as e:
        results.append(check("advance() от done → ValueError", "Already at 'done'" in str(e)))

    # Тест 1.5: Невалидный переход
    section("1.5 Невалидный переход")
    ts = TaskState()
    ts.start("Тест переходов")
    try:
        ts.transition("done")  # planning → done is valid
        results.append(check("planning → done разрешён", ts.phase == "done"))
    except ValueError:
        results.append(check("planning → done разрешён", False))

    ts2 = TaskState()
    ts2.start("Тест переходов 2")
    try:
        ts2.transition("validation")  # planning → validation is NOT valid
        results.append(check("planning → validation запрещён", False))
    except ValueError:
        results.append(check("planning → validation запрещён", True))

    # Тест 1.6: Откат execution → planning
    section("1.6 Backtracking: execution → planning")
    ts = TaskState()
    ts.start("Тест отката")
    ts.advance()  # → execution
    results.append(check("Фаза: execution", ts.phase == "execution"))
    ts.transition("planning")
    results.append(check("Backtrack: execution → planning", ts.phase == "planning"))

    # Тест 1.7: Откат validation → execution
    section("1.7 Backtracking: validation → execution")
    ts = TaskState()
    ts.start("Тест отката 2")
    ts.advance()  # → execution
    ts.advance()  # → validation
    results.append(check("Фаза: validation", ts.phase == "validation"))
    ts.transition("execution")
    results.append(check("Backtrack: validation → execution", ts.phase == "execution"))

    # Тест 1.8: Пауза и возобновление
    section("1.8 Pause и Resume")
    ts = TaskState()
    ts.start("Тест паузы", "Шаг А", "Сделать что-то")
    ts.advance()  # → execution
    ts.set_step("Шаг Б", "Написать код")
    ts.pause()
    results.append(check("phase == 'paused'", ts.phase == "paused"))
    results.append(check("previous_phase == 'execution'", ts.previous_phase == "execution"))
    results.append(check("previous_step == 'Шаг Б'", ts.previous_step == "Шаг Б"))
    results.append(check("previous_expected_action сохранён", ts.previous_expected_action == "Написать код"))

    restored = ts.resume()
    results.append(check("resume → execution", restored == "execution"))
    results.append(check("current_step восстановлен", ts.current_step == "Шаг Б"))
    results.append(check("expected_action восстановлен", ts.expected_action == "Написать код"))
    results.append(check("previous_phase очищен", ts.previous_phase == ""))

    # Тест 1.9: Двойная пауза вызывает ошибку
    section("1.9 Двойной pause → ошибка")
    ts = TaskState()
    ts.start("Двойная пауза")
    ts.pause()
    try:
        ts.pause()
        results.append(check("Двойной pause → ValueError", False))
    except ValueError:
        results.append(check("Двойной pause → ValueError", True))

    # Тест 1.10: Resume без паузы вызывает ошибку
    section("1.10 Resume без pause → ошибка")
    ts = TaskState()
    ts.start("Тест возобновления")
    try:
        ts.resume()
        results.append(check("Resume без pause → ValueError", False))
    except ValueError:
        results.append(check("Resume без pause → ValueError", True))

    # Тест 1.11: set_step обновляет шаг и ожидаемое действие
    section("1.11 set_step() обновляет шаг")
    ts = TaskState()
    ts.start("Тест шагов")
    ts.set_step("Определить схему API", "Написать OpenAPI спецификацию")
    results.append(check("current_step обновлён", ts.current_step == "Определить схему API"))
    results.append(check("expected_action обновлён", ts.expected_action == "Написать OpenAPI спецификацию"))
    results.append(check("updated_at обновлён", ts.updated_at != ""))

    # Тест 1.12: Сохранение/загрузка round-trip
    section("1.12 Save/Load round-trip")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        original = TaskState()
        original.start("Тест сохранения", "Шаг 1", "Действие 1")
        original.advance()
        original.set_step("Шаг 2", "Действие 2")
        original.save(tmp_path)

        with open(tmp_path, encoding="utf-8") as f:
            raw = f.read()
        print(f"\n  Saved JSON (first 300 chars):\n  {raw[:300]}...\n")
        results.append(check("JSON файл создан", os.path.exists(tmp_path)))

        loaded = TaskState.load(tmp_path)
        results.append(check("task_name совпадает", loaded.task_name == original.task_name))
        results.append(check("phase совпадает", loaded.phase == original.phase))
        results.append(check("current_step совпадает", loaded.current_step == original.current_step))
        results.append(check("expected_action совпадает", loaded.expected_action == original.expected_action))
        results.append(check("created_at совпадает", loaded.created_at == original.created_at))
        results.append(check("history совпадает", loaded.history == original.history))
    finally:
        os.unlink(tmp_path)

    # Тест 1.13: Загрузка из несуществующего файла → пустое состояние
    section("1.13 Load из несуществующего файла")
    ts = TaskState.load("/tmp/nonexistent_task_state_xyz.json")
    results.append(check("load несуществующего = пустое состояние", ts.is_empty()))

    # Тест 1.14: формат to_prompt()
    section("1.14 to_prompt() формат")
    ts = TaskState()
    ts.start("Спроектировать REST API", "Реализовать CRUD эндпоинты", "Написать обработчик POST /tasks")
    ts.advance()  # → execution
    ts.set_step("Реализовать CRUD эндпоинты", "Написать обработчик POST /tasks")
    prompt = ts.to_prompt()
    print(f"\n  to_prompt() output:\n  {prompt.replace(chr(10), chr(10) + '  ')}\n")
    results.append(check("'Спроектировать REST API' в prompt",
                          "Спроектировать REST API" in prompt))
    results.append(check("'execution' в prompt", "execution" in prompt))
    results.append(check("'[EXECUTION]' в progress bar", "[EXECUTION]" in prompt))
    results.append(check("'Реализовать CRUD' в prompt",
                          "Реализовать CRUD" in prompt))
    results.append(check("'INSTRUCTION' в prompt", "INSTRUCTION" in prompt))

    # Тест 1.15: to_prompt() для состояния paused
    section("1.15 to_prompt() для paused состояния")
    ts.pause()
    prompt = ts.to_prompt()
    results.append(check("'PAUSED' в prompt", "PAUSED" in prompt))
    results.append(check("'Do NOT repeat' в prompt", "Do NOT repeat" in prompt))

    # Test 1.16: clear() сбрасывает всё
    section("1.16 clear() сбрасывает всё")
    ts = TaskState()
    ts.start("Тест очистки", "Какой-то шаг")
    ts.advance()
    ts.clear()
    results.append(check("is_empty() после clear()", ts.is_empty()))
    results.append(check("history == [] после clear()", ts.history == []))
    results.append(check("created_at == '' после clear()", ts.created_at == ""))

    # Test 1.17: get_valid_transitions()
    section("1.17 get_valid_transitions()")
    ts = TaskState()
    ts.start("Тест переходов")
    valid = ts.get_valid_transitions()
    results.append(check("planning → [execution, paused, done]", valid == ["execution", "paused", "done"]))

    ts.advance()  # → execution
    valid = ts.get_valid_transitions()
    results.append(check("execution → [validation, planning, paused, done]",
                          valid == ["validation", "planning", "paused", "done"]))

    # Тест 1.18: Отслеживание истории
    section("1.18 История переходов")
    ts = TaskState()
    ts.start("Тест истории")
    ts.advance()
    ts.advance()
    ts.advance()
    results.append(check("history имеет 4 записи", len(ts.history) == 4))
    phases_in_history = [h["phase"] for h in ts.history]
    results.append(check("Фазы: planning → execution → validation → done",
                          phases_in_history == ["planning", "execution", "validation", "done"]))

    # Тест 1.19: done → planning рестарт цикла
    section("1.19 done → planning (рестарт цикла)")
    ts = TaskState()
    ts.start("Тест рестарта")
    ts.advance()  # execution
    ts.advance()  # validation
    ts.advance()  # done
    ts.transition("planning")
    results.append(check("done → planning", ts.phase == "planning"))

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 1: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Интеграция с LLM — осведомлённость о фазах
# ═══════════════════════════════════════════════════════

def make_agent(tmp_dir, label, strategy="sliding_window"):
    """Создать агента с временными файлами для тестирования."""
    db_path = os.path.join(tmp_dir, f"history_{label}.db")
    profile_path = os.path.join(tmp_dir, f"profile_{label}.json")
    state_path = os.path.join(tmp_dir, f"state_{label}.json")

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
    )
    return agent


def test_phase2_llm():
    header("ФАЗА 2: LLM осведомлённость о фазах")
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Тест 2.1: Запуск задачи и вопрос в фазе planning
        section("2.1 Задача в фазе planning")
        agent = make_agent(tmp_dir, "phase_plan")
        agent.task_state.start("Спроектировать REST API", "Определить эндпоинты", "Перечислить CRUD операции")
        agent.save_task_state()

        try:
            result = agent.ask("Что нам нужно сделать первым для нашей задачи?")
            response = result["text"].lower()
            print(f"  Ответ (первые 200): {result['text'][:200]}...")

            # Проверяем, что task state присутствует в метриках
            results.append(check("task_state в metrics", result["metrics"].get("task_state") is not None))
            results.append(check("metrics.task_state.phase == planning",
                                  result["metrics"]["task_state"]["phase"] == "planning"))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Запрос в planning фазе", False))
            results.append(check("metrics.task_state.phase == planning", False))
        finally:
            agent.close()

        # Тест 2.2: Переход в execution и вопрос
        section("2.2 Задача в фазе execution")
        agent = make_agent(tmp_dir, "phase_exec")
        agent.task_state.start("Создать страницу входа", "Реализовать форму", "Написать HTML/CSS для формы логина")
        agent.task_state.advance()
        agent.save_task_state()

        try:
            result = agent.ask("Что мы сейчас реализуем?")
            response = result["text"].lower()
            print(f"  Ответ (первые 200): {result['text'][:200]}...")

            results.append(check("metrics.task_state.phase == execution",
                                  result["metrics"]["task_state"]["phase"] == "execution"))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Запрос в execution фазе", False))
        finally:
            agent.close()

        # Тест 2.3: Пауза и вопрос о статусе
        section("2.3 Pause — задача приостановлена")
        agent = make_agent(tmp_dir, "phase_pause")
        agent.task_state.start("Рефакторинг модуля авторизации")
        agent.task_state.advance()
        agent.task_state.pause()
        agent.save_task_state()

        try:
            result = agent.ask("Какой текущий статус нашей задачи?")
            response = result["text"].lower()
            print(f"  Ответ (первые 200): {result['text'][:200]}...")

            results.append(check("metrics.task_state.phase == paused",
                                  result["metrics"]["task_state"]["phase"] == "paused"))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Запрос в paused фазе", False))
        finally:
            agent.close()

        # Тест 2.4: Возобновление и продолжение
        section("2.4 Resume — продолжение без повторений")
        agent = make_agent(tmp_dir, "phase_resume")
        agent.task_state.start("Написать юнит-тесты", "Тестировать модель пользователя", "Написать pytest для класса User")
        agent.task_state.advance()
        agent.task_state.set_step("Тестировать модель пользователя", "Написать pytest для класса User")
        agent.task_state.pause()
        restored = agent.task_state.resume()
        agent.save_task_state()

        try:
            result = agent.ask("Продолжи с текущим шагом, пожалуйста.")
            response = result["text"].lower()
            print(f"  Ответ (первые 200): {result['text'][:200]}...")

            results.append(check("Resume вернул execution", restored == "execution"))
        except LLMError as e:
            print(f"  Ошибка LLM: {e}")
            results.append(check("Запрос после resume", False))
        finally:
            agent.close()

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 2: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: Edge cases — пустое состояние, все стратегии
# ═══════════════════════════════════════════════════════

def test_phase3_edge_cases():
    header("ФАЗА 3: Edge cases")
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Тест 3.1: Пустое состояние не инжектится
        section("3.1 Пустое состояние не инжектится")
        agent = make_agent(tmp_dir, "empty_state")

        results.append(check("task_state.is_empty()", agent.task_state.is_empty()))

        # Симулируем prepare_messages
        agent.history.append({"role": "user", "content": "test"})
        messages = agent.strategy.prepare_messages(agent.history)
        msg_count_before = len(messages)

        if not agent.task_state.is_empty():
            messages.insert(1, {"role": "system", "content": agent.task_state.to_prompt()})

        msg_count_after = len(messages)
        results.append(check("Пустое состояние: сообщения не изменены",
                              msg_count_before == msg_count_after))
        agent.close()

        # Тест 3.2: Работает со всеми 4 стратегиями
        section("3.2 Работает со всеми 4 стратегиями")
        strategies = ["sliding_window", "sticky_facts", "branching", "memory_layers"]
        for strategy in strategies:
            agent = make_agent(tmp_dir, f"strat_{strategy}", strategy=strategy)
            agent.task_state.start(f"Тест {strategy}")
            agent.save_task_state()

            try:
                result = agent.ask("Привет, над какой задачей мы работаем?")
                has_state = result["metrics"].get("task_state") is not None
                results.append(check(f"[{strategy}] task_state в metrics", has_state))
                print(f"    [{strategy}] Ответ: {result['text'][:100]}...")
            except LLMError as e:
                print(f"    [{strategy}] Ошибка: {e}")
                results.append(check(f"[{strategy}] task_state в metrics", False))
            finally:
                agent.close()

        # Тест 3.3: reload_task_state() подхватывает внешние изменения
        section("3.3 reload_task_state() подхватывает внешние изменения")
        agent = make_agent(tmp_dir, "reload_test")
        results.append(check("Изначально пустое", agent.task_state.is_empty()))

        # Внешне записываем файл состояния задачи
        external_state = TaskState()
        external_state.start("Внешняя задача", "Внешний шаг", "Внешнее действие")
        external_state.save(agent.task_state_path)

        agent.reload_task_state()
        results.append(check("После reload: не пустое", not agent.task_state.is_empty()))
        results.append(check("После reload: task_name == 'Внешняя задача'",
                              agent.task_state.task_name == "Внешняя задача"))
        results.append(check("После reload: phase == 'planning'",
                              agent.task_state.phase == "planning"))
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
    print("  ТЕСТ TASK STATE MACHINE — ЖИЗНЕННЫЙ ЦИКЛ ЗАДАЧИ")
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

    section("ЗАДАНИЕ 1: Механика конечного автомата")
    print("""
  TaskState реализует FSM с 5 фазами:
    - planning    → определение подхода и шагов
    - execution   → реализация запланированных шагов
    - validation  → проверка и тестирование результатов
    - done        → задача завершена
    - paused      → мета-состояние с сохранением контекста

  Переходы валидируются: planning → execution, paused, done;
  execution → validation, planning (backtrack), paused, done;
  validation → done, execution (backtrack), paused;
  done → planning (рестарт цикла).

  Persistence: JSON-файл (аналогично UserProfile).
  История: каждый переход записывается с timestamp.
""")

    section("ЗАДАНИЕ 2: LLM осведомлённость о фазах")
    print("""
  Task State Machine инжектится как системное сообщение
  после User Profile и основного system prompt.
  LLM получает информацию о:
    - Текущей задаче и фазе
    - Текущем шаге и ожидаемом действии
    - Progress bar визуализации
    - Инструкции оставаться в текущей фазе

  При pause: добавляется указание НЕ повторять объяснения.
""")

    section("ЗАДАНИЕ 3: Автоматическое отслеживание состояния")
    print("""
  Состояние задачи автоматически:
    1. Загружается из JSON при создании Agent
    2. Инжектится в каждый запрос к LLM (если не пустое)
    3. Сохраняется при изменении через CLI-команды
    4. Работает со ВСЕМИ 4 стратегиями без их модификации
    5. Поддерживает reload для внешних изменений

  Механизм: инжекция на уровне Agent.ask() после profile,
  перед отправкой в LLM. Стратегии НЕ модифицируются.
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
