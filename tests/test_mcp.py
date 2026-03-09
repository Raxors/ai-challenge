"""
Автоматический тест MCP (Model Context Protocol) — подключение и инструменты.

Проверяет ТРИ вещи:
  1. Протокол JSON-RPC: корректность сообщений, handshake, ошибки
  2. Подключение и список инструментов: клиент → сервер через stdio
  3. Вызов инструментов: tools/call для инвариантов и состояния задачи

Стратегии для интеграции: все 4 (sliding_window, sticky_facts, branching, memory_layers)
"""

import os
import sys
import json
import tempfile

from dotenv import load_dotenv

load_dotenv()

# Добавляем корень проекта в путь для импортов
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mcp_server import handle_message, TOOLS, PROTOCOL_VERSION, SERVER_NAME
from mcp_client import MCPClient

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
#   ФАЗА 1: Unit-тесты протокола (без subprocess)
# ═══════════════════════════════════════════════════════

def test_phase1_protocol():
    header("ФАЗА 1: Unit-тесты JSON-RPC протокола")
    results = []

    # Тест 1.1: initialize handshake
    section("1.1 initialize handshake")
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0.0"},
        },
    }
    resp = handle_message(req)
    results.append(check("Ответ не None", resp is not None))
    results.append(check("jsonrpc == '2.0'", resp.get("jsonrpc") == "2.0"))
    results.append(check("id совпадает", resp.get("id") == 1))

    result = resp.get("result", {})
    results.append(check("protocolVersion присутствует",
                          result.get("protocolVersion") == PROTOCOL_VERSION))
    results.append(check("serverInfo.name == context-agent-mcp",
                          result.get("serverInfo", {}).get("name") == SERVER_NAME))
    results.append(check("capabilities.tools присутствует",
                          "tools" in result.get("capabilities", {})))

    # Тест 1.2: notifications/initialized — нет ответа
    section("1.2 notifications/initialized")
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = handle_message(req)
    results.append(check("Notification → None (нет ответа)", resp is None))

    # Тест 1.3: tools/list возвращает инструменты
    section("1.3 tools/list")
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = handle_message(req)
    results.append(check("Ответ не None", resp is not None))
    results.append(check("id == 2", resp.get("id") == 2))

    tools = resp.get("result", {}).get("tools", [])
    results.append(check("tools — непустой список", len(tools) > 0))

    tool_names = [t["name"] for t in tools]
    print(f"    Инструменты: {tool_names}")

    results.append(check("'ask' в списке", "ask" in tool_names))
    results.append(check("'invariant_list' в списке", "invariant_list" in tool_names))
    results.append(check("'invariant_add' в списке", "invariant_add" in tool_names))
    results.append(check("'invariant_remove' в списке", "invariant_remove" in tool_names))
    results.append(check("'state_show' в списке", "state_show" in tool_names))
    results.append(check("'state_start' в списке", "state_start" in tool_names))
    results.append(check("'state_advance' в списке", "state_advance" in tool_names))
    results.append(check("'list_strategies' в списке", "list_strategies" in tool_names))
    results.append(check("'set_strategy' в списке", "set_strategy" in tool_names))

    # Тест 1.4: Каждый инструмент имеет description и inputSchema
    section("1.4 Структура инструментов")
    all_have_desc = all("description" in t for t in tools)
    all_have_schema = all("inputSchema" in t for t in tools)
    results.append(check("Все имеют description", all_have_desc))
    results.append(check("Все имеют inputSchema", all_have_schema))

    # Проверяем required поля в inputSchema
    ask_tool = next(t for t in tools if t["name"] == "ask")
    results.append(check("ask.inputSchema.required == ['message']",
                          ask_tool["inputSchema"].get("required") == ["message"]))

    inv_add_tool = next(t for t in tools if t["name"] == "invariant_add")
    results.append(check("invariant_add.required == ['category', 'rule']",
                          inv_add_tool["inputSchema"].get("required") == ["category", "rule"]))

    # Тест 1.5: Неизвестный метод → error
    section("1.5 Неизвестный метод → error")
    req = {"jsonrpc": "2.0", "id": 99, "method": "unknown/method"}
    resp = handle_message(req)
    results.append(check("Ответ содержит error", "error" in resp))
    results.append(check("error.code == -32601",
                          resp.get("error", {}).get("code") == -32601))

    # Тест 1.6: Notification без id для неизвестного метода → None
    section("1.6 Notification для неизвестного метода")
    req = {"jsonrpc": "2.0", "method": "unknown/notification"}
    resp = handle_message(req)
    results.append(check("Неизвестный notification → None", resp is None))

    # Тест 1.7: tools/list совпадает с TOOLS
    section("1.7 tools/list == TOOLS")
    results.append(check("Количество совпадает", len(tools) == len(TOOLS)))
    results.append(check("Содержимое совпадает", tools == TOOLS))

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 1: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Интеграция — клиент подключается к серверу
# ═══════════════════════════════════════════════════════

def test_phase2_connection():
    header("ФАЗА 2: Подключение клиента к серверу")
    results = []

    server_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "mcp_server.py")]

    # Тест 2.1: Handshake — соединение устанавливается
    section("2.1 MCP handshake через subprocess")
    client = MCPClient(server_cmd=server_cmd)
    try:
        init_result = client.connect()
        print(f"    protocol: {client.protocol_version}")
        print(f"    server: {client.server_info}")

        results.append(check("Соединение установлено", client.process is not None))
        results.append(check("protocolVersion получен",
                              client.protocol_version == PROTOCOL_VERSION))
        results.append(check("serverInfo.name корректен",
                              client.server_info.get("name") == SERVER_NAME))
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.append(check("Соединение установлено", False))
        results.append(check("protocolVersion получен", False))
        results.append(check("serverInfo.name корректен", False))

    # Тест 2.2: list_tools — список инструментов возвращается
    section("2.2 list_tools через subprocess")
    try:
        tools = client.list_tools()
        tool_names = [t["name"] for t in tools]
        print(f"    Получено инструментов: {len(tools)}")
        print(f"    Имена: {tool_names}")

        results.append(check("tools — непустой список", len(tools) > 0))
        results.append(check("Количество == TOOLS", len(tools) == len(TOOLS)))
        results.append(check("'ask' присутствует", "ask" in tool_names))
        results.append(check("'invariant_list' присутствует", "invariant_list" in tool_names))
        results.append(check("'state_show' присутствует", "state_show" in tool_names))

        # Проверяем структуру каждого инструмента
        all_valid = all(
            "name" in t and "description" in t and "inputSchema" in t
            for t in tools
        )
        results.append(check("Все инструменты имеют name/description/inputSchema", all_valid))
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.append(check("tools — непустой список", False))
        results.append(check("Количество == TOOLS", False))
        results.append(check("'ask' присутствует", False))
        results.append(check("'invariant_list' присутствует", False))
        results.append(check("'state_show' присутствует", False))
        results.append(check("Все инструменты имеют name/description/inputSchema", False))

    # Тест 2.3: Повторный list_tools — стабильность
    section("2.3 Повторный list_tools")
    try:
        tools2 = client.list_tools()
        results.append(check("Повторный вызов стабилен", tools2 == tools))
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.append(check("Повторный вызов стабилен", False))

    # Тест 2.4: Неизвестный метод через клиент → error
    section("2.4 Неизвестный метод → RuntimeError")
    try:
        client._request("nonexistent/method")
        results.append(check("Неизвестный метод → RuntimeError", False))
    except RuntimeError as e:
        results.append(check("Неизвестный метод → RuntimeError",
                              "Method not found" in str(e)))
    except Exception as e:
        print(f"    Неожиданная ошибка: {e}")
        results.append(check("Неизвестный метод → RuntimeError", False))

    # Закрываем клиент
    try:
        client.close()
    except Exception:
        pass

    # Тест 2.5: Context manager
    section("2.5 Context manager (with MCPClient)")
    try:
        with MCPClient(server_cmd=server_cmd) as ctx_client:
            ctx_tools = ctx_client.list_tools()
            results.append(check("Context manager: tools получены",
                                  len(ctx_tools) == len(TOOLS)))
        results.append(check("Context manager: close() вызван",
                              ctx_client.process is None))
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.append(check("Context manager: tools получены", False))
        results.append(check("Context manager: close() вызван", False))

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 2: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: Вызов инструментов — tools/call
# ═══════════════════════════════════════════════════════

def test_phase3_tool_calls():
    header("ФАЗА 3: Вызов инструментов (tools/call)")
    results = []

    server_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "mcp_server.py")]

    with MCPClient(server_cmd=server_cmd) as client:

        # Тест 3.1: invariant_add + invariant_list
        section("3.1 invariant_add → invariant_list")
        try:
            # Добавляем инвариант
            add_result = client.call_tool("invariant_add", {
                "category": "стек",
                "rule": "Backend только на Python",
            })
            content_text = add_result.get("content", [{}])[0].get("text", "")
            add_data = json.loads(content_text)
            results.append(check("invariant_add вернул id",
                                  "id" in add_data and add_data["id"] >= 1))
            results.append(check("isError == False",
                                  add_result.get("isError") == False))
            print(f"    Добавлен инвариант: {add_data}")

            # Получаем список
            list_result = client.call_tool("invariant_list")
            content_text = list_result.get("content", [{}])[0].get("text", "")
            inv_list = json.loads(content_text)
            results.append(check("invariant_list не пуст", len(inv_list) > 0))
            results.append(check("Инвариант содержит category='стек'",
                                  any(i.get("category") == "стек" for i in inv_list)))
            print(f"    Список: {inv_list}")
        except Exception as e:
            print(f"    Ошибка: {e}")
            results.append(check("invariant_add", False))
            results.append(check("isError == False", False))
            results.append(check("invariant_list не пуст", False))
            results.append(check("Инвариант содержит category", False))

        # Тест 3.2: invariant_remove
        section("3.2 invariant_remove")
        try:
            remove_result = client.call_tool("invariant_remove", {"id": add_data["id"]})
            content_text = remove_result.get("content", [{}])[0].get("text", "")
            remove_data = json.loads(content_text)
            results.append(check("remove вернул success=True",
                                  remove_data.get("success") == True))

            # Проверяем, что список пуст
            list_result2 = client.call_tool("invariant_list")
            content_text = list_result2.get("content", [{}])[0].get("text", "")
            inv_list2 = json.loads(content_text)
            results.append(check("После удаления список пуст", len(inv_list2) == 0))
        except Exception as e:
            print(f"    Ошибка: {e}")
            results.append(check("remove вернул success=True", False))
            results.append(check("После удаления список пуст", False))

        # Тест 3.3: state_start + state_show + state_advance
        section("3.3 state_start → state_show → state_advance")
        try:
            # Старт задачи
            start_result = client.call_tool("state_start", {
                "task_name": "MCP тестовая задача",
                "first_step": "Определить эндпоинты",
            })
            content_text = start_result.get("content", [{}])[0].get("text", "")
            start_data = json.loads(content_text)
            results.append(check("state_start phase == 'planning'",
                                  start_data.get("phase") == "planning"))
            print(f"    Start: {start_data}")

            # Показать состояние
            show_result = client.call_tool("state_show")
            content_text = show_result.get("content", [{}])[0].get("text", "")
            show_data = json.loads(content_text)
            results.append(check("state_show active == True",
                                  show_data.get("active") == True))
            results.append(check("task_name совпадает",
                                  show_data.get("task_name") == "MCP тестовая задача"))
            print(f"    Show: {show_data}")

            # Продвинуть фазу
            advance_result = client.call_tool("state_advance")
            content_text = advance_result.get("content", [{}])[0].get("text", "")
            advance_data = json.loads(content_text)
            results.append(check("state_advance phase == 'execution'",
                                  advance_data.get("phase") == "execution"))
            print(f"    Advance: {advance_data}")
        except Exception as e:
            print(f"    Ошибка: {e}")
            results.append(check("state_start", False))
            results.append(check("state_show active", False))
            results.append(check("task_name совпадает", False))
            results.append(check("state_advance", False))

        # Тест 3.4: list_strategies
        section("3.4 list_strategies")
        try:
            strat_result = client.call_tool("list_strategies")
            content_text = strat_result.get("content", [{}])[0].get("text", "")
            strat_data = json.loads(content_text)
            strategies = strat_data.get("strategies", [])
            current = strat_data.get("current", "")
            results.append(check("strategies не пуст", len(strategies) > 0))
            results.append(check("sliding_window в списке",
                                  "sliding_window" in strategies))
            results.append(check("current содержит SlidingWindow",
                                  "SlidingWindow" in current))
            print(f"    Стратегии: {strategies}")
            print(f"    Текущая: {current}")
        except Exception as e:
            print(f"    Ошибка: {e}")
            results.append(check("strategies не пуст", False))
            results.append(check("sliding_window в списке", False))
            results.append(check("current содержит SlidingWindow", False))

        # Тест 3.5: Вызов неизвестного инструмента → isError
        section("3.5 Неизвестный инструмент → isError")
        try:
            err_result = client.call_tool("nonexistent_tool", {})
            results.append(check("isError == True",
                                  err_result.get("isError") == True))
            err_text = err_result.get("content", [{}])[0].get("text", "")
            results.append(check("Сообщение об ошибке",
                                  "Unknown tool" in err_text))
            print(f"    Error: {err_text}")
        except Exception as e:
            print(f"    Ошибка: {e}")
            results.append(check("isError == True", False))
            results.append(check("Сообщение об ошибке", False))

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 3: {passed}/{total} тестов")
    return passed, total


# ═══════════════════════════════════════════════════════
#   MAIN — запуск всех фаз + итоговый отчёт
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "█" * W)
    print("  ТЕСТ MCP — MODEL CONTEXT PROTOCOL")
    print("█" * W)

    total_passed = 0
    total_tests = 0

    # Фаза 1: Unit-тесты протокола (без subprocess)
    p1_passed, p1_total = test_phase1_protocol()
    total_passed += p1_passed
    total_tests += p1_total

    # Фаза 2: Подключение клиента к серверу
    p2_passed, p2_total = test_phase2_connection()
    total_passed += p2_passed
    total_tests += p2_total

    # Фаза 3: Вызов инструментов
    p3_passed, p3_total = test_phase3_tool_calls()
    total_passed += p3_passed
    total_tests += p3_total

    # ── Итоговый отчёт ──────────────────────────────────
    header("ИТОГОВЫЙ ОТЧЁТ")

    section("ЗАДАНИЕ 1: MCP-соединение")
    print("""
  Реализован MCP-сервер (mcp_server.py) и клиент (mcp_client.py)
  по протоколу JSON-RPC 2.0 поверх stdio (newline-delimited).

  Handshake:
    1. Клиент запускает сервер как subprocess
    2. Отправляет 'initialize' с protocolVersion и clientInfo
    3. Получает serverInfo и capabilities
    4. Отправляет 'notifications/initialized'

  Транспорт: stdin/stdout (совместим с MCP SDK при Python 3.10+).
  Протокол: MCP 2024-11-05.
""")

    section("ЗАДАНИЕ 2: Список инструментов")
    print("""
  MCP-сервер предоставляет 9 инструментов:
    - ask              — отправить сообщение ассистенту
    - list_strategies  — список стратегий контекста
    - set_strategy     — сменить стратегию
    - invariant_list   — список инвариантов
    - invariant_add    — добавить инвариант
    - invariant_remove — удалить инвариант
    - state_show       — текущее состояние задачи
    - state_start      — начать задачу
    - state_advance    — продвинуть фазу

  Каждый инструмент имеет name, description, inputSchema
  (JSON Schema для валидации аргументов).
""")

    section("ЗАДАНИЕ 3: Вызов инструментов")
    print("""
  tools/call позволяет вызывать инструменты с аргументами.
  Результат: {content: [{type, text}], isError: bool}.

  Проверены: добавление/удаление инвариантов,
  управление состоянием задачи, список стратегий,
  обработка ошибок (неизвестный инструмент).
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
