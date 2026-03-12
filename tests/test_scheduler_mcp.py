"""
Автоматический тест MCP-сервера планировщика задач.

Проверяет ТРИ вещи:
  1. Протокол JSON-RPC: handshake, tools/list, структура инструментов, ошибки
  2. Engine: работа с temp DB, вызов handle_tool_call для каждого инструмента
  3. MCPClient интеграция: запуск scheduler_mcp_server.py как subprocess
"""

import os
import sys
import json
import tempfile
import time

# Добавляем корень проекта в путь для импортов
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scheduler_mcp"))

from scheduler_mcp.scheduler_mcp_server import (
    handle_message, handle_tool_call, TOOLS, PROTOCOL_VERSION, SERVER_NAME, _get_engine,
)
from scheduler_mcp.scheduler_engine import SchedulerEngine
from mcp_client import MCPClient

W = 60


def header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def section(title):
    print(f"\n  {'-' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'-' * (W - 4)}")


def check(label, passed):
    mark = "+" if passed else "x"
    print(f"  {mark} {label}: {passed}")
    return passed


# ═══════════════════════════════════════════════════════
#   ФАЗА 1: Unit-тесты протокола (без subprocess)
# ═══════════════════════════════════════════════════════

def test_phase1_protocol():
    header("PHASE 1: Unit-tests JSON-RPC protocol")
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
    results.append(check("Response is not None", resp is not None))
    results.append(check("jsonrpc == '2.0'", resp.get("jsonrpc") == "2.0"))
    results.append(check("id matches", resp.get("id") == 1))

    result = resp.get("result", {})
    results.append(check("protocolVersion present",
                          result.get("protocolVersion") == PROTOCOL_VERSION))
    results.append(check("serverInfo.name == scheduler-mcp",
                          result.get("serverInfo", {}).get("name") == SERVER_NAME))
    results.append(check("capabilities.tools present",
                          "tools" in result.get("capabilities", {})))

    # Тест 1.2: notifications/initialized — нет ответа
    section("1.2 notifications/initialized")
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = handle_message(req)
    results.append(check("Notification -> None", resp is None))

    # Тест 1.3: tools/list возвращает 9 инструментов
    section("1.3 tools/list")
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = handle_message(req)
    results.append(check("Response is not None", resp is not None))
    results.append(check("id == 2", resp.get("id") == 2))

    tools = resp.get("result", {}).get("tools", [])
    results.append(check("tools count == 11", len(tools) == 11))

    tool_names = [t["name"] for t in tools]
    print(f"    Tools: {tool_names}")

    expected_names = [
        "scheduler_add_reminder", "scheduler_add_periodic",
        "scheduler_list_tasks", "scheduler_get_task",
        "scheduler_cancel_task", "scheduler_get_results",
        "scheduler_get_summary", "scheduler_log_data",
        "scheduler_check_notifications",
        "scheduler_cancel_all", "scheduler_delete_all",
    ]
    for name in expected_names:
        results.append(check(f"'{name}' in list", name in tool_names))

    # Тест 1.4: Каждый инструмент имеет description и inputSchema
    section("1.4 Tool structure")
    all_have_desc = all("description" in t for t in tools)
    all_have_schema = all("inputSchema" in t for t in tools)
    results.append(check("All have description", all_have_desc))
    results.append(check("All have inputSchema", all_have_schema))

    # inputSchema has properties
    all_have_props = all(
        "properties" in t.get("inputSchema", {}) for t in tools
    )
    results.append(check("All inputSchema have properties", all_have_props))

    # Required fields check
    reminder_tool = next(t for t in tools if t["name"] == "scheduler_add_reminder")
    results.append(check("scheduler_add_reminder.required has name, message, delay_seconds",
                          set(reminder_tool["inputSchema"].get("required", [])) ==
                          {"name", "message", "delay_seconds"}))

    log_data_tool = next(t for t in tools if t["name"] == "scheduler_log_data")
    results.append(check("scheduler_log_data.required has task_id, data",
                          set(log_data_tool["inputSchema"].get("required", [])) ==
                          {"task_id", "data"}))

    # Тест 1.5: Неизвестный метод → error
    section("1.5 Unknown method -> error")
    req = {"jsonrpc": "2.0", "id": 99, "method": "unknown/method"}
    resp = handle_message(req)
    results.append(check("Response contains error", "error" in resp))
    results.append(check("error.code == -32601",
                          resp.get("error", {}).get("code") == -32601))

    # Тест 1.6: Notification для неизвестного метода → None
    section("1.6 Unknown notification -> None")
    req = {"jsonrpc": "2.0", "method": "unknown/notification"}
    resp = handle_message(req)
    results.append(check("Unknown notification -> None", resp is None))

    # Тест 1.7: tools/list == TOOLS
    section("1.7 tools/list == TOOLS")
    results.append(check("Content matches TOOLS", tools == TOOLS))

    passed = sum(results)
    total = len(results)
    section(f"PHASE 1 total: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Engine с temp DB
# ═══════════════════════════════════════════════════════

def test_phase2_engine():
    header("PHASE 2: Engine tests with temp DB")
    results = []

    # Создаём temp DB и inject engine
    db_path = tempfile.mktemp(suffix=".db")
    engine = SchedulerEngine(db_path=db_path, check_interval=1)
    engine.start()
    _get_engine._engine = engine

    try:
        # 2.1: add_reminder
        section("2.1 scheduler_add_reminder")
        req = {
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {
                "name": "scheduler_add_reminder",
                "arguments": {"name": "Test reminder", "message": "Hello!", "delay_seconds": 60},
            },
        }
        resp = handle_message(req)
        result = resp.get("result", {})
        content = result.get("content", [{}])
        text = content[0].get("text", "{}")
        data = json.loads(text)
        results.append(check("Returns dict with id", "id" in data))
        results.append(check("task_type == 'once'", data.get("task_type") == "once"))
        results.append(check("status == 'active'", data.get("status") == "active"))
        reminder_id = data.get("id")

        # 2.2: add_periodic
        section("2.2 scheduler_add_periodic")
        content = handle_tool_call("scheduler_add_periodic", {
            "name": "Periodic summary", "interval_seconds": 30, "task_type": "summary",
        })
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns dict with id", "id" in data))
        results.append(check("task_type == 'periodic'", data.get("task_type") == "periodic"))
        periodic_id = data.get("id")

        # 2.3: list_tasks
        section("2.3 scheduler_list_tasks")
        content = handle_tool_call("scheduler_list_tasks", {})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns list", isinstance(data, list)))
        results.append(check("Has 2 tasks", len(data) == 2))

        # 2.4: get_task
        section("2.4 scheduler_get_task")
        content = handle_tool_call("scheduler_get_task", {"task_id": reminder_id})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns dict", isinstance(data, dict)))
        results.append(check("Has execution_count", "execution_count" in data))
        results.append(check("execution_count == 0", data.get("execution_count") == 0))

        # 2.5: cancel_task
        section("2.5 scheduler_cancel_task")
        content = handle_tool_call("scheduler_cancel_task", {"task_id": reminder_id})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("success == True", data.get("success") == True))

        # 2.6: log_data
        section("2.6 scheduler_log_data")
        content = handle_tool_call("scheduler_log_data", {
            "task_id": periodic_id, "data": {"metric": "cpu", "value": 42},
        })
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns dict with id", "id" in data))
        results.append(check("data preserved", data.get("data", {}).get("value") == 42))

        # 2.7: get_results
        section("2.7 scheduler_get_results")
        content = handle_tool_call("scheduler_get_results", {"task_id": periodic_id})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns list", isinstance(data, list)))
        results.append(check("Has 1 result", len(data) == 1))

        # 2.8: get_summary
        section("2.8 scheduler_get_summary")
        content = handle_tool_call("scheduler_get_summary", {})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Has total", "total" in data))
        results.append(check("total == 2", data.get("total") == 2))
        results.append(check("active == 1", data.get("active") == 1))
        results.append(check("cancelled == 1", data.get("cancelled") == 1))

        # 2.9: content format — type == text, valid JSON
        section("2.9 Content format")
        content = handle_tool_call("scheduler_list_tasks", {})
        results.append(check("content[0].type == 'text'", content[0]["type"] == "text"))
        parsed = json.loads(content[0]["text"])
        results.append(check("content[0].text is valid JSON", parsed is not None))

        # 2.10: check_notifications — пустой список когда нет уведомлений
        section("2.10 scheduler_check_notifications — empty")
        content = handle_tool_call("scheduler_check_notifications", {})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns list", isinstance(data, list)))
        results.append(check("Empty when no fired results", len(data) == 0))

        # 2.11: fire + check_notifications — непустой список
        section("2.11 scheduler_check_notifications — after fire")
        # Создаём reminder с delay=0 и запускаем _check_and_fire
        content = handle_tool_call("scheduler_add_reminder", {
            "name": "Notify test", "message": "Test notification!", "delay_seconds": 0,
        })
        notify_task = json.loads(content[0]["text"])
        time.sleep(0.1)
        engine._check_and_fire()
        content = handle_tool_call("scheduler_check_notifications", {})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns list", isinstance(data, list)))
        results.append(check("Has notifications after fire", len(data) > 0))
        if data:
            results.append(check("Has task_name", "task_name" in data[0]))
            results.append(check("Has result", "result" in data[0]))
            r = data[0].get("result", {})
            results.append(check("Result has tasks_summary", "tasks_summary" in r))
            results.append(check("Result has stats", "stats" in r))
            results.append(check("Stats has active count", r.get("stats", {}).get("active", -1) >= 0))

        # 2.12: Повторный вызов — снова пустой (delivered=1)
        section("2.12 scheduler_check_notifications — repeat is empty")
        content = handle_tool_call("scheduler_check_notifications", {})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Empty after delivery", len(data) == 0))

        # 2.13: Unknown tool → isError via handle_message
        section("2.13 Unknown tool -> isError")
        req = {
            "jsonrpc": "2.0", "id": 20, "method": "tools/call",
            "params": {"name": "unknown_tool", "arguments": {}},
        }
        resp = handle_message(req)
        result = resp.get("result", {})
        results.append(check("isError == True", result.get("isError") == True))

    finally:
        engine.close()
        if hasattr(_get_engine, "_engine"):
            del _get_engine._engine
        try:
            os.unlink(db_path)
        except OSError:
            pass

    passed = sum(results)
    total = len(results)
    section(f"PHASE 2 total: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: MCPClient интеграция (subprocess)
# ═══════════════════════════════════════════════════════

def test_phase3_mcp_client():
    header("PHASE 3: MCPClient integration (subprocess)")
    results = []

    server_script = os.path.join(PROJECT_ROOT, "scheduler_mcp", "scheduler_mcp_server.py")
    server_cmd = [sys.executable, server_script]

    # 3.1: Connect and handshake
    section("3.1 Connect and handshake")
    client = MCPClient(server_cmd=server_cmd)
    try:
        init_result = client.connect()
        print(f"    protocol: {client.protocol_version}")
        print(f"    server: {client.server_info}")

        results.append(check("Connection established", client.process is not None))
        results.append(check("protocolVersion == 2024-11-05",
                              client.protocol_version == PROTOCOL_VERSION))
        results.append(check("serverInfo.name == scheduler-mcp",
                              client.server_info.get("name") == SERVER_NAME))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("Connection established", False))
        results.append(check("protocolVersion", False))
        results.append(check("serverInfo.name", False))

    # 3.2: list_tools — 9 инструментов
    section("3.2 list_tools")
    try:
        tools = client.list_tools()
        tool_names = [t["name"] for t in tools]
        print(f"    Tools ({len(tools)}): {tool_names}")

        results.append(check("tools count == 11", len(tools) == 11))

        expected_names = [
            "scheduler_add_reminder", "scheduler_add_periodic",
            "scheduler_list_tasks", "scheduler_get_task",
            "scheduler_cancel_task", "scheduler_get_results",
            "scheduler_get_summary", "scheduler_log_data",
            "scheduler_check_notifications",
            "scheduler_cancel_all", "scheduler_delete_all",
        ]
        names_match = set(tool_names) == set(expected_names)
        results.append(check("Tool names match TOOLS", names_match))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("tools count == 11", False))
        results.append(check("Tool names match TOOLS", False))

    # 3.3: Повторный list_tools стабилен
    section("3.3 Repeated list_tools")
    try:
        tools2 = client.list_tools()
        results.append(check("Repeated call stable", tools2 == tools))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("Repeated call stable", False))

    # 3.4: Неизвестный метод → RuntimeError
    section("3.4 Unknown method -> RuntimeError")
    try:
        client._request("nonexistent/method")
        results.append(check("Unknown method -> RuntimeError", False))
    except RuntimeError as e:
        results.append(check("Unknown method -> RuntimeError",
                              "Method not found" in str(e)))
    except Exception as e:
        print(f"    Unexpected: {e}")
        results.append(check("Unknown method -> RuntimeError", False))

    # 3.5: Close
    section("3.5 Close")
    try:
        client.close()
        results.append(check("close() without error", client.process is None))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("close() without error", False))

    # 3.6: Context manager
    section("3.6 Context manager")
    try:
        with MCPClient(server_cmd=server_cmd) as ctx_client:
            ctx_tools = ctx_client.list_tools()
            results.append(check("Context manager: 9 tools", len(ctx_tools) == 11))
        results.append(check("Context manager: close() called",
                              ctx_client.process is None))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("Context manager: 9 tools", False))
        results.append(check("Context manager: close() called", False))

    passed = sum(results)
    total = len(results)
    section(f"PHASE 3 total: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "=" * W)
    print("  TEST SCHEDULER MCP SERVER")
    print("=" * W)

    total_passed = 0
    total_tests = 0

    p1_passed, p1_total = test_phase1_protocol()
    total_passed += p1_passed
    total_tests += p1_total

    p2_passed, p2_total = test_phase2_engine()
    total_passed += p2_passed
    total_tests += p2_total

    p3_passed, p3_total = test_phase3_mcp_client()
    total_passed += p3_passed
    total_tests += p3_total

    header("FINAL REPORT")
    print(f"\n  Phase 1 (Protocol):       {p1_passed}/{p1_total}")
    print(f"  Phase 2 (Engine):         {p2_passed}/{p2_total}")
    print(f"  Phase 3 (MCPClient):      {p3_passed}/{p3_total}")

    print(f"\n{'=' * W}")
    print(f"  TOTAL: {total_passed}/{total_tests} tests passed")
    if total_passed == total_tests:
        print("  + ALL TESTS PASSED")
    else:
        print(f"  x FAILED: {total_tests - total_passed} tests")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
