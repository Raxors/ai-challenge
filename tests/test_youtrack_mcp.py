"""
Автоматический тест MCP-сервера YouTrack.

Проверяет ТРИ вещи:
  1. Протокол JSON-RPC: handshake, tools/list, структура инструментов, ошибки
  2. Mock API: подмена YouTrackAPI, вызов handle_tool_call для каждого инструмента
  3. MCPClient интеграция: запуск youtrack_mcp_server.py как subprocess
"""

import os
import sys
import json

# Добавляем корень проекта в путь для импортов
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "youtrack_mcp"))

from youtrack_mcp.youtrack_mcp_server import (
    handle_message, handle_tool_call, TOOLS, PROTOCOL_VERSION, SERVER_NAME, _get_api,
)
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
    results.append(check("serverInfo.name == youtrack-mcp",
                          result.get("serverInfo", {}).get("name") == SERVER_NAME))
    results.append(check("capabilities.tools present",
                          "tools" in result.get("capabilities", {})))

    # Тест 1.2: notifications/initialized — нет ответа
    section("1.2 notifications/initialized")
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = handle_message(req)
    results.append(check("Notification -> None", resp is None))

    # Тест 1.3: tools/list возвращает 6 инструментов
    section("1.3 tools/list")
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = handle_message(req)
    results.append(check("Response is not None", resp is not None))
    results.append(check("id == 2", resp.get("id") == 2))

    tools = resp.get("result", {}).get("tools", [])
    results.append(check("tools count == 6", len(tools) == 6))

    tool_names = [t["name"] for t in tools]
    print(f"    Tools: {tool_names}")

    expected_names = [
        "youtrack_get_projects", "youtrack_get_issues", "youtrack_get_issue",
        "youtrack_create_issue", "youtrack_update_issue", "youtrack_delete_issue",
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
    get_issue_tool = next(t for t in tools if t["name"] == "youtrack_get_issue")
    results.append(check("youtrack_get_issue.required == ['issue_id']",
                          get_issue_tool["inputSchema"].get("required") == ["issue_id"]))

    create_issue_tool = next(t for t in tools if t["name"] == "youtrack_create_issue")
    results.append(check("youtrack_create_issue.required has project_id and summary",
                          set(create_issue_tool["inputSchema"].get("required", [])) ==
                          {"project_id", "summary"}))

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
#   ФАЗА 2: Mock API — подмена YouTrackAPI
# ═══════════════════════════════════════════════════════

class MockYouTrackAPI:
    """Mock YouTrackAPI для тестов без реального HTTP."""

    def get_projects(self, top=10):
        return [{"id": "0-0", "name": "Test Project", "shortName": "TP", "description": "desc"}]

    def get_issues(self, query=None, project=None, top=10, skip=0):
        return [{"id": "2-1", "idReadable": "TP-1", "summary": "Test issue"}]

    def get_issue(self, issue_id):
        return {
            "id": "2-1", "idReadable": issue_id, "summary": "Test issue",
            "description": "Details", "comments": [],
        }

    def create_issue(self, project_id, summary, description=None):
        return {"id": "2-2", "idReadable": "TP-2", "summary": summary}

    def update_issue(self, issue_id, summary=None, description=None):
        return {"id": "2-1", "idReadable": issue_id, "summary": summary or "Updated"}

    def delete_issue(self, issue_id):
        return None


def test_phase2_mock_api():
    header("PHASE 2: Mock API tests")
    results = []

    # Подменяем _get_api на mock
    mock_api = MockYouTrackAPI()
    _get_api._api = mock_api

    try:
        # 2.1: get_projects
        section("2.1 youtrack_get_projects")
        content = handle_tool_call("youtrack_get_projects", {"top": 5})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns list", isinstance(data, list)))
        results.append(check("Has project", len(data) > 0 and data[0]["shortName"] == "TP"))

        # 2.2: get_issues
        section("2.2 youtrack_get_issues")
        content = handle_tool_call("youtrack_get_issues", {"query": "test", "top": 5})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns list", isinstance(data, list)))
        results.append(check("Has issue", len(data) > 0 and data[0]["idReadable"] == "TP-1"))

        # 2.3: get_issue
        section("2.3 youtrack_get_issue")
        content = handle_tool_call("youtrack_get_issue", {"issue_id": "TP-1"})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns dict", isinstance(data, dict)))
        results.append(check("idReadable == TP-1", data.get("idReadable") == "TP-1"))

        # 2.4: create_issue
        section("2.4 youtrack_create_issue")
        content = handle_tool_call("youtrack_create_issue", {
            "project_id": "0-0", "summary": "New task", "description": "Details",
        })
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns dict with idReadable", "idReadable" in data))
        results.append(check("summary matches", data.get("summary") == "New task"))

        # 2.5: update_issue
        section("2.5 youtrack_update_issue")
        content = handle_tool_call("youtrack_update_issue", {
            "issue_id": "TP-1", "summary": "Updated title",
        })
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns dict", isinstance(data, dict)))
        results.append(check("idReadable == TP-1", data.get("idReadable") == "TP-1"))

        # 2.6: delete_issue
        section("2.6 youtrack_delete_issue")
        content = handle_tool_call("youtrack_delete_issue", {"issue_id": "TP-1"})
        text = content[0]["text"]
        data = json.loads(text)
        results.append(check("Returns success", data.get("success") == True))

        # 2.7: content format — type == text
        section("2.7 Content format")
        content = handle_tool_call("youtrack_get_projects", {})
        results.append(check("content[0].type == 'text'", content[0]["type"] == "text"))
        # Valid JSON
        parsed = json.loads(content[0]["text"])
        results.append(check("content[0].text is valid JSON", parsed is not None))

        # 2.8: API error → isError
        section("2.8 API error -> isError")

        class ErrorAPI:
            def get_projects(self, top=10):
                raise RuntimeError("YouTrack API error 401: Unauthorized")

        _get_api._api = ErrorAPI()
        req = {
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "youtrack_get_projects", "arguments": {}},
        }
        resp = handle_message(req)
        result = resp.get("result", {})
        results.append(check("isError == True", result.get("isError") == True))
        err_text = result.get("content", [{}])[0].get("text", "")
        results.append(check("Error message present", "401" in err_text))

    finally:
        # Сбрасываем mock
        if hasattr(_get_api, "_api"):
            del _get_api._api

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

    server_script = os.path.join(PROJECT_ROOT, "youtrack_mcp", "youtrack_mcp_server.py")
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
        results.append(check("serverInfo.name == youtrack-mcp",
                              client.server_info.get("name") == SERVER_NAME))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("Connection established", False))
        results.append(check("protocolVersion", False))
        results.append(check("serverInfo.name", False))

    # 3.2: list_tools — 6 инструментов
    section("3.2 list_tools")
    try:
        tools = client.list_tools()
        tool_names = [t["name"] for t in tools]
        print(f"    Tools ({len(tools)}): {tool_names}")

        results.append(check("tools count == 6", len(tools) == 6))

        expected_names = [
            "youtrack_get_projects", "youtrack_get_issues", "youtrack_get_issue",
            "youtrack_create_issue", "youtrack_update_issue", "youtrack_delete_issue",
        ]
        names_match = set(tool_names) == set(expected_names)
        results.append(check("Tool names match TOOLS", names_match))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("tools count == 6", False))
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
            results.append(check("Context manager: 6 tools", len(ctx_tools) == 6))
        results.append(check("Context manager: close() called",
                              ctx_client.process is None))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("Context manager: 6 tools", False))
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
    print("  TEST YOUTRACK MCP SERVER")
    print("=" * W)

    total_passed = 0
    total_tests = 0

    p1_passed, p1_total = test_phase1_protocol()
    total_passed += p1_passed
    total_tests += p1_total

    p2_passed, p2_total = test_phase2_mock_api()
    total_passed += p2_passed
    total_tests += p2_total

    p3_passed, p3_total = test_phase3_mcp_client()
    total_passed += p3_passed
    total_tests += p3_total

    header("FINAL REPORT")
    print(f"\n  Phase 1 (Protocol):       {p1_passed}/{p1_total}")
    print(f"  Phase 2 (Mock API):       {p2_passed}/{p2_total}")
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
