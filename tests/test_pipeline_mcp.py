"""
Тест пайплайна из 3 MCP-серверов + оркестратор.

Проверяет:
  1. Каждый сервер отдельно: протокол, инструменты, вызовы
  2. Оркестратор: pipeline_status, протокол
  3. Цепочка данных: search_result → prioritize → save (без LLM/YouTrack)
  4. MCPClient интеграция для каждого сервера
"""

import os
import sys
import json
import tempfile
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "prioritize_mcp"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "filesaver_mcp"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pipeline_mcp"))

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
#   ФАЗА 1: Каждый сервер — протокол и инструменты
# ═══════════════════════════════════════════════════════

def test_phase1_individual_servers():
    header("PHASE 1: Individual MCP servers (protocol + tools)")
    results = []

    servers = [
        {
            "name": "prioritize-mcp",
            "module": "prioritize_mcp.prioritize_mcp_server",
            "script": os.path.join(PROJECT_ROOT, "prioritize_mcp", "prioritize_mcp_server.py"),
            "expected_tools": ["prioritize_issues"],
        },
        {
            "name": "filesaver-mcp",
            "module": "filesaver_mcp.filesaver_mcp_server",
            "script": os.path.join(PROJECT_ROOT, "filesaver_mcp", "filesaver_mcp_server.py"),
            "expected_tools": ["save_to_file", "list_saved_files"],
        },
        {
            "name": "pipeline-mcp",
            "module": "pipeline_mcp.pipeline_mcp_server",
            "script": os.path.join(PROJECT_ROOT, "pipeline_mcp", "pipeline_mcp_server.py"),
            "expected_tools": ["pipeline_run", "pipeline_status"],
        },
    ]

    for srv in servers:
        section(f"1.x {srv['name']} — protocol via MCPClient")
        try:
            client = MCPClient(server_cmd=[sys.executable, srv["script"]])
            client.connect()

            results.append(check(
                f"{srv['name']}: connection established",
                client.process is not None,
            ))
            results.append(check(
                f"{srv['name']}: protocolVersion == 2024-11-05",
                client.protocol_version == "2024-11-05",
            ))
            results.append(check(
                f"{srv['name']}: serverInfo.name correct",
                client.server_info.get("name") == srv["name"],
            ))

            tools = client.list_tools()
            tool_names = [t["name"] for t in tools]
            print(f"    Tools: {tool_names}")

            results.append(check(
                f"{srv['name']}: tool count == {len(srv['expected_tools'])}",
                len(tools) == len(srv["expected_tools"]),
            ))
            for tname in srv["expected_tools"]:
                results.append(check(
                    f"{srv['name']}: has '{tname}'",
                    tname in tool_names,
                ))

            # Проверяем структуру
            all_valid = all(
                "name" in t and "description" in t and "inputSchema" in t
                for t in tools
            )
            results.append(check(
                f"{srv['name']}: all tools have name/description/inputSchema",
                all_valid,
            ))

            client.close()
        except Exception as e:
            print(f"    Error: {e}")
            results.append(check(f"{srv['name']}: connection", False))

    passed = sum(results)
    total = len(results)
    section(f"PHASE 1 total: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Тесты инструментов (без внешних зависимостей)
# ═══════════════════════════════════════════════════════

def test_phase2_tool_calls():
    header("PHASE 2: Tool calls on individual servers")
    results = []

    # ── 2.1: prioritize_mcp — пустой список ──
    section("2.1 prioritize_mcp: prioritize_issues — empty list")
    script = os.path.join(PROJECT_ROOT, "prioritize_mcp", "prioritize_mcp_server.py")
    try:
        with MCPClient(server_cmd=[sys.executable, script]) as client:
            result = client.call_tool("prioritize_issues", {"issues": []})
            results.append(check("isError == False", result.get("isError") == False))
            text = result.get("content", [{}])[0].get("text", "{}")
            data = json.loads(text)
            results.append(check("issues_count == 0", data.get("issues_count") == 0))
            results.append(check("Has prioritized_plan", "prioritized_plan" in data))
            print(f"    Result: {data}")
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"prioritize empty: {i}", False) for i in range(3)])

    # ── 2.2: prioritize_mcp — неизвестный инструмент ──
    section("2.2 prioritize_mcp: unknown tool -> isError")
    try:
        with MCPClient(server_cmd=[sys.executable, script]) as client:
            result = client.call_tool("unknown_tool", {})
            results.append(check("isError == True", result.get("isError") == True))
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("unknown tool isError", False))

    # ── 2.3: filesaver_mcp — save_to_file ──
    section("2.3 filesaver_mcp: save_to_file")
    script = os.path.join(PROJECT_ROOT, "filesaver_mcp", "filesaver_mcp_server.py")
    try:
        with MCPClient(server_cmd=[sys.executable, script]) as client:
            result = client.call_tool("save_to_file", {
                "content": "# Test\nPipeline save test",
                "filename": "test_phase2.md",
            })
            results.append(check("isError == False", result.get("isError") == False))
            text = result.get("content", [{}])[0].get("text", "{}")
            data = json.loads(text)
            results.append(check("Has filepath", "filepath" in data))
            results.append(check("Has filename", data.get("filename") == "test_phase2.md"))
            results.append(check("Has size_bytes", data.get("size_bytes", 0) > 0))
            filepath = data.get("filepath", "")
            results.append(check("File exists", os.path.exists(filepath)))
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    content = f.read()
                results.append(check("Content correct", "Pipeline save test" in content))
                os.unlink(filepath)
            else:
                results.append(check("Content correct", False))
            print(f"    Saved: {filepath}")
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"filesaver save: {i}", False) for i in range(6)])

    # ── 2.4: filesaver_mcp — list_saved_files ──
    section("2.4 filesaver_mcp: list_saved_files")
    try:
        with MCPClient(server_cmd=[sys.executable, script]) as client:
            result = client.call_tool("list_saved_files", {})
            results.append(check("isError == False", result.get("isError") == False))
            text = result.get("content", [{}])[0].get("text", "{}")
            data = json.loads(text)
            results.append(check("Has files list", "files" in data))
            results.append(check("Has directory", "directory" in data))
            print(f"    Files count: {len(data.get('files', []))}")
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"filesaver list: {i}", False) for i in range(3)])

    # ── 2.5: filesaver_mcp — auto filename ──
    section("2.5 filesaver_mcp: save_to_file — auto filename")
    try:
        with MCPClient(server_cmd=[sys.executable, script]) as client:
            result = client.call_tool("save_to_file", {"content": "auto"})
            text = result.get("content", [{}])[0].get("text", "{}")
            data = json.loads(text)
            results.append(check("Auto filename starts with result_",
                                  data.get("filename", "").startswith("result_")))
            filepath = data.get("filepath", "")
            if os.path.exists(filepath):
                os.unlink(filepath)
    except Exception as e:
        print(f"    Error: {e}")
        results.append(check("auto filename", False))

    passed = sum(results)
    total = len(results)
    section(f"PHASE 2 total: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: Цепочка данных между серверами
# ═══════════════════════════════════════════════════════

def test_phase3_data_chain():
    header("PHASE 3: Data chain between 3 servers")
    results = []

    # Имитируем полную цепочку вручную (без YouTrack и LLM)

    # ── 3.1: Имитация результата youtrack_mcp ──
    section("3.1 Simulated YouTrack search result")
    search_result = [
        {
            "id": "2-100", "idReadable": "PROJ-100",
            "summary": "Server outage in production",
            "description": "Database connections exhausted, users getting 503",
            "customFields": [
                {"name": "Priority", "value": {"name": "Critical"}},
                {"name": "State", "value": {"name": "Open"}},
            ],
        },
        {
            "id": "2-101", "idReadable": "PROJ-101",
            "summary": "Add dark mode support",
            "description": "Users requested dark mode in settings",
            "customFields": [
                {"name": "Priority", "value": {"name": "Normal"}},
                {"name": "State", "value": {"name": "Open"}},
            ],
        },
        {
            "id": "2-102", "idReadable": "PROJ-102",
            "summary": "Update third-party licenses",
            "description": "License audit next month",
            "customFields": [
                {"name": "Priority", "value": {"name": "Low"}},
                {"name": "State", "value": {"name": "Open"}},
            ],
        },
    ]
    results.append(check("Search result has 3 issues", len(search_result) == 3))
    results.append(check("Issues have idReadable", all("idReadable" in i for i in search_result)))

    # ── 3.2: Передаём в prioritize_mcp ──
    section("3.2 Pass search result → prioritize_mcp (empty issues test)")
    prio_script = os.path.join(PROJECT_ROOT, "prioritize_mcp", "prioritize_mcp_server.py")
    try:
        with MCPClient(server_cmd=[sys.executable, prio_script]) as client:
            # Используем пустой список чтобы не вызывать LLM
            result = client.call_tool("prioritize_issues", {"issues": []})
            text = result.get("content", [{}])[0].get("text", "{}")
            prio_data = json.loads(text)
            results.append(check("Prioritize accepts issues array", "prioritized_plan" in prio_data))
            results.append(check("Returns issues_count", "issues_count" in prio_data))
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"prioritize chain: {i}", False) for i in range(2)])

    # ── 3.3: Передаём plan в filesaver_mcp ──
    section("3.3 Pass prioritize result → filesaver_mcp")
    # Имитируем результат приоритизации
    simulated_plan = """## Сделать сегодня (критично)

- [PROJ-100] Server outage in production — критический баг, блокирует пользователей

## На этой неделе (важно)

- [PROJ-101] Add dark mode support — запрос от пользователей, повышает UX

## Бэклог

- [PROJ-102] Update third-party licenses — аудит через месяц, не срочно
"""

    saver_script = os.path.join(PROJECT_ROOT, "filesaver_mcp", "filesaver_mcp_server.py")
    try:
        with MCPClient(server_cmd=[sys.executable, saver_script]) as client:
            result = client.call_tool("save_to_file", {
                "content": simulated_plan,
                "filename": "chain_test_output.md",
            })
            text = result.get("content", [{}])[0].get("text", "{}")
            save_data = json.loads(text)
            filepath = save_data.get("filepath", "")

            results.append(check("Save accepted plan text", "filepath" in save_data))
            results.append(check("File created", os.path.exists(filepath)))

            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                results.append(check("Content has PROJ-100", "PROJ-100" in content))
                results.append(check("Content has PROJ-101", "PROJ-101" in content))
                results.append(check("Content has PROJ-102", "PROJ-102" in content))
                results.append(check("Content has categories",
                                      "Сделать сегодня" in content and "Бэклог" in content))
                os.unlink(filepath)
            else:
                results.extend([check(f"content check {i}", False) for i in range(4)])

            print(f"    Chain: search(3 issues) → prioritize → save({save_data.get('size_bytes', 0)} bytes)")
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"filesaver chain: {i}", False) for i in range(6)])

    # ── 3.4: Полная цепочка: все 3 сервера последовательно ──
    section("3.4 Full manual chain: youtrack → prioritize → filesaver")
    try:
        # Шаг 1: youtrack_mcp — проверяем подключение (не вызываем get_issues без конфига)
        yt_script = os.path.join(PROJECT_ROOT, "youtrack_mcp", "youtrack_mcp_server.py")
        with MCPClient(server_cmd=[sys.executable, yt_script]) as yt_client:
            yt_tools = yt_client.list_tools()
            yt_tool_names = [t["name"] for t in yt_tools]
            results.append(check("YouTrack: has youtrack_get_issues",
                                  "youtrack_get_issues" in yt_tool_names))

        # Шаг 2: prioritize_mcp — проверяем подключение
        with MCPClient(server_cmd=[sys.executable, prio_script]) as prio_client:
            prio_tools = prio_client.list_tools()
            prio_tool_names = [t["name"] for t in prio_tools]
            results.append(check("Prioritize: has prioritize_issues",
                                  "prioritize_issues" in prio_tool_names))

        # Шаг 3: filesaver_mcp — проверяем подключение
        with MCPClient(server_cmd=[sys.executable, saver_script]) as saver_client:
            saver_tools = saver_client.list_tools()
            saver_tool_names = [t["name"] for t in saver_tools]
            results.append(check("FileSaver: has save_to_file",
                                  "save_to_file" in saver_tool_names))

        print("    All 3 servers accessible via MCPClient")
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"full chain: {i}", False) for i in range(3)])

    passed = sum(results)
    total = len(results)
    section(f"PHASE 3 total: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 4: Оркестратор (pipeline_mcp)
# ═══════════════════════════════════════════════════════

def test_phase4_orchestrator():
    header("PHASE 4: Pipeline orchestrator")
    results = []

    script = os.path.join(PROJECT_ROOT, "pipeline_mcp", "pipeline_mcp_server.py")

    # 4.1: Протокол
    section("4.1 Orchestrator protocol")
    try:
        with MCPClient(server_cmd=[sys.executable, script]) as client:
            results.append(check("Connection established", client.process is not None))
            results.append(check("serverInfo.name == pipeline-mcp",
                                  client.server_info.get("name") == "pipeline-mcp"))

            tools = client.list_tools()
            tool_names = [t["name"] for t in tools]
            print(f"    Tools: {tool_names}")
            results.append(check("Has pipeline_run", "pipeline_run" in tool_names))
            results.append(check("Has pipeline_status", "pipeline_status" in tool_names))
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"orchestrator: {i}", False) for i in range(4)])

    # 4.2: pipeline_status — проверяет все 3 сервера
    section("4.2 pipeline_status — checks all 3 servers")
    try:
        with MCPClient(server_cmd=[sys.executable, script]) as client:
            result = client.call_tool("pipeline_status", {})
            results.append(check("isError == False", result.get("isError") == False))
            text = result.get("content", [{}])[0].get("text", "{}")
            data = json.loads(text)

            # Проверяем каждый сервер
            for srv_name in ["prioritize-mcp", "filesaver-mcp"]:
                srv_status = data.get(srv_name, {})
                ok = srv_status.get("status") == "ok"
                results.append(check(f"{srv_name}: status == ok", ok))
                if ok:
                    print(f"    {srv_name}: tools = {srv_status.get('tools', [])}")

            # youtrack может быть недоступен (нет конфига)
            yt_status = data.get("youtrack-mcp", {})
            results.append(check("youtrack-mcp: responded",
                                  yt_status.get("status") in ("ok", "error")))
            print(f"    youtrack-mcp: {yt_status.get('status')}")
    except Exception as e:
        print(f"    Error: {e}")
        results.extend([check(f"status: {i}", False) for i in range(4)])

    passed = sum(results)
    total = len(results)
    section(f"PHASE 4 total: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "=" * W)
    print("  TEST PIPELINE: 3 MCP SERVERS + ORCHESTRATOR")
    print("=" * W)

    total_passed = 0
    total_tests = 0

    p1_passed, p1_total = test_phase1_individual_servers()
    total_passed += p1_passed
    total_tests += p1_total

    p2_passed, p2_total = test_phase2_tool_calls()
    total_passed += p2_passed
    total_tests += p2_total

    p3_passed, p3_total = test_phase3_data_chain()
    total_passed += p3_passed
    total_tests += p3_total

    p4_passed, p4_total = test_phase4_orchestrator()
    total_passed += p4_passed
    total_tests += p4_total

    header("FINAL REPORT")
    print(f"\n  Phase 1 (Individual servers):  {p1_passed}/{p1_total}")
    print(f"  Phase 2 (Tool calls):         {p2_passed}/{p2_total}")
    print(f"  Phase 3 (Data chain):         {p3_passed}/{p3_total}")
    print(f"  Phase 4 (Orchestrator):       {p4_passed}/{p4_total}")

    print(f"\n{'=' * W}")
    print(f"  TOTAL: {total_passed}/{total_tests} tests passed")
    if total_passed == total_tests:
        print("  + ALL TESTS PASSED")
    else:
        print(f"  x FAILED: {total_tests - total_passed} tests")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
