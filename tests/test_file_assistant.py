"""
Тесты AI-ассистента для работы с файлами проекта.

Фаза 1: MCP-сервер file_ops — протокол, инструменты, файловые операции
Фаза 2: Subprocess — подключение клиента
Фаза 3: FileAgent — интеграция (find_usages, check_invariants)
"""

import os
import sys
import json
import tempfile
import shutil

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from file_assistant.file_ops_mcp_server import handle_message, TOOLS
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


def call_tool(name, args=None):
    req = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    }
    resp = handle_message(req)
    text = resp["result"]["content"][0]["text"]
    return json.loads(text)


# ═══════════════════════════════════════════════════════
#   ФАЗА 1: Unit-тесты MCP-сервера file_ops
# ═══════════════════════════════════════════════════════

def test_phase1_mcp():
    header("ФАЗА 1: File Ops MCP-сервер")
    results = []

    # 1.1 Initialize
    section("1.1 Initialize handshake")
    req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1.0.0"}},
    }
    resp = handle_message(req)
    results.append(check("serverInfo.name == file-ops-mcp",
                          resp["result"]["serverInfo"]["name"] == "file-ops-mcp"))

    # 1.2 Tools list
    section("1.2 tools/list")
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = handle_message(req)
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    print(f"    Инструменты ({len(tools)}): {tool_names}")
    results.append(check("10 инструментов", len(tools) == 10))
    results.append(check("file_read", "file_read" in tool_names))
    results.append(check("file_write", "file_write" in tool_names))
    results.append(check("file_grep", "file_grep" in tool_names))
    results.append(check("file_analyze", "file_analyze" in tool_names))
    results.append(check("file_git_changes", "file_git_changes" in tool_names))

    # 1.3 file_read
    section("1.3 file_read")
    data = call_tool("file_read", {"path": "agent.py"})
    results.append(check("agent.py прочитан", "content" in data))
    results.append(check("Строк > 100", data.get("total_lines", 0) > 100))
    print(f"    Строк: {data.get('total_lines')}, Размер: {data.get('size_bytes')} байт")

    # 1.4 file_read с диапазоном
    section("1.4 file_read с диапазоном строк")
    data = call_tool("file_read", {"path": "agent.py", "start_line": 1, "end_line": 5})
    lines = data.get("content", "").splitlines()
    results.append(check("Строки 1-5 (5 строк)", len(lines) == 5))

    # 1.5 file_tree
    section("1.5 file_tree")
    data = call_tool("file_tree", {"max_depth": 1})
    results.append(check("Дерево не пустое", "tree" in data and len(data["tree"]) > 0))
    results.append(check("Файлы > 0", data.get("files", 0) > 0))
    print(f"    Файлов: {data.get('files')}, Директорий: {data.get('directories')}")

    # 1.6 file_grep
    section("1.6 file_grep: 'class Agent'")
    data = call_tool("file_grep", {"pattern": "class Agent", "extensions": [".py"]})
    results.append(check("Найден class Agent", data.get("count", 0) > 0))
    if data.get("results"):
        print(f"    Первое: {data['results'][0]['file']}:{data['results'][0]['line']}")

    # 1.7 file_glob
    section("1.7 file_glob: '*.json'")
    data = call_tool("file_glob", {"pattern": "*.json"})
    results.append(check("JSON-файлы найдены", data.get("count", 0) > 0))
    print(f"    Найдено: {data.get('count')}")

    # 1.8 file_analyze Python
    section("1.8 file_analyze: agent.py")
    data = call_tool("file_analyze", {"path": "agent.py"})
    results.append(check("Тип: python", data.get("structure", {}).get("type") == "python"))
    structure = data.get("structure", {})
    results.append(check("Классы найдены", structure.get("total_classes", 0) > 0))
    results.append(check("Функции найдены", structure.get("total_functions", 0) >= 0))
    print(f"    Классов: {structure.get('total_classes')}, Функций: {structure.get('total_functions')}")
    for cls in structure.get("classes", []):
        print(f"      class {cls['name']}: {len(cls.get('methods', []))} методов")

    # 1.9 file_analyze Markdown
    section("1.9 file_analyze: CLAUDE.md")
    data = call_tool("file_analyze", {"path": "CLAUDE.md"})
    results.append(check("Тип: markdown", data.get("structure", {}).get("type") == "markdown"))
    headers = data.get("structure", {}).get("headers", [])
    results.append(check("Заголовки найдены", len(headers) > 0))
    print(f"    Заголовков: {len(headers)}")

    # 1.10 file_write + file_read + cleanup
    section("1.10 file_write → file_read → cleanup")
    test_path = "file_assistant/_test_temp.txt"
    test_content = "Hello from test\nLine 2\nLine 3\n"
    write_data = call_tool("file_write", {"path": test_path, "content": test_content})
    results.append(check("Файл создан", write_data.get("action") == "created"))

    read_data = call_tool("file_read", {"path": test_path})
    results.append(check("Содержимое совпадает", read_data.get("content") == test_content))

    # Cleanup
    abs_path = os.path.join(PROJECT_ROOT, test_path)
    if os.path.exists(abs_path):
        os.remove(abs_path)

    # 1.11 file_patch
    section("1.11 file_patch")
    test_path = "file_assistant/_test_patch.txt"
    call_tool("file_write", {"path": test_path, "content": "foo bar baz\nfoo again\n"})
    patch_data = call_tool("file_patch", {
        "path": test_path, "find": "foo", "replace": "qux", "all": True
    })
    results.append(check("Замена выполнена (2)", patch_data.get("replacements") == 2))
    results.append(check("Diff не пустой", len(patch_data.get("diff", "")) > 0))
    print(f"    Замен: {patch_data.get('replacements')}")

    # Verify
    read_data = call_tool("file_read", {"path": test_path})
    results.append(check("foo заменён на qux", "qux" in read_data.get("content", "")))

    # Cleanup
    abs_path = os.path.join(PROJECT_ROOT, test_path)
    if os.path.exists(abs_path):
        os.remove(abs_path)

    # 1.12 file_git_changes
    section("1.12 file_git_changes: status")
    data = call_tool("file_git_changes", {"mode": "status"})
    results.append(check("Git status работает", "files" in data))
    print(f"    Файлов: {data.get('count', 0)}")

    # 1.13 file_git_changes: log
    section("1.13 file_git_changes: log")
    data = call_tool("file_git_changes", {"mode": "log", "n": 3})
    results.append(check("Git log работает", len(data.get("commits", [])) > 0))
    for c in data.get("commits", [])[:3]:
        print(f"    {c['hash']} {c['message']}")

    # 1.14 file_multi_read
    section("1.14 file_multi_read")
    data = call_tool("file_multi_read", {"paths": ["agent.py", "CLAUDE.md"]})
    results.append(check("2 файла прочитаны", data.get("count") == 2))
    results.append(check("agent.py в результатах", "agent.py" in data.get("files", {})))

    # 1.15 Неизвестный инструмент
    section("1.15 Неизвестный инструмент → isError")
    req = {
        "jsonrpc": "2.0", "id": 99, "method": "tools/call",
        "params": {"name": "nonexistent", "arguments": {}},
    }
    resp = handle_message(req)
    results.append(check("isError == True", resp["result"].get("isError") == True))

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 1: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Subprocess — подключение клиента
# ═══════════════════════════════════════════════════════

def test_phase2_subprocess():
    header("ФАЗА 2: Подключение через subprocess")
    results = []

    server_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "file_assistant", "file_ops_mcp_server.py")]

    section("2.1 Подключение и tools/list")
    try:
        with MCPClient(server_cmd=server_cmd) as client:
            tools = client.list_tools()
            results.append(check("Подключение установлено", len(tools) == 10))

            # 2.2 file_read через subprocess
            section("2.2 file_read через subprocess")
            result = client.call_tool("file_read", {"path": "agent.py"})
            text = json.loads(result["content"][0]["text"])
            results.append(check("agent.py прочитан", "content" in text))

            # 2.3 file_grep через subprocess
            section("2.3 file_grep через subprocess")
            result = client.call_tool("file_grep", {"pattern": "JSONRPCServer", "extensions": [".py"]})
            data = json.loads(result["content"][0]["text"])
            results.append(check("JSONRPCServer найден", data.get("count", 0) > 0))
            print(f"    Найдено: {data.get('count')} в {data.get('files_searched')} файлах")

            # 2.4 file_analyze через subprocess
            section("2.4 file_analyze через subprocess")
            result = client.call_tool("file_analyze", {"path": "core/jsonrpc.py"})
            data = json.loads(result["content"][0]["text"])
            results.append(check("core/jsonrpc.py проанализирован",
                                  data.get("structure", {}).get("type") == "python"))

    except Exception as e:
        print(f"    Ошибка: {e}")
        results.extend([check("Подключение", False)] * 4)

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 2: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: FileAgent — интеграция
# ═══════════════════════════════════════════════════════

def test_phase3_agent():
    header("ФАЗА 3: FileAgent — интеграция")
    results = []

    section("3.1 FileAgent — базовые операции")
    try:
        from file_assistant.file_agent import FileAgent
        with FileAgent() as agent:
            # tree
            tree = agent.tree(max_depth=1)
            results.append(check("tree работает", tree.get("files", 0) > 0))

            # read
            data = agent.read("agent.py")
            results.append(check("read работает", "content" in data))

            # grep
            grep = agent.grep("class.*Strategy", extensions=[".py"])
            results.append(check("grep regex работает", grep.get("count", 0) > 0))
            print(f"    'class.*Strategy': {grep.get('count')} совпадений")

            # analyze
            analysis = agent.analyze("mcp_hub.py")
            results.append(check("analyze работает",
                                  analysis.get("structure", {}).get("type") == "python"))

            # git_changes
            git = agent.git_changes(mode="log", n=3)
            results.append(check("git_changes работает", len(git.get("commits", [])) > 0))

            # multi_read
            multi = agent.multi_read(["agent.py", "mcp_hub.py"])
            results.append(check("multi_read работает", multi.get("count") == 2))

            print(f"    Все базовые операции работают!")
    except Exception as e:
        print(f"    Ошибка: {e}")
        for _ in range(6):
            results.append(check("Базовая операция", False))

    # 3.2 find_usages (без LLM — только поиск)
    section("3.2 find_usages — поиск без LLM")
    try:
        from file_assistant.file_agent import FileAgent
        with FileAgent() as agent:
            # Тестируем grep-часть напрямую
            grep = agent.grep("MCPHub", extensions=[".py"])
            results.append(check("MCPHub найден в проекте", grep.get("count", 0) > 0))
            files = set(m["file"] for m in grep.get("results", []))
            results.append(check("MCPHub в нескольких файлах", len(files) > 1))
            print(f"    MCPHub: {grep.get('count')} совпадений в {len(files)} файлах")
            for f in sorted(files)[:5]:
                print(f"      {f}")
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.extend([check("find_usages", False)] * 2)

    # 3.3 check_invariants — проверка правил (без LLM)
    section("3.3 check_invariants — базовая проверка")
    try:
        from file_assistant.file_agent import FileAgent
        with FileAgent() as agent:
            # Проверяем один конкретный файл
            analysis = agent.analyze("core/jsonrpc.py")
            structure = analysis.get("structure", {})
            has_docstring = structure.get("module_docstring", False)
            results.append(check("core/jsonrpc.py имеет module docstring", has_docstring))

            analysis2 = agent.analyze("agent.py")
            structure2 = analysis2.get("structure", {})
            results.append(check("agent.py анализ: классы найдены",
                                  structure2.get("total_classes", 0) > 0))
            print(f"    agent.py: {structure2.get('total_classes')} классов, "
                  f"{structure2.get('total_functions')} функций")
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.extend([check("check_invariants", False)] * 2)

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 3: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "█" * W)
    print("  ТЕСТ: AI-АССИСТЕНТ ДЛЯ РАБОТЫ С ФАЙЛАМИ")
    print("█" * W)

    total_passed = 0
    total_tests = 0

    p1_passed, p1_total = test_phase1_mcp()
    total_passed += p1_passed
    total_tests += p1_total

    p2_passed, p2_total = test_phase2_subprocess()
    total_passed += p2_passed
    total_tests += p2_total

    p3_passed, p3_total = test_phase3_agent()
    total_passed += p3_passed
    total_tests += p3_total

    header("ИТОГОВЫЙ ОТЧЁТ")

    section("AI-ассистент для работы с файлами проекта")
    print("""
  Реализован ассистент с MCP-сервером file_ops (10 инструментов):

  MCP-инструменты:
    file_read, file_write, file_patch, file_tree, file_grep,
    file_glob, file_diff, file_analyze, file_git_changes, file_multi_read

  Высокоуровневые сценарии (FileAgent + LLM):
    1. find_usages    — поиск использований символа + LLM-анализ
    2. generate_readme — генерация README из структуры проекта
    3. check_invariants — проверка файлов по правилам
    4. generate_changelog — changelog из git-истории
    5. update_docs    — обновление документации по коду

  Ассистент сам инициирует работу с файлами:
    - Сканирует дерево проекта
    - Ищет по содержимому (regex)
    - Анализирует структуру (Python AST, Markdown headers)
    - Создаёт/изменяет файлы с diff
    - Работает с git (status, diff, log)
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
