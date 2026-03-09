"""
Автоматический тест MCP Remote — MCPHub, мульти-серверы, OpenAI function calling.

Проверяет ТРИ вещи:
  1. MCPHub: конфигурация, добавление/удаление серверов, save/load
  2. Подключение: stdio серверы через Hub, агрегация инструментов, OpenAI формат
  3. Agent интеграция: tool use loop, метрики, CLI-совместимость
"""

import os
import sys
import json
import tempfile

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mcp_hub import MCPHub
from mcp_client import MCPClient, SSEMCPClient

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
#   ФАЗА 1: Unit-тесты MCPHub (без подключения)
# ═══════════════════════════════════════════════════════

def test_phase1_hub_config():
    header("ФАЗА 1: Unit-тесты MCPHub конфигурации")
    results = []

    # Тест 1.1: Пустой Hub
    section("1.1 Пустой Hub")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("{}")
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        results.append(check("is_empty() == True", hub.is_empty()))
        results.append(check("get_server_names() == []", hub.get_server_names() == []))
        results.append(check("get_connected_names() == []", hub.get_connected_names() == []))
        results.append(check("list_all_tools() == []", hub.list_all_tools() == []))
        results.append(check("get_openai_tools() == []", hub.get_openai_tools() == []))
    finally:
        os.unlink(cfg_path)

    # Тест 1.2: Несуществующий конфиг
    section("1.2 Несуществующий конфиг")
    hub = MCPHub(config_path="/tmp/nonexistent_mcp_config_12345.json")
    results.append(check("Hub создан без ошибок", True))
    results.append(check("is_empty()", hub.is_empty()))
    results.append(check("get_server_names() == []", hub.get_server_names() == []))

    # Тест 1.3: Добавление stdio-сервера
    section("1.3 Добавление stdio-сервера")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_stdio_server("test-server", ["python3", "server.py"])
        results.append(check("Сервер в конфиге", "test-server" in hub.get_server_names()))
        results.append(check("Ещё не подключён", "test-server" not in hub.get_connected_names()))

        # Проверяем сохранение
        saved = json.loads(open(cfg_path).read())
        results.append(check("Конфиг сохранён", "test-server" in saved.get("servers", {})))
        srv_cfg = saved["servers"]["test-server"]
        results.append(check("transport == stdio", srv_cfg["transport"] == "stdio"))
        results.append(check("command сохранена", srv_cfg["command"] == ["python3", "server.py"]))
    finally:
        os.unlink(cfg_path)

    # Тест 1.4: Добавление SSE-сервера
    section("1.4 Добавление SSE-сервера")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_sse_server("remote", "http://localhost:8080/sse")
        results.append(check("Сервер в конфиге", "remote" in hub.get_server_names()))

        saved = json.loads(open(cfg_path).read())
        srv_cfg = saved["servers"]["remote"]
        results.append(check("transport == sse", srv_cfg["transport"] == "sse"))
        results.append(check("url сохранён", srv_cfg["url"] == "http://localhost:8080/sse"))
    finally:
        os.unlink(cfg_path)

    # Тест 1.5: Добавление с env
    section("1.5 Добавление stdio с env")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_stdio_server("with-env", ["node", "server.js"], env={"API_KEY": "secret"})

        saved = json.loads(open(cfg_path).read())
        srv_cfg = saved["servers"]["with-env"]
        results.append(check("env сохранён", srv_cfg.get("env") == {"API_KEY": "secret"}))
    finally:
        os.unlink(cfg_path)

    # Тест 1.6: Удаление сервера
    section("1.6 Удаление сервера")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_stdio_server("to-delete", ["python3", "s.py"])
        results.append(check("Сервер добавлен", "to-delete" in hub.get_server_names()))

        removed = hub.remove_server("to-delete")
        results.append(check("remove вернул True", removed))
        results.append(check("Сервер удалён", "to-delete" not in hub.get_server_names()))

        removed2 = hub.remove_server("nonexistent")
        results.append(check("remove несуществующего вернул False", not removed2))
    finally:
        os.unlink(cfg_path)

    # Тест 1.7: Несколько серверов
    section("1.7 Несколько серверов в конфиге")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_stdio_server("server-a", ["python3", "a.py"])
        hub.add_sse_server("server-b", "http://b:8080/sse")
        hub.add_stdio_server("server-c", ["node", "c.js"])
        results.append(check("3 сервера в конфиге", len(hub.get_server_names()) == 3))
        results.append(check("Все имена верны", set(hub.get_server_names()) == {"server-a", "server-b", "server-c"}))
    finally:
        os.unlink(cfg_path)

    # Тест 1.8: _make_func_name
    section("1.8 Генерация имён функций OpenAI")
    results.append(check("server__tool", MCPHub._make_func_name("server", "tool") == "server__tool"))
    results.append(check("Спецсимволы заменены", MCPHub._make_func_name("my-srv", "my.tool") == "my-srv__my_tool"))
    long_name = MCPHub._make_func_name("a" * 40, "b" * 40)
    results.append(check("Длина <= 64", len(long_name) <= 64))

    # Тест 1.9: resolve_tool_call
    section("1.9 Разрешение tool_call")
    server, tool = hub.resolve_tool_call("myserver__mytool")
    results.append(check("server == myserver", server == "myserver"))
    results.append(check("tool == mytool", tool == "mytool"))

    server2, tool2 = hub.resolve_tool_call("simple_tool")
    results.append(check("Без __ — server=None", server2 is None))
    results.append(check("Без __ — tool=simple_tool", tool2 == "simple_tool"))

    # Тест 1.10: Corrupted JSON config
    section("1.10 Corrupted JSON конфиг")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("{broken json!!!")
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        results.append(check("Hub создан без ошибок", True))
        results.append(check("is_empty()", hub.is_empty()))
        results.append(check("Серверов нет", hub.get_server_names() == []))
    finally:
        os.unlink(cfg_path)

    # Тест 1.11: Reload конфига
    section("1.11 Reload конфига из файла")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {"preloaded": {"transport": "stdio", "command": ["echo"]}}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        results.append(check("Сервер загружен из файла", "preloaded" in hub.get_server_names()))
    finally:
        os.unlink(cfg_path)

    passed = sum(results)
    total = len(results)
    print(f"\n  ФАЗА 1: {passed}/{total} тестов пройдено")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Интеграция — stdio через Hub
# ═══════════════════════════════════════════════════════

def test_phase2_stdio_integration():
    header("ФАЗА 2: Подключение stdio-серверов через Hub")
    results = []

    server_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "mcp_server.py")]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name

    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_stdio_server("local", server_cmd)

        # Тест 2.1: Подключение к серверу
        section("2.1 Подключение к stdio-серверу")
        hub.connect("local")
        results.append(check("Сервер подключён", "local" in hub.get_connected_names()))
        results.append(check("is_empty() == False", not hub.is_empty()))

        # Тест 2.2: Информация о сервере
        section("2.2 Информация о сервере")
        info = hub.get_server_info("local")
        results.append(check("info не None", info is not None))
        results.append(check("transport == stdio", info["transport"] == "stdio"))
        results.append(check("tools_count > 0", info["tools_count"] > 0))

        # Тест 2.3: Список инструментов
        section("2.3 Агрегация инструментов")
        tools = hub.list_all_tools()
        results.append(check("Инструменты не пусты", len(tools) > 0))
        results.append(check("server == local", all(t["server"] == "local" for t in tools)))
        tool_names = [t["name"] for t in tools]
        results.append(check("ask в списке", "ask" in tool_names))
        results.append(check("list_strategies в списке", "list_strategies" in tool_names))

        # Тест 2.4: OpenAI формат
        section("2.4 OpenAI function calling формат")
        openai_tools = hub.get_openai_tools()
        results.append(check("Не пустой", len(openai_tools) > 0))
        first = openai_tools[0]
        results.append(check("type == function", first["type"] == "function"))
        results.append(check("function.name содержит __", "__" in first["function"]["name"]))
        results.append(check("function.description есть", "description" in first["function"]))
        results.append(check("function.parameters есть", "parameters" in first["function"]))

        # Тест 2.5: Вызов инструмента через Hub
        section("2.5 Вызов инструмента через Hub")
        result = hub.call_tool("local", "list_strategies", {})
        results.append(check("Результат получен", result is not None))
        content = result.get("content", [])
        results.append(check("content не пуст", len(content) > 0))
        text = content[0].get("text", "") if content else ""
        results.append(check("sliding_window в ответе", "sliding_window" in text))

        # Тест 2.6: resolve_tool_call
        section("2.6 resolve_tool_call для реального инструмента")
        func_name = openai_tools[0]["function"]["name"]
        server, tool = hub.resolve_tool_call(func_name)
        results.append(check("server == local", server == "local"))
        results.append(check("tool в списке инструментов", tool in tool_names))

        # Тест 2.7: Отключение
        section("2.7 Отключение от сервера")
        hub.disconnect("local")
        results.append(check("Сервер отключён", "local" not in hub.get_connected_names()))
        results.append(check("is_empty() после отключения", hub.is_empty()))
        results.append(check("Конфиг остался", "local" in hub.get_server_names()))

        # Тест 2.8: Reconnect
        section("2.8 Повторное подключение (connect_all)")
        errors = hub.connect_all()
        results.append(check("Нет ошибок", len(errors) == 0))
        results.append(check("Снова подключён", "local" in hub.get_connected_names()))

        hub.close_all()

    finally:
        os.unlink(cfg_path)

    passed = sum(results)
    total = len(results)
    print(f"\n  ФАЗА 2: {passed}/{total} тестов пройдено")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: Agent интеграция и edge cases
# ═══════════════════════════════════════════════════════

def test_phase3_agent_integration():
    header("ФАЗА 3: Agent интеграция и edge cases")
    results = []

    # Тест 3.1: Agent создаётся с mcp_config_path
    section("3.1 Agent с MCP конфигурацией")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as pf:
        json.dump({}, pf)
        profile_path = pf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        json.dump({}, tf)
        task_path = tf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as inv_f:
        json.dump({"invariants": [], "_next_id": 1}, inv_f)
        inv_path = inv_f.name

    try:
        from core import LLMModel

        class MockLLM(LLMModel):
            def generate(self, messages, max_tokens=1024, tools=None):
                return {
                    "text": "Mock response",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }

        from agent import Agent
        agent = Agent(
            model=MockLLM(),
            model_name="gpt-4o",
            max_tokens=100,
            system_prompt="Test",
            db_path=db_path,
            mcp_config_path=cfg_path,
            profile_path=profile_path,
            task_state_path=task_path,
            invariants_path=inv_path,
        )
        results.append(check("Agent создан", agent is not None))
        results.append(check("mcp_hub создан", agent.mcp_hub is not None))
        results.append(check("mcp_hub пуст", agent.mcp_hub.is_empty()))

        # Тест 3.2: ask() без MCP серверов — обычный ответ
        section("3.2 ask() без MCP серверов")
        result = agent.ask("hello")
        results.append(check("Ответ получен", result["text"] == "Mock response"))
        results.append(check("mcp_tools == 0", result["metrics"]["mcp_tools"] == 0))
        results.append(check("mcp_tool_iterations == 0", result["metrics"]["mcp_tool_iterations"] == 0))

        agent.close()
    finally:
        for p in [cfg_path, db_path, profile_path, task_path, inv_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    # Тест 3.3: Agent с подключённым MCP-сервером
    section("3.3 Agent с подключённым MCP-сервером")
    server_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "mcp_server.py")]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {"local": {"transport": "stdio", "command": server_cmd}}}, f)
        cfg_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as pf:
        json.dump({}, pf)
        profile_path = pf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        json.dump({}, tf)
        task_path = tf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as inv_f:
        json.dump({"invariants": [], "_next_id": 1}, inv_f)
        inv_path = inv_f.name

    try:
        # Mock LLM that calls tools on first request, then responds
        call_count = [0]

        class ToolCallingLLM(LLMModel):
            def generate(self, messages, max_tokens=1024, tools=None):
                call_count[0] += 1
                # Check if there's a tool result in messages — if so, return text
                has_tool_result = any(m.get("role") == "tool" for m in messages)
                if has_tool_result or tools is None:
                    return {
                        "text": "The strategies are: sliding_window, sticky_facts, branching, memory_layers",
                        "input_tokens": 20,
                        "output_tokens": 15,
                        "total_tokens": 35,
                    }
                # First call with tools — request a tool call
                return {
                    "text": "",
                    "input_tokens": 15,
                    "output_tokens": 10,
                    "total_tokens": 25,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "function": {
                                "name": "local__list_strategies",
                                "arguments": "{}",
                            },
                        }
                    ],
                }

        from agent import Agent
        agent = Agent(
            model=ToolCallingLLM(),
            model_name="gpt-4o",
            max_tokens=100,
            system_prompt="Test",
            db_path=db_path,
            mcp_config_path=cfg_path,
            profile_path=profile_path,
            task_state_path=task_path,
            invariants_path=inv_path,
        )

        # Подключаем серверы
        errors = agent.mcp_hub.connect_all()
        results.append(check("MCP подключён без ошибок", len(errors) == 0))
        results.append(check("Есть инструменты", not agent.mcp_hub.is_empty()))

        openai_tools = agent.mcp_hub.get_openai_tools()
        results.append(check("OpenAI tools > 0", len(openai_tools) > 0))

        # ask() с tool use
        section("3.4 ask() с вызовом инструмента")
        result = agent.ask("Какие стратегии доступны?")
        results.append(check("Ответ получен", "strategies" in result["text"].lower() or "strateg" in result["text"].lower()))
        results.append(check("mcp_tools > 0", result["metrics"]["mcp_tools"] > 0))
        results.append(check("mcp_tool_iterations == 1", result["metrics"]["mcp_tool_iterations"] == 1))
        results.append(check("input_tokens суммарные", result["input_tokens"] == 35))
        results.append(check("output_tokens суммарные", result["output_tokens"] == 25))
        results.append(check("LLM вызван 2 раза", call_count[0] == 2))

        agent.close()
    finally:
        for p in [cfg_path, db_path, profile_path, task_path, inv_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    # Тест 3.5: connect к несуществующему серверу
    section("3.5 Подключение к несуществующему серверу")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        try:
            hub.connect("nonexistent")
            results.append(check("ValueError при connect к несуществующему", False))
        except ValueError:
            results.append(check("ValueError при connect к несуществующему", True))
    finally:
        os.unlink(cfg_path)

    # Тест 3.6: call_tool к неподключённому серверу
    section("3.6 call_tool к неподключённому серверу")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        try:
            hub.call_tool("no-server", "tool", {})
            results.append(check("ValueError при call_tool", False))
        except ValueError:
            results.append(check("ValueError при call_tool", True))
    finally:
        os.unlink(cfg_path)

    # Тест 3.7: get_server_info для неподключённого
    section("3.7 get_server_info для неподключённого")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        info = hub.get_server_info("unknown")
        results.append(check("None для неизвестного сервера", info is None))
    finally:
        os.unlink(cfg_path)

    # Тест 3.8: Два stdio сервера одновременно
    section("3.8 Два stdio сервера одновременно")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_stdio_server("srv1", server_cmd)
        hub.add_stdio_server("srv2", server_cmd)
        errors = hub.connect_all()
        results.append(check("Оба подключены", len(hub.get_connected_names()) == 2))
        results.append(check("Нет ошибок", len(errors) == 0))

        tools = hub.list_all_tools()
        srv1_tools = [t for t in tools if t["server"] == "srv1"]
        srv2_tools = [t for t in tools if t["server"] == "srv2"]
        results.append(check("Инструменты от srv1", len(srv1_tools) > 0))
        results.append(check("Инструменты от srv2", len(srv2_tools) > 0))
        results.append(check("Инструменты удвоены", len(tools) == len(srv1_tools) + len(srv2_tools)))

        openai_tools = hub.get_openai_tools()
        names = [t["function"]["name"] for t in openai_tools]
        has_srv1 = any("srv1__" in n for n in names)
        has_srv2 = any("srv2__" in n for n in names)
        results.append(check("srv1__ в именах", has_srv1))
        results.append(check("srv2__ в именах", has_srv2))

        hub.close_all()
    finally:
        os.unlink(cfg_path)

    # Тест 3.9: close_all очищает всё
    section("3.9 close_all очищает подключения")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"servers": {}}, f)
        cfg_path = f.name
    try:
        hub = MCPHub(config_path=cfg_path)
        hub.add_stdio_server("srv", server_cmd)
        hub.connect("srv")
        results.append(check("Подключён", not hub.is_empty()))
        hub.close_all()
        results.append(check("is_empty() после close_all", hub.is_empty()))
        results.append(check("Конфиг не удалён", "srv" in hub.get_server_names()))
    finally:
        os.unlink(cfg_path)

    # Тест 3.10: SSEMCPClient создаётся без ошибок
    section("3.10 SSEMCPClient конструктор")
    client = SSEMCPClient(url="http://localhost:9999/sse")
    results.append(check("SSEMCPClient создан", client is not None))
    results.append(check("url сохранён", client.url == "http://localhost:9999/sse"))
    results.append(check("messages_url == None", client.messages_url is None))

    passed = sum(results)
    total = len(results)
    print(f"\n  ФАЗА 3: {passed}/{total} тестов пройдено")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ЗАПУСК
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * W)
    print("  ТЕСТ: MCP Remote — MCPHub, мульти-серверы, Agent")
    print("=" * W)

    p1, t1 = test_phase1_hub_config()
    p2, t2 = test_phase2_stdio_integration()
    p3, t3 = test_phase3_agent_integration()

    total_passed = p1 + p2 + p3
    total_tests = t1 + t2 + t3

    print(f"\n{'=' * W}")
    print(f"  ИТОГО: {total_passed}/{total_tests} тестов пройдено")

    print(f"\n  ЗАДАНИЕ 1: MCPHub конфигурация — {p1}/{t1}")
    print(f"  ЗАДАНИЕ 2: Stdio интеграция через Hub — {p2}/{t2}")
    print(f"  ЗАДАНИЕ 3: Agent интеграция и edge cases — {p3}/{t3}")
    print(f"{'=' * W}")

    if total_passed < total_tests:
        sys.exit(1)
