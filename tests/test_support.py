"""
Тесты AI-ассистента поддержки пользователей.

Фаза 1: CRM MCP — протокол, инструменты, CRUD тикетов/пользователей
Фаза 2: Support RAG MCP — индексация FAQ, поиск
Фаза 3: Интеграция — SupportAgent оркестрирует оба сервера
"""

import os
import sys
import json

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from support_assistant.crm_mcp_server import handle_message as crm_handle, TOOLS as CRM_TOOLS
from support_assistant.support_rag_mcp_server import handle_message as rag_handle, TOOLS as RAG_TOOLS
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
#   ФАЗА 1: CRM MCP — unit-тесты протокола и данных
# ═══════════════════════════════════════════════════════

def test_phase1_crm():
    header("ФАЗА 1: CRM MCP-сервер")
    results = []

    # 1.1 Initialize
    section("1.1 CRM initialize handshake")
    req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1.0.0"}},
    }
    resp = crm_handle(req)
    results.append(check("Ответ не None", resp is not None))
    results.append(check("serverInfo.name == crm-mcp",
                          resp.get("result", {}).get("serverInfo", {}).get("name") == "crm-mcp"))

    # 1.2 Tools list
    section("1.2 CRM tools/list")
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = crm_handle(req)
    tools = resp.get("result", {}).get("tools", [])
    tool_names = [t["name"] for t in tools]
    print(f"    Инструменты: {tool_names}")
    results.append(check("8 инструментов", len(tools) == 8))
    results.append(check("crm_get_user в списке", "crm_get_user" in tool_names))
    results.append(check("crm_get_ticket в списке", "crm_get_ticket" in tool_names))
    results.append(check("crm_search_tickets в списке", "crm_search_tickets" in tool_names))
    results.append(check("crm_get_user_context в списке", "crm_get_user_context" in tool_names))

    # 1.3 Get user by ID
    section("1.3 crm_get_user по ID")
    req = {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "crm_get_user", "arguments": {"user_id": 1}},
    }
    resp = crm_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Найден пользователь ID=1", content.get("id") == 1))
    results.append(check("Имя: Алексей Петров", content.get("name") == "Алексей Петров"))
    results.append(check("План: pro", content.get("plan") == "pro"))
    print(f"    Пользователь: {content.get('name')} ({content.get('email')})")

    # 1.4 Get user by email
    section("1.4 crm_get_user по email")
    req = {
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "crm_get_user", "arguments": {"email": "dmitry@example.com"}},
    }
    resp = crm_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Найден по email", content.get("id") == 3))
    results.append(check("План: enterprise", content.get("plan") == "enterprise"))

    # 1.5 Get ticket
    section("1.5 crm_get_ticket")
    req = {
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "crm_get_ticket", "arguments": {"ticket_id": "TK-001"}},
    }
    resp = crm_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Тикет TK-001 найден", content.get("id") == "TK-001"))
    results.append(check("Категория: auth", content.get("category") == "auth"))
    results.append(check("Есть история", len(content.get("history", [])) > 0))
    print(f"    Тикет: {content.get('subject')}")

    # 1.6 Search tickets
    section("1.6 crm_search_tickets")
    req = {
        "jsonrpc": "2.0", "id": 6, "method": "tools/call",
        "params": {"name": "crm_search_tickets", "arguments": {"query": "авторизация"}},
    }
    resp = crm_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Найдены тикеты по 'авторизация'", content.get("count", 0) > 0))
    print(f"    Найдено: {content.get('count')}")

    # 1.7 List tickets by category
    section("1.7 crm_list_tickets по категории")
    req = {
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "crm_list_tickets", "arguments": {"category": "auth"}},
    }
    resp = crm_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Auth-тикеты найдены", content.get("count", 0) >= 2))
    print(f"    Auth-тикетов: {content.get('count')}")

    # 1.8 User context
    section("1.8 crm_get_user_context")
    req = {
        "jsonrpc": "2.0", "id": 8, "method": "tools/call",
        "params": {"name": "crm_get_user_context", "arguments": {"user_id": 1}},
    }
    resp = crm_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Контекст содержит user", "user" in content))
    results.append(check("Контекст содержит tickets", "tickets" in content))
    results.append(check("Тикеты пользователя > 0", content.get("tickets_count", 0) > 0))
    print(f"    Тикетов: {content.get('tickets_count')}, открытых: {content.get('open_tickets')}")

    # 1.9 Unknown tool
    section("1.9 Неизвестный инструмент → isError")
    req = {
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "nonexistent", "arguments": {}},
    }
    resp = crm_handle(req)
    results.append(check("isError == True", resp["result"].get("isError") == True))

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 1: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 2: Support RAG MCP — unit-тесты
# ═══════════════════════════════════════════════════════

def test_phase2_rag():
    header("ФАЗА 2: Support RAG MCP-сервер")
    results = []

    # 2.1 Initialize
    section("2.1 RAG initialize handshake")
    req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1.0.0"}},
    }
    resp = rag_handle(req)
    results.append(check("serverInfo.name == support-rag-mcp",
                          resp.get("result", {}).get("serverInfo", {}).get("name") == "support-rag-mcp"))

    # 2.2 Tools list
    section("2.2 RAG tools/list")
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = rag_handle(req)
    tools = resp.get("result", {}).get("tools", [])
    tool_names = [t["name"] for t in tools]
    print(f"    Инструменты: {tool_names}")
    results.append(check("4 инструмента", len(tools) == 4))
    results.append(check("support_index_faq", "support_index_faq" in tool_names))
    results.append(check("support_search", "support_search" in tool_names))
    results.append(check("support_ask", "support_ask" in tool_names))
    results.append(check("support_stats", "support_stats" in tool_names))

    # 2.3 Index FAQ
    section("2.3 support_index_faq")
    req = {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "support_index_faq", "arguments": {"force": True}},
    }
    resp = rag_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Статус: indexed", content.get("status") == "indexed"))
    results.append(check("Файлы проиндексированы", content.get("files_processed", 0) > 0))
    results.append(check("Чанки созданы", content.get("total_chunks", 0) > 0))
    print(f"    Файлы: {content.get('files')}")
    print(f"    Чанков: {content.get('total_chunks')}")

    # 2.4 Search
    section("2.4 support_search: 'авторизация'")
    req = {
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "support_search", "arguments": {"query": "как сбросить пароль", "top_k": 3}},
    }
    resp = rag_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("Результаты найдены", content.get("count", 0) > 0))
    search_results = content.get("results", [])
    if search_results:
        best = search_results[0]
        print(f"    Лучший результат: [{best['source']}] sim={best['similarity']:.3f}")
        results.append(check("Релевантность > 0.3", best["similarity"] > 0.3))
    else:
        results.append(check("Релевантность > 0.3", False))

    # 2.5 Search API
    section("2.5 support_search: 'rate limit 429'")
    req = {
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "support_search", "arguments": {"query": "rate limit 429", "top_k": 3}},
    }
    resp = rag_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    results.append(check("API rate limit найден", content.get("count", 0) > 0))
    if content.get("results"):
        print(f"    Источник: {content['results'][0]['source']}")

    # 2.6 Stats
    section("2.6 support_stats")
    req = {
        "jsonrpc": "2.0", "id": 6, "method": "tools/call",
        "params": {"name": "support_stats", "arguments": {}},
    }
    resp = rag_handle(req)
    content = json.loads(resp["result"]["content"][0]["text"])
    stats = content.get("stats", [])
    results.append(check("Статистика не пустая", len(stats) > 0))
    if stats:
        print(f"    Стратегия: {stats[0].get('strategy')}, чанков: {stats[0].get('chunks')}")

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 2: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   ФАЗА 3: Интеграция — SupportAgent через subprocess
# ═══════════════════════════════════════════════════════

def test_phase3_integration():
    header("ФАЗА 3: Интеграция (SupportAgent)")
    results = []

    server_dir = os.path.join(PROJECT_ROOT, "support_assistant")

    # 3.1 CRM через subprocess
    section("3.1 CRM MCP через subprocess")
    crm_cmd = [sys.executable, os.path.join(server_dir, "crm_mcp_server.py")]
    try:
        with MCPClient(server_cmd=crm_cmd) as client:
            tools = client.list_tools()
            results.append(check("CRM подключён", len(tools) == 8))

            result = client.call_tool("crm_get_user", {"user_id": 3})
            user = json.loads(result["content"][0]["text"])
            results.append(check("Пользователь через subprocess", user.get("plan") == "enterprise"))

            result = client.call_tool("crm_get_ticket", {"ticket_id": "TK-003"})
            ticket = json.loads(result["content"][0]["text"])
            results.append(check("Тикет через subprocess", ticket.get("priority") == "critical"))
            print(f"    Тикет: {ticket.get('subject')}")
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.extend([check("CRM подключён", False), check("Пользователь", False), check("Тикет", False)])

    # 3.2 RAG через subprocess
    section("3.2 RAG MCP через subprocess")
    rag_cmd = [sys.executable, os.path.join(server_dir, "support_rag_mcp_server.py")]
    try:
        with MCPClient(server_cmd=rag_cmd) as client:
            tools = client.list_tools()
            results.append(check("RAG подключён", len(tools) == 4))

            # Индексируем
            client.call_tool("support_index_faq", {"force": True})

            # Поиск
            result = client.call_tool("support_search", {"query": "двухфакторная аутентификация"})
            search = json.loads(result["content"][0]["text"])
            results.append(check("RAG поиск работает", search.get("count", 0) > 0))
            if search.get("results"):
                print(f"    Лучший: [{search['results'][0]['source']}] sim={search['results'][0]['similarity']:.3f}")
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.extend([check("RAG подключён", False), check("RAG поиск", False)])

    # 3.3 SupportAgent
    section("3.3 SupportAgent — полный цикл")
    try:
        from support_assistant.support_agent import SupportAgent
        with SupportAgent() as agent:
            # Получить пользователя
            user = agent.get_user(user_id=1)
            results.append(check("Agent.get_user работает", user.get("id") == 1))

            # Получить тикет
            ticket = agent.get_ticket("TK-001")
            results.append(check("Agent.get_ticket работает", ticket.get("id") == "TK-001"))

            # Контекст
            ctx = agent.get_user_context(user_id=1)
            results.append(check("Agent.get_user_context", ctx.get("tickets_count", 0) > 0))

            # FAQ поиск
            faq = agent.search_faq("вебхуки")
            results.append(check("Agent.search_faq", faq.get("count", 0) > 0))

            print(f"    Всё работает через SupportAgent!")
    except Exception as e:
        print(f"    Ошибка: {e}")
        results.extend([
            check("Agent.get_user", False),
            check("Agent.get_ticket", False),
            check("Agent.get_user_context", False),
            check("Agent.search_faq", False),
        ])

    passed = sum(results)
    total = len(results)
    section(f"Итого ФАЗА 3: {passed}/{total}")
    return passed, total


# ═══════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "█" * W)
    print("  ТЕСТ: AI-АССИСТЕНТ ПОДДЕРЖКИ ПОЛЬЗОВАТЕЛЕЙ")
    print("█" * W)

    total_passed = 0
    total_tests = 0

    p1_passed, p1_total = test_phase1_crm()
    total_passed += p1_passed
    total_tests += p1_total

    p2_passed, p2_total = test_phase2_rag()
    total_passed += p2_passed
    total_tests += p2_total

    p3_passed, p3_total = test_phase3_integration()
    total_passed += p3_passed
    total_tests += p3_total

    header("ИТОГОВЫЙ ОТЧЁТ")

    section("AI-ассистент поддержки пользователей")
    print("""
  Реализован мини-сервис поддержки с двумя MCP-серверами:

  1. CRM MCP (crm_mcp_server.py):
     - Данные пользователей и тикетов (JSON)
     - 8 инструментов: CRUD тикетов, поиск, контекст
     - Фильтрация по статусу/категории/приоритету

  2. Support RAG MCP (support_rag_mcp_server.py):
     - Индексация FAQ-документации (4 файла: auth, billing, api, general)
     - Семантический поиск по базе знаний
     - RAG: ответ с учётом контекста пользователя и тикета

  3. SupportAgent (support_agent.py):
     - Оркестрирует оба MCP-сервера
     - Загружает контекст пользователя/тикета из CRM
     - Генерирует ответ через RAG с контекстом
     - CLI интерфейс (cli.py)
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
