#!/usr/bin/env python3
"""
MCP-сервер-оркестратор пайплайна.

Соединяет 3 отдельных MCP-сервера в цепочку через MCPClient:
  1. youtrack_mcp    — поиск задач (youtrack_get_issues)
  2. prioritize_mcp  — приоритизация через LLM (prioritize_issues)
  3. filesaver_mcp   — сохранение результата (save_to_file)

Реализует JSON-RPC 2.0 поверх stdio (newline-delimited).
Протокол: MCP 2024-11-05.
"""

import sys
import json
import os
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer
from mcp_client import MCPClient

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "pipeline-mcp"
SERVER_VERSION = "1.0.0"

# Пути к серверам
_YOUTRACK_SERVER = os.path.join(_PROJECT_ROOT, "youtrack_mcp", "youtrack_mcp_server.py")
_PRIORITIZE_SERVER = os.path.join(_PROJECT_ROOT, "prioritize_mcp", "prioritize_mcp_server.py")
_FILESAVER_SERVER = os.path.join(_PROJECT_ROOT, "filesaver_mcp", "filesaver_mcp_server.py")

# ── Инструменты ──────────────────────────────────────

TOOLS = [
    {
        "name": "pipeline_run",
        "description": (
            "Автоматический пайплайн из 3 MCP-серверов: "
            "youtrack_mcp (поиск) → prioritize_mcp (LLM-приоритизация) → filesaver_mcp (сохранение). "
            "Выполняет все 3 этапа за один вызов."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос YouTrack (например '#Unresolved')",
                },
                "project": {
                    "type": "string",
                    "description": "Short name проекта (опционально)",
                },
                "top": {
                    "type": "integer",
                    "description": "Максимальное количество задач (по умолчанию 20)",
                },
                "context": {
                    "type": "string",
                    "description": "Дополнительный контекст для приоритизации (опционально)",
                },
                "filename": {
                    "type": "string",
                    "description": "Имя выходного файла (опционально)",
                },
            },
        },
    },
    {
        "name": "pipeline_status",
        "description": "Проверить доступность всех 3 MCP-серверов пайплайна.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Управление MCP-клиентами ─────────────────────────

def _connect_server(script_path):
    """Подключиться к MCP-серверу через MCPClient."""
    client = MCPClient(server_cmd=[sys.executable, script_path])
    client.connect()
    return client


def _call_and_parse(client, tool_name, arguments):
    """Вызвать инструмент на MCP-сервере и вернуть распарсенный результат."""
    result = client.call_tool(tool_name, arguments)
    if result.get("isError"):
        error_text = result.get("content", [{}])[0].get("text", "Unknown error")
        raise RuntimeError(f"MCP tool '{tool_name}' error: {error_text}")
    text = result.get("content", [{}])[0].get("text", "{}")
    return json.loads(text)


# ── Пайплайн ─────────────────────────────────────────

def _run_pipeline(args):
    """
    Автоматический пайплайн через 3 MCP-сервера:
      1. youtrack_mcp.youtrack_get_issues  → issues
      2. prioritize_mcp.prioritize_issues  → plan
      3. filesaver_mcp.save_to_file        → file
    """
    pipeline_log = []
    clients = {}

    try:
        # ── Этап 1: Подключаемся к youtrack_mcp и ищем задачи ──
        clients["youtrack"] = _connect_server(_YOUTRACK_SERVER)
        search_args = {}
        if args.get("query"):
            search_args["query"] = args["query"]
        if args.get("project"):
            search_args["project"] = args["project"]
        search_args["top"] = args.get("top", 20)

        issues = _call_and_parse(
            clients["youtrack"], "youtrack_get_issues", search_args,
        )
        issues_count = len(issues) if isinstance(issues, list) else 0

        pipeline_log.append({
            "stage": 1,
            "server": "youtrack-mcp",
            "tool": "youtrack_get_issues",
            "issues_found": issues_count,
        })

        if not issues:
            return {
                "pipeline": "completed",
                "stages": pipeline_log,
                "result": "Задачи не найдены. Пайплайн завершён без результата.",
            }

        # ── Этап 2: Подключаемся к prioritize_mcp и приоритизируем ──
        clients["prioritize"] = _connect_server(_PRIORITIZE_SERVER)
        prio_args = {"issues": issues}
        if args.get("context"):
            prio_args["context"] = args["context"]

        prio_result = _call_and_parse(
            clients["prioritize"], "prioritize_issues", prio_args,
        )
        plan_text = prio_result.get("prioritized_plan", "")

        pipeline_log.append({
            "stage": 2,
            "server": "prioritize-mcp",
            "tool": "prioritize_issues",
            "issues_processed": prio_result.get("issues_count", 0),
            "llm_tokens": prio_result.get("llm_tokens"),
        })

        # Формируем полный документ
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_content = (
            f"# План работы — приоритизация задач\n\n"
            f"Сгенерировано: {timestamp}\n"
            f"Задач обработано: {prio_result.get('issues_count', 0)}\n\n"
            f"---\n\n"
            f"{plan_text}"
        )

        # ── Этап 3: Подключаемся к filesaver_mcp и сохраняем ──
        clients["filesaver"] = _connect_server(_FILESAVER_SERVER)
        save_args = {"content": full_content}
        if args.get("filename"):
            save_args["filename"] = args["filename"]

        save_result = _call_and_parse(
            clients["filesaver"], "save_to_file", save_args,
        )

        pipeline_log.append({
            "stage": 3,
            "server": "filesaver-mcp",
            "tool": "save_to_file",
            "filepath": save_result.get("filepath"),
            "size_bytes": save_result.get("size_bytes"),
        })

        return {
            "pipeline": "completed",
            "stages": pipeline_log,
            "filepath": save_result.get("filepath"),
            "plan_preview": plan_text[:500] + ("..." if len(plan_text) > 500 else ""),
        }

    finally:
        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass


def _pipeline_status(args):
    """Проверить доступность всех 3 серверов."""
    servers = {
        "youtrack-mcp": _YOUTRACK_SERVER,
        "prioritize-mcp": _PRIORITIZE_SERVER,
        "filesaver-mcp": _FILESAVER_SERVER,
    }
    status = {}
    for name, script in servers.items():
        try:
            client = _connect_server(script)
            tools = client.list_tools()
            tool_names = [t["name"] for t in tools]
            client.close()
            status[name] = {
                "status": "ok",
                "tools": tool_names,
            }
        except Exception as e:
            status[name] = {
                "status": "error",
                "error": str(e),
            }
    return status


# ── Диспетчер ────────────────────────────────────────

_TOOL_HANDLERS = {
    "pipeline_run": lambda args: _run_pipeline(args),
    "pipeline_status": lambda args: _pipeline_status(args),
}


def handle_tool_call(name, arguments):
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    result = handler(arguments)
    return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]


# ── JSON-RPC ─────────────────────────────────────────

_server = JSONRPCServer(SERVER_NAME, SERVER_VERSION, PROTOCOL_VERSION)
_server.set_tools(TOOLS)


def handle_message(raw_message):
    _server.handle_tool_call = handle_tool_call
    return _server.handle_message(raw_message)


def main():
    _server.handle_tool_call = handle_tool_call
    _server.run_stdio()


if __name__ == "__main__":
    main()
