#!/usr/bin/env python3
"""
MCP-сервер для YouTrack — предоставляет инструменты для работы с JetBrains YouTrack API.

Реализует JSON-RPC 2.0 поверх stdio (newline-delimited).
Протокол: MCP 2024-11-05.
"""

import sys
import json
import os

# Добавляем корень проекта в путь для импорта core
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "youtrack-mcp"
SERVER_VERSION = "1.0.0"

# ── Определение инструментов ─────────────────────────

TOOLS = [
    {
        "name": "youtrack_get_projects",
        "description": "Получить список проектов YouTrack",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top": {
                    "type": "integer",
                    "description": "Максимальное количество проектов (по умолчанию 10)",
                },
            },
        },
    },
    {
        "name": "youtrack_get_issues",
        "description": "Поиск задач в YouTrack",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос YouTrack",
                },
                "project": {
                    "type": "string",
                    "description": "Short name проекта (фильтр)",
                },
                "top": {
                    "type": "integer",
                    "description": "Максимальное количество задач (по умолчанию 10)",
                },
                "skip": {
                    "type": "integer",
                    "description": "Сколько задач пропустить (по умолчанию 0)",
                },
            },
        },
    },
    {
        "name": "youtrack_get_issue",
        "description": "Получить задачу по ID (например PROJ-123)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "issue_id": {
                    "type": "string",
                    "description": "ID задачи (например PROJ-123)",
                },
            },
            "required": ["issue_id"],
        },
    },
    {
        "name": "youtrack_create_issue",
        "description": "Создать новую задачу в проекте YouTrack",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "ID проекта",
                },
                "summary": {
                    "type": "string",
                    "description": "Заголовок задачи",
                },
                "description": {
                    "type": "string",
                    "description": "Описание задачи (опционально)",
                },
            },
            "required": ["project_id", "summary"],
        },
    },
    {
        "name": "youtrack_update_issue",
        "description": "Обновить существующую задачу в YouTrack",
        "inputSchema": {
            "type": "object",
            "properties": {
                "issue_id": {
                    "type": "string",
                    "description": "ID задачи (например PROJ-123)",
                },
                "summary": {
                    "type": "string",
                    "description": "Новый заголовок (опционально)",
                },
                "description": {
                    "type": "string",
                    "description": "Новое описание (опционально)",
                },
            },
            "required": ["issue_id"],
        },
    },
    {
        "name": "youtrack_delete_issue",
        "description": "Удалить задачу из YouTrack",
        "inputSchema": {
            "type": "object",
            "properties": {
                "issue_id": {
                    "type": "string",
                    "description": "ID задачи (например PROJ-123)",
                },
            },
            "required": ["issue_id"],
        },
    },
]


# ── Загрузка конфигурации ────────────────────────────

def _load_config():
    """Загрузить YOUTRACK_URL и YOUTRACK_TOKEN из .env.youtrack."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.youtrack")
    config = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    return config


def _get_api():
    """Ленивое создание YouTrackAPI (при первом вызове tools/call)."""
    if not hasattr(_get_api, "_api"):
        from youtrack_api import YouTrackAPI

        config = _load_config()
        base_url = config.get("YOUTRACK_URL", "")
        token = config.get("YOUTRACK_TOKEN", "")
        if not base_url or not token:
            raise RuntimeError(
                "YOUTRACK_URL и YOUTRACK_TOKEN должны быть заданы в .env.youtrack"
            )
        _get_api._api = YouTrackAPI(base_url, token)
    return _get_api._api


# ── Обработка инструментов ────────────────────────────

_TOOL_HANDLERS = {
    "youtrack_get_projects": lambda api, args: api.get_projects(
        top=args.get("top", 10),
    ),
    "youtrack_get_issues": lambda api, args: api.get_issues(
        query=args.get("query"),
        project=args.get("project"),
        top=args.get("top", 10),
        skip=args.get("skip", 0),
    ),
    "youtrack_get_issue": lambda api, args: api.get_issue(args["issue_id"]),
    "youtrack_create_issue": lambda api, args: api.create_issue(
        project_id=args["project_id"],
        summary=args["summary"],
        description=args.get("description"),
    ),
    "youtrack_update_issue": lambda api, args: api.update_issue(
        issue_id=args["issue_id"],
        summary=args.get("summary"),
        description=args.get("description"),
    ),
    "youtrack_delete_issue": lambda api, args: api.delete_issue(args["issue_id"]),
}


def handle_tool_call(name, arguments):
    """Выполнить инструмент и вернуть результат."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    api = _get_api()
    result = handler(api, arguments)
    if result is None:
        result = {"success": True}
    return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]


# ── JSON-RPC обработка (совместимость с тестами) ──────

_server = JSONRPCServer(SERVER_NAME, SERVER_VERSION, PROTOCOL_VERSION)
_server.set_tools(TOOLS)


def handle_message(raw_message):
    _server.handle_tool_call = handle_tool_call
    return _server.handle_message(raw_message)


def make_response(req_id, result):
    from core.jsonrpc import make_response as _mr
    return _mr(req_id, result)


def make_error(req_id, code, message):
    from core.jsonrpc import make_error as _me
    return _me(req_id, code, message)


def main():
    _server.handle_tool_call = handle_tool_call
    _server.run_stdio()


if __name__ == "__main__":
    main()
