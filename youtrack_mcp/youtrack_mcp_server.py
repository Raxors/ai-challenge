#!/usr/bin/env python3
"""
MCP-сервер для YouTrack — предоставляет инструменты для работы с JetBrains YouTrack API.

Реализует JSON-RPC 2.0 поверх stdio (newline-delimited).
Протокол: MCP 2024-11-05.

Поддерживаемые методы:
  - initialize            -> handshake
  - notifications/initialized -> подтверждение
  - tools/list            -> список доступных инструментов
  - tools/call            -> вызов инструмента
"""

import sys
import json
import os

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

def handle_tool_call(name, arguments):
    """Выполнить инструмент и вернуть результат."""
    api = _get_api()

    if name == "youtrack_get_projects":
        top = arguments.get("top", 10)
        result = api.get_projects(top=top)
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]

    elif name == "youtrack_get_issues":
        result = api.get_issues(
            query=arguments.get("query"),
            project=arguments.get("project"),
            top=arguments.get("top", 10),
            skip=arguments.get("skip", 0),
        )
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]

    elif name == "youtrack_get_issue":
        result = api.get_issue(arguments["issue_id"])
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]

    elif name == "youtrack_create_issue":
        result = api.create_issue(
            project_id=arguments["project_id"],
            summary=arguments["summary"],
            description=arguments.get("description"),
        )
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]

    elif name == "youtrack_update_issue":
        result = api.update_issue(
            issue_id=arguments["issue_id"],
            summary=arguments.get("summary"),
            description=arguments.get("description"),
        )
        return [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]

    elif name == "youtrack_delete_issue":
        result = api.delete_issue(arguments["issue_id"])
        return [{"type": "text", "text": json.dumps(
            result if result else {"success": True}, ensure_ascii=False
        )}]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ── JSON-RPC обработка ────────────────────────────────

def make_response(req_id, result):
    """Создать JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id, code, message):
    """Создать JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_message(raw_message):
    """Обработать одно JSON-RPC сообщение. Вернуть ответ или None для notifications."""
    method = raw_message.get("method", "")
    params = raw_message.get("params", {})
    req_id = raw_message.get("id")

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
        return make_response(req_id, result)

    elif method == "notifications/initialized":
        return None  # notification — ответ не требуется

    elif method == "tools/list":
        return make_response(req_id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            content = handle_tool_call(tool_name, arguments)
            return make_response(req_id, {"content": content, "isError": False})
        except Exception as e:
            return make_response(req_id, {
                "content": [{"type": "text", "text": str(e)}],
                "isError": True,
            })

    else:
        # Неизвестный метод
        if req_id is not None:
            return make_error(req_id, -32601, f"Method not found: {method}")
        return None  # notification для неизвестного метода — игнорируем


# ── Основной цикл ─────────────────────────────────────

def main():
    """Читать JSON-RPC из stdin, обрабатывать, писать в stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as e:
            error_resp = make_error(None, -32700, f"Parse error: {e}")
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
            continue

        response = handle_message(message)

        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
