#!/usr/bin/env python3
"""
MCP-сервер для планировщика задач — напоминания, периодический сбор данных, сводки.

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
SERVER_NAME = "scheduler-mcp"
SERVER_VERSION = "1.0.0"

# ── Определение инструментов ─────────────────────────

TOOLS = [
    {
        "name": "scheduler_add_reminder",
        "description": "Создать одноразовое напоминание с задержкой",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Название напоминания",
                },
                "message": {
                    "type": "string",
                    "description": "Текст напоминания",
                },
                "delay_seconds": {
                    "type": "integer",
                    "description": "Задержка в секундах до срабатывания",
                },
            },
            "required": ["name", "message", "delay_seconds"],
        },
    },
    {
        "name": "scheduler_add_periodic",
        "description": "Создать периодическую задачу (summary или reminder)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Название задачи",
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "Интервал в секундах между запусками",
                },
                "task_type": {
                    "type": "string",
                    "description": "Тип callback: 'summary' или 'reminder'",
                },
                "payload": {
                    "type": "object",
                    "description": "Дополнительные данные (опционально)",
                },
            },
            "required": ["name", "interval_seconds", "task_type"],
        },
    },
    {
        "name": "scheduler_list_tasks",
        "description": "Получить список всех задач планировщика",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "scheduler_get_task",
        "description": "Получить детали задачи по ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID задачи",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "scheduler_cancel_task",
        "description": "Отменить задачу по ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID задачи",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "scheduler_get_results",
        "description": "Получить историю результатов выполнения задачи",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID задачи",
                },
                "limit": {
                    "type": "integer",
                    "description": "Максимальное количество результатов (по умолчанию 20)",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "scheduler_get_summary",
        "description": "Получить общую сводку планировщика: счётчики и последние результаты",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "scheduler_log_data",
        "description": "Вручную добавить точку данных для задачи",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID задачи",
                },
                "data": {
                    "type": "object",
                    "description": "Данные для записи",
                },
            },
            "required": ["task_id", "data"],
        },
    },
    {
        "name": "scheduler_check_notifications",
        "description": "Получить список недоставленных уведомлений планировщика",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "scheduler_cancel_all",
        "description": "Отменить все активные задачи планировщика",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "scheduler_delete_all",
        "description": "Полностью удалить все задачи и результаты из планировщика",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Загрузка конфигурации ────────────────────────────

def _load_config():
    """Загрузить настройки из .env.scheduler."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.scheduler")
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


def _get_engine():
    """Ленивое создание SchedulerEngine (при первом вызове tools/call)."""
    if not hasattr(_get_engine, "_engine"):
        from scheduler_engine import SchedulerEngine

        config = _load_config()
        db_path = config.get("SCHEDULER_DB_PATH", "scheduler.db")
        check_interval = int(config.get("SCHEDULER_CHECK_INTERVAL", "5"))
        _get_engine._engine = SchedulerEngine(db_path=db_path, check_interval=check_interval)
        _get_engine._engine.start()
    return _get_engine._engine


# ── Обработка инструментов ────────────────────────────

_TOOL_HANDLERS = {
    "scheduler_add_reminder": lambda engine, args: engine.add_reminder(
        name=args["name"], message=args["message"], delay_seconds=args["delay_seconds"],
    ),
    "scheduler_add_periodic": lambda engine, args: engine.add_periodic(
        name=args["name"], interval_seconds=args["interval_seconds"],
        task_type=args["task_type"], payload=args.get("payload"),
    ),
    "scheduler_list_tasks": lambda engine, args: engine.list_tasks(),
    "scheduler_get_task": lambda engine, args: _require_task(engine, args["task_id"]),
    "scheduler_cancel_task": lambda engine, args: engine.cancel_task(args["task_id"]),
    "scheduler_get_results": lambda engine, args: engine.get_results(
        task_id=args["task_id"], limit=args.get("limit", 20),
    ),
    "scheduler_get_summary": lambda engine, args: engine.get_summary(),
    "scheduler_log_data": lambda engine, args: engine.log_data(
        task_id=args["task_id"], data=args["data"],
    ),
    "scheduler_check_notifications": lambda engine, args: engine.get_pending_notifications(),
    "scheduler_cancel_all": lambda engine, args: engine.cancel_all(),
    "scheduler_delete_all": lambda engine, args: engine.delete_all(),
}


def _require_task(engine, task_id):
    result = engine.get_task(task_id)
    if result is None:
        raise ValueError(f"Task not found: {task_id}")
    return result


def handle_tool_call(name, arguments):
    """Выполнить инструмент и вернуть результат."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    engine = _get_engine()
    result = handler(engine, arguments)
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
