#!/usr/bin/env python3
"""
MCP-сервер — предоставляет инструменты агента через Model Context Protocol.

Реализует JSON-RPC 2.0 поверх stdio (newline-delimited).
Протокол: MCP 2024-11-05.
"""

import json
from core.jsonrpc import JSONRPCServer

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "context-agent-mcp"
SERVER_VERSION = "1.0.0"

TOOLS = [
    {
        "name": "ask",
        "description": "Отправить сообщение ассистенту и получить ответ",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Текст сообщения пользователя",
                }
            },
            "required": ["message"],
        },
    },
    {
        "name": "list_strategies",
        "description": "Получить список доступных стратегий контекста",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "set_strategy",
        "description": "Сменить стратегию управления контекстом",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Имя стратегии (sliding_window, sticky_facts, branching, memory_layers)",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "invariant_list",
        "description": "Получить список всех инвариантов проекта",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "invariant_add",
        "description": "Добавить инвариант проекта (жёсткое ограничение)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Категория (стек, архитектура, бизнес-правило и т.д.)",
                },
                "rule": {
                    "type": "string",
                    "description": "Текст правила",
                },
            },
            "required": ["category", "rule"],
        },
    },
    {
        "name": "invariant_remove",
        "description": "Удалить инвариант по ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "ID инварианта"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "state_show",
        "description": "Показать текущее состояние задачи (Task State Machine)",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "state_start",
        "description": "Начать новую задачу (фаза: planning)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Название задачи"},
                "first_step": {"type": "string", "description": "Первый шаг (опционально)"},
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "state_advance",
        "description": "Перейти к следующей фазе задачи",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def _get_agent():
    """Ленивое создание агента (при первом вызове tools/call)."""
    if not hasattr(_get_agent, "_agent"):
        from dotenv import load_dotenv
        load_dotenv()
        from llm import OpenAIModel
        from agent import Agent

        llm = OpenAIModel(model="gpt-4o")
        _get_agent._agent = Agent(
            model=llm,
            model_name="gpt-4o",
            max_tokens=4096,
            system_prompt=(
                "You are a helpful assistant with access to MCP tools. "
                "When the user asks you to perform a task that involves multiple steps, "
                "use the available tools in sequence. "
                "Always use tools when they are relevant to the user's request. "
                "Answer concisely and clearly."
            ),
            strategy_name="sliding_window",
            window_size=10,
        )
    return _get_agent._agent


def handle_tool_call(name, arguments):
    """Выполнить инструмент и вернуть результат."""
    agent = _get_agent()

    if name == "ask":
        result = agent.ask(arguments["message"])
        return [{"type": "text", "text": result["text"]}]

    elif name == "list_strategies":
        from strategies import get_strategy_names
        names = get_strategy_names()
        current = agent.strategy.get_name()
        return [{"type": "text", "text": json.dumps({
            "strategies": names, "current": current,
        }, ensure_ascii=False)}]

    elif name == "set_strategy":
        new_name = agent.set_strategy(arguments["name"])
        return [{"type": "text", "text": f"Strategy changed to: {new_name}"}]

    elif name == "invariant_list":
        inv_list = agent.invariants.get_all()
        return [{"type": "text", "text": json.dumps(inv_list, ensure_ascii=False)}]

    elif name == "invariant_add":
        inv_id = agent.invariants.add(arguments["category"], arguments["rule"])
        agent.save_invariants()
        return [{"type": "text", "text": json.dumps({"id": inv_id})}]

    elif name == "invariant_remove":
        success = agent.invariants.remove(arguments["id"])
        if success:
            agent.save_invariants()
        return [{"type": "text", "text": json.dumps({"success": success})}]

    elif name == "state_show":
        ts = agent.task_state
        if ts.is_empty():
            return [{"type": "text", "text": json.dumps({"active": False})}]
        return [{"type": "text", "text": json.dumps({
            "active": True,
            "task_name": ts.task_name,
            "phase": ts.phase,
            "current_step": ts.current_step,
        }, ensure_ascii=False)}]

    elif name == "state_start":
        agent.task_state.start(
            arguments["task_name"],
            arguments.get("first_step", ""),
        )
        agent.save_task_state()
        return [{"type": "text", "text": json.dumps({"phase": "planning"})}]

    elif name == "state_advance":
        new_phase = agent.task_state.advance()
        agent.save_task_state()
        return [{"type": "text", "text": json.dumps({"phase": new_phase})}]

    else:
        raise ValueError(f"Unknown tool: {name}")


# Совместимость: handle_message и make_response/make_error как функции модуля
_server = JSONRPCServer(SERVER_NAME, SERVER_VERSION, PROTOCOL_VERSION)
_server.set_tools(TOOLS)
_server.handle_tool_call = staticmethod(handle_tool_call)


def handle_message(raw_message):
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
