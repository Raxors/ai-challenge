#!/usr/bin/env python3
"""
MCP-сервер приоритизации — принимает список задач, возвращает план работы через LLM.

Реализует JSON-RPC 2.0 поверх stdio (newline-delimited).
Протокол: MCP 2024-11-05.
"""

import sys
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "prioritize-mcp"
SERVER_VERSION = "1.0.0"

# ── Инструменты ──────────────────────────────────────

TOOLS = [
    {
        "name": "prioritize_issues",
        "description": "Приоритизировать задачи с помощью LLM. Группирует в: сегодня / на неделе / бэклог.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "description": "Список задач (объекты с полями id/idReadable, summary, description, customFields)",
                    "items": {"type": "object"},
                },
                "context": {
                    "type": "string",
                    "description": "Дополнительный контекст для приоритизации (опционально)",
                },
            },
            "required": ["issues"],
        },
    },
]


# ── LLM ──────────────────────────────────────────────

def _get_llm():
    """Ленивое создание LLM-модели."""
    if not hasattr(_get_llm, "_llm"):
        from dotenv import load_dotenv
        load_dotenv()
        from llm import OpenAIModel
        _get_llm._llm = OpenAIModel(model="gpt-4o")
    return _get_llm._llm


def _format_issues_for_llm(issues):
    """Подготовить задачи для LLM-промпта."""
    lines = []
    for issue in issues:
        issue_id = issue.get("idReadable", issue.get("id", "?"))
        summary = issue.get("summary", "No summary")
        description = issue.get("description", "")
        priority = ""
        state = ""
        for cf in issue.get("customFields", []):
            name = cf.get("name", "")
            val = cf.get("value")
            if val and isinstance(val, dict):
                val = val.get("name", "")
            elif val is None:
                val = ""
            if name.lower() == "priority":
                priority = val
            elif name.lower() == "state":
                state = val
        desc_short = (description[:150] + "...") if description and len(description) > 150 else (description or "")
        line = f"- [{issue_id}] {summary}"
        if priority:
            line += f" | Priority: {priority}"
        if state:
            line += f" | State: {state}"
        if desc_short:
            line += f"\n  Description: {desc_short}"
        lines.append(line)
    return "\n".join(lines)


# ── Обработчики ──────────────────────────────────────

def _prioritize(args):
    """Приоритизировать задачи через LLM."""
    issues = args.get("issues", [])
    context = args.get("context", "")

    if issues:
        issues_text = _format_issues_for_llm(issues)
        issues_block = f"Задачи:\n{issues_text}"
    else:
        issues_block = "Список задач пуст — задачи не были найдены."

    context_block = ""
    if context:
        context_block = f"\nДополнительный контекст: {context}\n"

    prompt = f"""Ты — менеджер проекта. Проанализируй список задач и распредели их по приоритету.

{issues_block}
{context_block}
Сгруппируй задачи в три категории:

## Сделать сегодня (критично / блокирует других)
Задачи, которые требуют немедленного внимания.

## На этой неделе (важно, но не горит)
Задачи, которые нужно завершить в ближайшие дни.

## Бэклог (можно отложить)
Задачи, которые можно спланировать позже.

Для каждой задачи укажи [ID] и кратко объясни почему она в этой категории.
Отвечай на русском языке в формате Markdown."""

    llm = _get_llm()
    result = llm.generate(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    return {
        "prioritized_plan": result["text"],
        "issues_count": len(issues),
        "llm_tokens": {
            "input": result.get("input_tokens", 0),
            "output": result.get("output_tokens", 0),
        },
    }


_TOOL_HANDLERS = {
    "prioritize_issues": lambda args: _prioritize(args),
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
