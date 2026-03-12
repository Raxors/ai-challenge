#!/usr/bin/env python3
"""
MCP-сервер сохранения файлов — принимает текст, сохраняет в файл.

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

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "filesaver-mcp"
SERVER_VERSION = "1.0.0"

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "pipeline_output")

# ── Инструменты ──────────────────────────────────────

TOOLS = [
    {
        "name": "save_to_file",
        "description": "Сохранить текст в файл. По умолчанию сохраняет в pipeline_output/.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Текст для сохранения",
                },
                "filename": {
                    "type": "string",
                    "description": "Имя файла (по умолчанию: result_ДАТА.md)",
                },
                "directory": {
                    "type": "string",
                    "description": "Директория для сохранения (опционально, по умолчанию pipeline_output/)",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "list_saved_files",
        "description": "Показать список сохранённых файлов в pipeline_output/.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Обработчики ──────────────────────────────────────

def _save_to_file(args):
    """Сохранить контент в файл."""
    content = args["content"]
    filename = args.get("filename")
    directory = args.get("directory", OUTPUT_DIR)

    if not filename:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"result_{timestamp}.md"

    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "filepath": filepath,
        "filename": filename,
        "size_bytes": len(content.encode("utf-8")),
    }


def _list_saved_files(args):
    """Список файлов в output-директории."""
    if not os.path.exists(OUTPUT_DIR):
        return {"files": [], "directory": OUTPUT_DIR}

    files = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            files.append({
                "filename": fname,
                "size_bytes": stat.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(stat.st_mtime)),
            })
    return {"files": files, "directory": OUTPUT_DIR}


_TOOL_HANDLERS = {
    "save_to_file": lambda args: _save_to_file(args),
    "list_saved_files": lambda args: _list_saved_files(args),
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
