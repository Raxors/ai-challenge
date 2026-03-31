"""
MCP-сервер для работы с Git.

Предоставляет инструменты:
  - git_branch       — текущая ветка
  - git_branches     — список всех веток
  - git_status       — статус рабочего дерева
  - git_log          — последние коммиты
  - git_diff         — diff текущих изменений
  - git_list_files   — список файлов в репозитории
"""

import os
import sys
import subprocess

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer

TOOLS = [
    {
        "name": "git_branch",
        "description": "Показать текущую git-ветку",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "git_branches",
        "description": "Показать все git-ветки (локальные)",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "git_status",
        "description": "Показать git status (изменённые, добавленные, удалённые файлы)",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "git_log",
        "description": "Показать последние коммиты",
        "inputSchema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Количество коммитов (по умолчанию 10)",
                },
            },
        },
    },
    {
        "name": "git_diff",
        "description": "Показать diff текущих изменений (unstaged или staged)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Показать staged изменения (по умолчанию false — unstaged)",
                },
            },
        },
    },
    {
        "name": "git_list_files",
        "description": "Список отслеживаемых файлов в репозитории (или в указанной директории)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Поддиректория для фильтрации (по умолчанию — весь репозиторий)",
                },
            },
        },
    },
]


def _run_git(*args):
    """Выполнить git-команду и вернуть stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout.strip()


def _git_branch():
    return _run_git("rev-parse", "--abbrev-ref", "HEAD")


def _git_branches():
    return _run_git("branch", "--list")


def _git_status():
    return _run_git("status", "--short")


def _git_log(count=10):
    return _run_git("log", f"--oneline", f"-{count}")


def _git_diff(staged=False):
    if staged:
        return _run_git("diff", "--cached")
    return _run_git("diff")


def _git_list_files(directory=None):
    if directory:
        return _run_git("ls-files", directory)
    return _run_git("ls-files")


_TOOL_HANDLERS = {
    "git_branch": lambda args: _git_branch(),
    "git_branches": lambda args: _git_branches(),
    "git_status": lambda args: _git_status(),
    "git_log": lambda args: _git_log(args.get("count", 10)),
    "git_diff": lambda args: _git_diff(args.get("staged", False)),
    "git_list_files": lambda args: _git_list_files(args.get("directory")),
}


server = JSONRPCServer("git_mcp", "1.0.0")
server.set_tools(TOOLS)


def handle_message(raw_message):
    return server.handle_message(raw_message)


def handle_tool_call(name, arguments):
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")
    result = handler(arguments)
    return [{"type": "text", "text": result or "(пусто)"}]


server.handle_tool_call = handle_tool_call

if __name__ == "__main__":
    server.run_stdio()
