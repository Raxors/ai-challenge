"""
Git-инструменты для ассистента разработчика.
Standalone — не зависит от core/jsonrpc или MCP-протокола.
"""

import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_git(*args):
    """Выполнить git-команду и вернуть stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout.strip()


def git_branch():
    """Текущая ветка."""
    return _run_git("rev-parse", "--abbrev-ref", "HEAD")


def git_branches():
    """Список всех локальных веток."""
    return _run_git("branch", "--list")


def git_status():
    """git status --short."""
    return _run_git("status", "--short")


def git_log(count=10):
    """Последние N коммитов (oneline)."""
    return _run_git("log", "--oneline", f"-{count}")


def git_diff(staged=False):
    """Diff текущих изменений."""
    if staged:
        return _run_git("diff", "--cached")
    return _run_git("diff")


def git_list_files(directory=None):
    """Список отслеживаемых файлов."""
    if directory:
        return _run_git("ls-files", directory)
    return _run_git("ls-files")


def git_diff_branch(base="main"):
    """Diff текущей ветки относительно базовой."""
    return _run_git("diff", f"{base}...HEAD")


def git_changed_files(base="main"):
    """Список изменённых файлов относительно базовой ветки."""
    return _run_git("diff", "--name-only", f"{base}...HEAD")


def git_show_file(path, ref="HEAD"):
    """Содержимое файла из определённого коммита."""
    return _run_git("show", f"{ref}:{path}")
