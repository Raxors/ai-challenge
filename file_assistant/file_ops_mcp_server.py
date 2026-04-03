#!/usr/bin/env python3
"""
MCP-сервер файловых операций — чтение, запись, поиск, анализ файлов проекта.

Инструменты:
  - file_read         — прочитать файл
  - file_write        — записать файл (создать или перезаписать)
  - file_patch        — применить патч (найти/заменить)
  - file_tree         — дерево файлов директории
  - file_grep         — поиск по содержимому (regex)
  - file_glob         — поиск файлов по паттерну
  - file_diff         — diff между двумя файлами или файлом и строкой
  - file_analyze      — структурный анализ файла (Python: классы/функции; MD: заголовки)
  - file_git_changes  — список изменённых файлов из git (diff/log)
  - file_multi_read   — прочитать несколько файлов за один вызов
"""

import sys
import json
import os
import re
import subprocess
import fnmatch
import difflib

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "file-ops-mcp"
SERVER_VERSION = "1.0.0"

# Защита: запрещённые директории
_DENY_PATTERNS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".env"}

TOOLS = [
    {
        "name": "file_read",
        "description": "Прочитать файл. Возвращает содержимое, количество строк, размер.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к файлу (относительно корня проекта)"},
                "start_line": {"type": "integer", "description": "Начальная строка (1-based, опционально)"},
                "end_line": {"type": "integer", "description": "Конечная строка (опционально)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Записать содержимое в файл. Создаёт директории при необходимости.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к файлу"},
                "content": {"type": "string", "description": "Содержимое для записи"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_patch",
        "description": "Найти и заменить текст в файле. Возвращает diff.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к файлу"},
                "find": {"type": "string", "description": "Текст для поиска"},
                "replace": {"type": "string", "description": "Текст для замены"},
                "all": {"type": "boolean", "description": "Заменить все вхождения (по умолчанию false)"},
            },
            "required": ["path", "find", "replace"],
        },
    },
    {
        "name": "file_tree",
        "description": "Дерево файлов директории с фильтрацией.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Директория (по умолчанию корень проекта)"},
                "max_depth": {"type": "integer", "description": "Максимальная глубина (по умолчанию 3)"},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Фильтр расширений (например ['.py', '.md'])",
                },
                "ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Директории для игнорирования",
                },
            },
        },
    },
    {
        "name": "file_grep",
        "description": "Поиск по содержимому файлов (regex). Возвращает файлы, строки, контекст.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex-паттерн для поиска"},
                "directory": {"type": "string", "description": "Директория поиска (по умолчанию корень проекта)"},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Фильтр расширений",
                },
                "context_lines": {"type": "integer", "description": "Строк контекста до/после (по умолчанию 0)"},
                "max_results": {"type": "integer", "description": "Максимум результатов (по умолчанию 50)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "file_glob",
        "description": "Найти файлы по glob-паттерну (например '**/*.py', 'tests/*.py').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob-паттерн"},
                "directory": {"type": "string", "description": "Базовая директория"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "file_diff",
        "description": "Unified diff между двумя файлами или файлом и строкой.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path_a": {"type": "string", "description": "Путь к первому файлу"},
                "path_b": {"type": "string", "description": "Путь ко второму файлу (опционально)"},
                "content_b": {"type": "string", "description": "Содержимое для сравнения (вместо path_b)"},
            },
            "required": ["path_a"],
        },
    },
    {
        "name": "file_analyze",
        "description": "Структурный анализ файла: Python (классы, функции, импорты), Markdown (заголовки, секции), общий (строки, размер).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к файлу"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_git_changes",
        "description": "Получить список изменений из git: файлы, diff, коммиты.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Режим: 'status' (рабочая директория), 'diff' (unstaged), 'staged' (staged), 'log' (последние коммиты), 'diff_branch' (от ветки), 'diff_file' (diff конкретного файла), 'all' (полный обзор: status + log + diff stat)",
                },
                "branch": {"type": "string", "description": "Ветка для сравнения (для diff_branch)"},
                "n": {"type": "integer", "description": "Количество коммитов (для log, по умолчанию 10)"},
                "file": {"type": "string", "description": "Путь к файлу (для diff_file)"},
            },
            "required": ["mode"],
        },
    },
    {
        "name": "file_multi_read",
        "description": "Прочитать несколько файлов за один вызов.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Список путей к файлам",
                },
            },
            "required": ["paths"],
        },
    },
]


# ── Утилиты ──────────────────────────────────────────

def _resolve_path(path):
    """Преобразовать относительный путь в абсолютный от корня проекта."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


def _is_safe_path(path):
    """Проверить, что путь внутри проекта."""
    resolved = os.path.realpath(path)
    project_real = os.path.realpath(_PROJECT_ROOT)
    return resolved.startswith(project_real)


def _should_ignore(name):
    """Должна ли директория/файл быть проигнорирована."""
    return name in _DENY_PATTERNS or name.startswith(".")


# ── Обработчики ──────────────────────────────────────

def _file_read(args):
    path = _resolve_path(args["path"])
    if not _is_safe_path(path):
        return {"error": "Путь вне проекта"}
    if not os.path.isfile(path):
        return {"error": f"Файл не найден: {args['path']}"}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    start = args.get("start_line")
    end = args.get("end_line")
    if start or end:
        s = (start or 1) - 1
        e = end or len(lines)
        selected = lines[s:e]
        content = "".join(selected)
        return {
            "path": args["path"],
            "content": content,
            "lines_shown": f"{s+1}-{min(e, len(lines))}",
            "total_lines": len(lines),
        }

    content = "".join(lines)
    return {
        "path": args["path"],
        "content": content,
        "total_lines": len(lines),
        "size_bytes": os.path.getsize(path),
    }


def _file_write(args):
    path = _resolve_path(args["path"])
    if not _is_safe_path(path):
        return {"error": "Путь вне проекта"}

    existed = os.path.isfile(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(args["content"])

    return {
        "path": args["path"],
        "action": "updated" if existed else "created",
        "size_bytes": len(args["content"].encode("utf-8")),
        "lines": args["content"].count("\n") + (1 if args["content"] and not args["content"].endswith("\n") else 0),
    }


def _file_patch(args):
    path = _resolve_path(args["path"])
    if not _is_safe_path(path):
        return {"error": "Путь вне проекта"}
    if not os.path.isfile(path):
        return {"error": f"Файл не найден: {args['path']}"}

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    find_text = args["find"]
    replace_text = args["replace"]
    replace_all = args.get("all", False)

    count = original.count(find_text)
    if count == 0:
        return {"error": "Текст не найден", "path": args["path"]}

    if replace_all:
        modified = original.replace(find_text, replace_text)
    else:
        modified = original.replace(find_text, replace_text, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(modified)

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{args['path']}",
        tofile=f"b/{args['path']}",
    )
    diff_text = "".join(diff)

    return {
        "path": args["path"],
        "replacements": count if replace_all else 1,
        "diff": diff_text,
    }


def _file_tree(args):
    directory = _resolve_path(args.get("directory", ""))
    max_depth = args.get("max_depth", 3)
    extensions = set(args.get("extensions", []))
    ignore = set(args.get("ignore", []))

    if not os.path.isdir(directory):
        return {"error": f"Директория не найдена: {args.get('directory', '.')}"}

    tree = []
    file_count = 0
    dir_count = 0

    def walk(dirpath, depth, prefix=""):
        nonlocal file_count, dir_count
        if depth > max_depth:
            return

        try:
            entries = sorted(os.listdir(dirpath))
        except PermissionError:
            return

        dirs = []
        files = []
        for name in entries:
            if _should_ignore(name) or name in ignore:
                continue
            full = os.path.join(dirpath, name)
            if os.path.isdir(full):
                dirs.append(name)
            elif os.path.isfile(full):
                if extensions:
                    ext = os.path.splitext(name)[1]
                    if ext not in extensions:
                        continue
                files.append(name)

        items = [(d, True) for d in dirs] + [(f, False) for f in files]
        for i, (name, is_dir) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            tree.append(f"{prefix}{connector}{name}{'/' if is_dir else ''}")
            if is_dir:
                dir_count += 1
                extension = "    " if is_last else "│   "
                walk(os.path.join(dirpath, name), depth + 1, prefix + extension)
            else:
                file_count += 1

    root_name = os.path.basename(directory) or "."
    tree.insert(0, f"{root_name}/")
    walk(directory, 1)

    return {
        "tree": "\n".join(tree),
        "files": file_count,
        "directories": dir_count,
    }


def _file_grep(args):
    pattern = args["pattern"]
    directory = _resolve_path(args.get("directory", ""))
    extensions = set(args.get("extensions", []))
    context_lines = args.get("context_lines", 0)
    max_results = args.get("max_results", 50)

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Невалидный regex: {e}"}

    results = []
    files_searched = 0

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not _should_ignore(d)]

        for fname in sorted(files):
            if _should_ignore(fname):
                continue
            if extensions:
                ext = os.path.splitext(fname)[1]
                if ext not in extensions:
                    continue

            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, _PROJECT_ROOT)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except (OSError, UnicodeDecodeError):
                continue

            files_searched += 1

            for i, line in enumerate(lines):
                if regex.search(line):
                    ctx_start = max(0, i - context_lines)
                    ctx_end = min(len(lines), i + context_lines + 1)
                    context = "".join(lines[ctx_start:ctx_end]).rstrip("\n")

                    results.append({
                        "file": rel_path,
                        "line": i + 1,
                        "match": line.rstrip("\n"),
                        "context": context if context_lines > 0 else None,
                    })

                    if len(results) >= max_results:
                        return {
                            "results": results,
                            "count": len(results),
                            "files_searched": files_searched,
                            "truncated": True,
                        }

    return {
        "results": results,
        "count": len(results),
        "files_searched": files_searched,
        "truncated": False,
    }


def _file_glob(args):
    pattern = args["pattern"]
    directory = _resolve_path(args.get("directory", ""))

    matches = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not _should_ignore(d)]
        for fname in sorted(files):
            if _should_ignore(fname):
                continue
            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, _PROJECT_ROOT)
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(fname, pattern):
                stat = os.stat(filepath)
                matches.append({
                    "path": rel_path,
                    "size_bytes": stat.st_size,
                })

    return {"matches": matches, "count": len(matches)}


def _file_diff(args):
    path_a = _resolve_path(args["path_a"])
    if not os.path.isfile(path_a):
        return {"error": f"Файл не найден: {args['path_a']}"}

    with open(path_a, "r", encoding="utf-8", errors="replace") as f:
        lines_a = f.readlines()

    if "path_b" in args and args["path_b"]:
        path_b = _resolve_path(args["path_b"])
        if not os.path.isfile(path_b):
            return {"error": f"Файл не найден: {args['path_b']}"}
        with open(path_b, "r", encoding="utf-8", errors="replace") as f:
            lines_b = f.readlines()
        label_b = args["path_b"]
    elif "content_b" in args:
        lines_b = args["content_b"].splitlines(keepends=True)
        label_b = "(new content)"
    else:
        return {"error": "Нужен path_b или content_b"}

    diff = difflib.unified_diff(
        lines_a, lines_b,
        fromfile=args["path_a"],
        tofile=label_b,
    )
    diff_text = "".join(diff)

    return {
        "diff": diff_text if diff_text else "(файлы идентичны)",
        "lines_a": len(lines_a),
        "lines_b": len(lines_b),
    }


def _file_analyze(args):
    path = _resolve_path(args["path"])
    if not os.path.isfile(path):
        return {"error": f"Файл не найден: {args['path']}"}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = content.splitlines()
    ext = os.path.splitext(path)[1].lower()

    result = {
        "path": args["path"],
        "extension": ext,
        "total_lines": len(lines),
        "blank_lines": sum(1 for l in lines if not l.strip()),
        "size_bytes": os.path.getsize(path),
    }

    if ext == ".py":
        result["structure"] = _analyze_python(content)
    elif ext in (".md", ".markdown"):
        result["structure"] = _analyze_markdown(content)
    elif ext in (".js", ".ts", ".tsx", ".jsx"):
        result["structure"] = _analyze_js(content)
    else:
        result["structure"] = {"type": "generic"}

    return result


def _analyze_python(content):
    """Анализ Python-файла: импорты, классы, функции, docstring."""
    import ast
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"type": "python", "parse_error": str(e)}

    imports = []
    classes = []
    functions = []
    module_docstring = ast.get_docstring(tree)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": item.name,
                        "line": item.lineno,
                        "args": [a.arg for a in item.args.args],
                        "has_docstring": ast.get_docstring(item) is not None,
                    })
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "bases": [_ast_name(b) for b in node.bases],
                "methods": methods,
                "has_docstring": ast.get_docstring(node) is not None,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "args": [a.arg for a in node.args.args],
                "has_docstring": ast.get_docstring(node) is not None,
            })

    return {
        "type": "python",
        "module_docstring": bool(module_docstring),
        "imports": imports,
        "classes": classes,
        "functions": functions,
        "total_classes": len(classes),
        "total_functions": len(functions),
    }


def _ast_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_ast_name(node.value)}.{node.attr}"
    return "?"


def _analyze_markdown(content):
    """Анализ Markdown-файла: заголовки, секции."""
    headers = []
    for i, line in enumerate(content.splitlines(), 1):
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m:
            headers.append({
                "level": len(m.group(1)),
                "title": m.group(2).strip(),
                "line": i,
            })
    return {
        "type": "markdown",
        "headers": headers,
        "total_headers": len(headers),
    }


def _analyze_js(content):
    """Базовый анализ JS/TS: экспорты, функции, классы."""
    exports = re.findall(r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)', content)
    functions = re.findall(r'(?:function|const|let|var)\s+(\w+)\s*[=(]', content)
    classes = re.findall(r'class\s+(\w+)', content)
    imports = re.findall(r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]', content)

    return {
        "type": "javascript",
        "exports": exports,
        "functions": functions[:30],
        "classes": classes,
        "imports": imports,
    }


def _git_run(*cmd):
    """Выполнить git-команду, вернуть stdout."""
    return subprocess.check_output(
        list(cmd), cwd=_PROJECT_ROOT, stderr=subprocess.STDOUT, text=True
    )


def _file_git_changes(args):
    mode = args["mode"]
    branch = args.get("branch", "main")
    n = args.get("n", 10)
    file_path = args.get("file")

    try:
        if mode == "status":
            out = _git_run("git", "status", "--porcelain")
            files = []
            for line in out.strip().splitlines():
                if len(line) >= 4:
                    status = line[:2].strip()
                    filepath = line[3:]
                    files.append({"status": status, "path": filepath})
            return {"mode": "status", "files": files, "count": len(files)}

        elif mode == "diff":
            stat = _git_run("git", "diff", "--stat")
            diff_out = _git_run("git", "diff")
            return {"mode": "diff", "stat": stat, "diff": diff_out[:15000]}

        elif mode == "staged":
            stat = _git_run("git", "diff", "--cached", "--stat")
            diff_out = _git_run("git", "diff", "--cached")
            return {"mode": "staged", "stat": stat, "diff": diff_out[:15000]}

        elif mode == "log":
            out = _git_run("git", "log", f"-{n}", "--oneline", "--no-decorate")
            commits = []
            for line in out.strip().splitlines():
                parts = line.split(" ", 1)
                commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
            return {"mode": "log", "commits": commits, "count": len(commits)}

        elif mode == "diff_branch":
            stat = _git_run("git", "diff", f"{branch}...HEAD", "--stat")
            diff_out = _git_run("git", "diff", f"{branch}...HEAD")
            return {"mode": "diff_branch", "branch": branch, "stat": stat, "diff": diff_out[:20000]}

        elif mode == "diff_file":
            if not file_path:
                return {"error": "Нужен параметр 'file' для режима diff_file"}
            # Пробуем: staged → unstaged → HEAD
            diff_out = _git_run("git", "diff", "--cached", "--", file_path)
            if not diff_out.strip():
                diff_out = _git_run("git", "diff", "--", file_path)
            if not diff_out.strip():
                diff_out = _git_run("git", "diff", "HEAD", "--", file_path)
            return {"mode": "diff_file", "file": file_path, "diff": diff_out[:15000]}

        elif mode == "all":
            # Полный обзор: status + log + diff_branch stat + per-file diff list
            status_out = _git_run("git", "status", "--porcelain")
            files = []
            for line in status_out.strip().splitlines():
                if len(line) >= 4:
                    status = line[:2].strip()
                    filepath = line[3:]
                    files.append({"status": status, "path": filepath})

            log_out = _git_run("git", "log", f"-{n}", "--oneline", "--no-decorate")
            commits = []
            for line in log_out.strip().splitlines():
                parts = line.split(" ", 1)
                commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})

            # Diff stat от ветки (если есть)
            try:
                branch_stat = _git_run("git", "diff", f"{branch}...HEAD", "--stat")
                branch_diff = _git_run("git", "diff", f"{branch}...HEAD", "--name-status")
            except subprocess.CalledProcessError:
                branch_stat = ""
                branch_diff = ""

            # Парсим name-status в список файлов с типом изменения
            changed_files = []
            for line in branch_diff.strip().splitlines():
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    change_type = parts[0].strip()
                    fpath = parts[1].strip()
                    type_map = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}
                    changed_files.append({
                        "path": fpath,
                        "change": type_map.get(change_type[0], change_type),
                    })

            # Per-file diff stat (компактно: кол-во добавленных/удалённых строк)
            diff_numstat = ""
            try:
                diff_numstat = _git_run("git", "diff", f"{branch}...HEAD", "--numstat")
            except subprocess.CalledProcessError:
                pass

            file_stats = {}
            for line in diff_numstat.strip().splitlines():
                parts = line.split("\t")
                if len(parts) == 3:
                    added = parts[0] if parts[0] != "-" else "bin"
                    deleted = parts[1] if parts[1] != "-" else "bin"
                    fpath = parts[2]
                    file_stats[fpath] = f"+{added}/-{deleted}"

            return {
                "mode": "all",
                "hint": "Используй file_git_changes(mode='diff_file', file='path') для подробного diff конкретного файла",
                "status_files": files,
                "status_count": len(files),
                "commits": commits,
                "commits_count": len(commits),
                "branch": branch,
                "branch_stat": branch_stat,
                "changed_files": [
                    {**cf, "stat": file_stats.get(cf["path"], "")}
                    for cf in changed_files
                ],
                "changed_count": len(changed_files),
            }

        else:
            return {"error": f"Неизвестный режим: {mode}"}

    except subprocess.CalledProcessError as e:
        return {"error": f"Git error: {e.output}"}


def _file_multi_read(args):
    paths = args["paths"]
    results = {}
    for p in paths:
        r = _file_read({"path": p})
        results[p] = r
    return {"files": results, "count": len(results)}


_TOOL_HANDLERS = {
    "file_read": _file_read,
    "file_write": _file_write,
    "file_patch": _file_patch,
    "file_tree": _file_tree,
    "file_grep": _file_grep,
    "file_glob": _file_glob,
    "file_diff": _file_diff,
    "file_analyze": _file_analyze,
    "file_git_changes": _file_git_changes,
    "file_multi_read": _file_multi_read,
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
