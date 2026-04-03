"""
Агент для работы с файлами проекта.

Оркестрирует MCP-сервер file_ops + LLM для выполнения
высокоуровневых задач:
  - Найти все использования компонента/API
  - Сгенерировать README/changelog/ADR
  - Проверить файлы на соответствие правилам
  - Подготовить diff/список изменений
  - Обновить документацию по коду
"""

import os
import sys
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp_client import MCPClient
from openai import OpenAI


class FileAgent:
    """AI-ассистент для работы с файлами проекта."""

    def __init__(self):
        self._server_dir = os.path.dirname(os.path.abspath(__file__))
        self._client = None
        self._llm = OpenAI()

    def connect(self):
        cmd = [sys.executable, os.path.join(self._server_dir, "file_ops_mcp_server.py")]
        self._client = MCPClient(server_cmd=cmd)
        self._client.connect()
        return {"status": "connected", "tools": len(self._client.list_tools())}

    def close(self):
        if self._client:
            self._client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def _call(self, tool, args=None):
        result = self._client.call_tool(tool, args or {})
        content = result.get("content", [])
        if content:
            try:
                return json.loads(content[0].get("text", "{}"))
            except json.JSONDecodeError:
                return {"raw": content[0].get("text", "")}
        return result

    def _llm_call(self, system, user, max_tokens=2048):
        resp = self._llm.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    # ── Прямые файловые операции ──────────────────────

    def read(self, path, start_line=None, end_line=None):
        args = {"path": path}
        if start_line:
            args["start_line"] = start_line
        if end_line:
            args["end_line"] = end_line
        return self._call("file_read", args)

    def write(self, path, content):
        return self._call("file_write", {"path": path, "content": content})

    def patch(self, path, find, replace, replace_all=False):
        return self._call("file_patch", {"path": path, "find": find, "replace": replace, "all": replace_all})

    def tree(self, directory="", max_depth=3, extensions=None):
        args = {"directory": directory, "max_depth": max_depth}
        if extensions:
            args["extensions"] = extensions
        return self._call("file_tree", args)

    def grep(self, pattern, directory="", extensions=None, context_lines=0, max_results=50):
        args = {"pattern": pattern, "directory": directory, "max_results": max_results, "context_lines": context_lines}
        if extensions:
            args["extensions"] = extensions
        return self._call("file_grep", args)

    def glob(self, pattern, directory=""):
        return self._call("file_glob", {"pattern": pattern, "directory": directory})

    def diff(self, path_a, path_b=None, content_b=None):
        args = {"path_a": path_a}
        if path_b:
            args["path_b"] = path_b
        if content_b:
            args["content_b"] = content_b
        return self._call("file_diff", args)

    def analyze(self, path):
        return self._call("file_analyze", {"path": path})

    def git_changes(self, mode="status", branch="main", n=10):
        return self._call("file_git_changes", {"mode": mode, "branch": branch, "n": n})

    def multi_read(self, paths):
        return self._call("file_multi_read", {"paths": paths})

    # ── Высокоуровневые сценарии ──────────────────────

    def find_usages(self, symbol, directory="", extensions=None):
        """
        Сценарий 1: Найти все использования символа/API/компонента.

        1. grep по символу
        2. Для каждого файла — analyze структуру
        3. LLM суммирует: где используется, как, зависимости
        """
        grep_result = self.grep(
            pattern=symbol,
            directory=directory,
            extensions=extensions or [".py", ".js", ".ts"],
            context_lines=2,
            max_results=30,
        )

        matches = grep_result.get("results", [])
        if not matches:
            return {
                "symbol": symbol,
                "usages": [],
                "summary": f"Символ '{symbol}' не найден в проекте.",
                "files_searched": grep_result.get("files_searched", 0),
            }

        # Группировка по файлам
        by_file = {}
        for m in matches:
            f = m["file"]
            if f not in by_file:
                by_file[f] = []
            by_file[f].append(m)

        # Анализ каждого файла
        file_analyses = {}
        for fpath in list(by_file.keys())[:10]:
            file_analyses[fpath] = self.analyze(fpath)

        # LLM суммирование
        context = json.dumps({
            "symbol": symbol,
            "matches_by_file": {
                f: [{"line": m["line"], "match": m["match"], "context": m.get("context")} for m in ms]
                for f, ms in by_file.items()
            },
            "file_structures": {
                f: a.get("structure", {})
                for f, a in file_analyses.items()
            },
        }, ensure_ascii=False, indent=2)

        summary = self._llm_call(
            system=(
                "Ты — ассистент для анализа кодовой базы. "
                "Дан символ и его вхождения по файлам проекта. "
                "Суммируй: где символ определён, где импортируется, где используется. "
                "Укажи файл:строку. Ответ на русском, кратко."
            ),
            user=f"Проанализируй использование символа:\n\n{context}",
            max_tokens=1024,
        )

        return {
            "symbol": symbol,
            "files_found": len(by_file),
            "total_matches": len(matches),
            "usages": [
                {"file": f, "lines": [m["line"] for m in ms]}
                for f, ms in by_file.items()
            ],
            "summary": summary,
            "files_searched": grep_result.get("files_searched", 0),
        }

    def generate_readme(self, directory="", output_path=None):
        """
        Сценарий 2: Сгенерировать README на основе анализа проекта.

        1. Дерево файлов
        2. Анализ ключевых файлов (setup.py, main, __init__, etc)
        3. Git log для описания активности
        4. LLM генерирует README
        5. Записывает в файл
        """
        # Дерево проекта
        tree_result = self.tree(directory=directory, max_depth=2, extensions=[".py", ".md", ".json", ".txt"])

        # Ищем ключевые файлы
        key_patterns = ["*.md", "*.txt", "*.cfg", "*.toml", "setup.py", "requirements.txt", "pyproject.toml"]
        key_files = {}
        for pat in key_patterns:
            found = self.glob(pat, directory=directory)
            for m in found.get("matches", [])[:3]:
                content = self.read(m["path"])
                if "error" not in content:
                    key_files[m["path"]] = content.get("content", "")[:2000]

        # Анализ Python-файлов
        py_files = self.glob("*.py", directory=directory)
        py_analyses = {}
        for m in py_files.get("matches", [])[:15]:
            a = self.analyze(m["path"])
            if "error" not in a:
                py_analyses[m["path"]] = {
                    "lines": a.get("total_lines", 0),
                    "structure": a.get("structure", {}),
                }

        # Git log
        git_log = self.git_changes(mode="log", n=5)

        context = json.dumps({
            "tree": tree_result.get("tree", ""),
            "key_files": {k: v[:1000] for k, v in key_files.items()},
            "python_files": py_analyses,
            "git_log": git_log.get("commits", []),
        }, ensure_ascii=False, indent=2)

        readme = self._llm_call(
            system=(
                "Ты — технический писатель. На основе структуры проекта, "
                "ключевых файлов и анализа кода сгенерируй README.md. "
                "Включи: описание проекта, структуру, установку, запуск, "
                "основные компоненты. На русском. Формат Markdown."
            ),
            user=f"Сгенерируй README.md для проекта:\n\n{context}",
            max_tokens=2048,
        )

        result = {"readme": readme, "files_analyzed": len(py_analyses)}

        # Записываем файл
        out_path = output_path or (os.path.join(directory, "README.md") if directory else "README_GENERATED.md")
        write_result = self.write(out_path, readme)
        result["written_to"] = out_path
        result["write_result"] = write_result

        return result

    def check_invariants(self, rules, directory="", extensions=None):
        """
        Сценарий 3: Проверить файлы на соответствие правилам.

        rules — список правил, например:
          - "Все Python-файлы должны иметь module docstring"
          - "Нет print() в production-коде"
          - "Все функции имеют docstring"
          - "Нет TODO/FIXME в коде"
        """
        exts = extensions or [".py"]
        files = self.glob(f"*{exts[0]}", directory=directory)

        violations = []
        files_checked = 0

        for m in files.get("matches", [])[:30]:
            fpath = m["path"]
            analysis = self.analyze(fpath)
            if "error" in analysis:
                continue

            content_result = self.read(fpath)
            content = content_result.get("content", "")
            files_checked += 1

            structure = analysis.get("structure", {})

            for rule in rules:
                rule_lower = rule.lower()

                # Встроенные проверки
                if "docstring" in rule_lower and "module" in rule_lower:
                    if structure.get("type") == "python" and not structure.get("module_docstring"):
                        violations.append({
                            "file": fpath,
                            "rule": rule,
                            "detail": "Отсутствует module docstring",
                        })

                if "docstring" in rule_lower and "функц" in rule_lower:
                    if structure.get("type") == "python":
                        for func in structure.get("functions", []):
                            if not func.get("has_docstring"):
                                violations.append({
                                    "file": fpath,
                                    "rule": rule,
                                    "detail": f"Функция {func['name']} (строка {func['line']}) без docstring",
                                })
                        for cls in structure.get("classes", []):
                            for method in cls.get("methods", []):
                                if not method.get("has_docstring") and not method["name"].startswith("_"):
                                    violations.append({
                                        "file": fpath,
                                        "rule": rule,
                                        "detail": f"Метод {cls['name']}.{method['name']} (строка {method['line']}) без docstring",
                                    })

                if "print" in rule_lower and "print()" in rule_lower:
                    for i, line in enumerate(content.splitlines(), 1):
                        stripped = line.lstrip()
                        if stripped.startswith("print(") and not stripped.startswith("#"):
                            violations.append({
                                "file": fpath,
                                "rule": rule,
                                "detail": f"Строка {i}: {stripped[:80]}",
                            })

                if "todo" in rule_lower or "fixme" in rule_lower:
                    for i, line in enumerate(content.splitlines(), 1):
                        if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', line):
                            violations.append({
                                "file": fpath,
                                "rule": rule,
                                "detail": f"Строка {i}: {line.strip()[:80]}",
                            })

        # LLM-сводка
        if violations:
            summary = self._llm_call(
                system="Суммируй нарушения правил в проекте. Кратко, по категориям. Русский язык.",
                user=json.dumps({"rules": rules, "violations": violations[:50]}, ensure_ascii=False),
                max_tokens=512,
            )
        else:
            summary = "Все проверенные файлы соответствуют правилам."

        return {
            "files_checked": files_checked,
            "rules": rules,
            "violations": violations,
            "violation_count": len(violations),
            "summary": summary,
        }

    def generate_changelog(self, branch="main", output_path=None):
        """
        Сценарий 4: Сгенерировать changelog на основе git diff.

        1. Git log + diff от ветки
        2. Анализ изменённых файлов
        3. LLM генерирует changelog
        4. Записывает в файл
        """
        # Git log
        log_result = self.git_changes(mode="log", n=20)
        commits = log_result.get("commits", [])

        # Git diff от основной ветки
        diff_result = self.git_changes(mode="diff_branch", branch=branch)

        # Анализ stat
        stat = diff_result.get("stat", "")
        diff_text = diff_result.get("diff", "")[:8000]

        context = json.dumps({
            "commits": commits,
            "diff_stat": stat,
            "diff_preview": diff_text[:5000],
        }, ensure_ascii=False, indent=2)

        changelog = self._llm_call(
            system=(
                "Ты — технический писатель. На основе git-коммитов и diff "
                "сгенерируй CHANGELOG в формате Keep a Changelog. "
                "Группируй по: Added, Changed, Fixed, Removed. "
                "На русском. Формат Markdown."
            ),
            user=f"Сгенерируй CHANGELOG:\n\n{context}",
            max_tokens=2048,
        )

        result = {
            "changelog": changelog,
            "commits_analyzed": len(commits),
        }

        out_path = output_path or "CHANGELOG_GENERATED.md"
        write_result = self.write(out_path, changelog)
        result["written_to"] = out_path
        result["write_result"] = write_result

        return result

    def update_docs(self, doc_path, source_paths):
        """
        Сценарий 5: Обновить документацию на основе изменений в коде.

        1. Прочитать существующий документ
        2. Проанализировать исходные файлы
        3. LLM определяет, что устарело / не хватает
        4. LLM генерирует обновлённый документ
        5. Показывает diff и записывает
        """
        # Читаем текущий документ
        doc = self.read(doc_path)
        if "error" in doc:
            return {"error": f"Документ не найден: {doc_path}"}
        doc_content = doc.get("content", "")

        # Анализ исходников
        source_data = {}
        for sp in source_paths:
            analysis = self.analyze(sp)
            content = self.read(sp)
            source_data[sp] = {
                "structure": analysis.get("structure", {}),
                "content_preview": content.get("content", "")[:3000],
            }

        context = json.dumps({
            "document_path": doc_path,
            "document_content": doc_content,
            "sources": source_data,
        }, ensure_ascii=False, indent=2)

        updated = self._llm_call(
            system=(
                "Ты — технический писатель. Дан существующий документ и текущий код. "
                "Обнови документ, чтобы он соответствовал коду: "
                "добавь недостающие разделы, обнови устаревшие описания, "
                "удали неактуальное. Верни ПОЛНЫЙ обновлённый документ. "
                "Формат как у оригинала."
            ),
            user=f"Обнови документацию:\n\n{context}",
            max_tokens=3000,
        )

        # Diff
        diff_result = self.diff(doc_path, content_b=updated)
        diff_text = diff_result.get("diff", "")

        # Записываем
        write_result = self.write(doc_path, updated)

        return {
            "path": doc_path,
            "diff": diff_text,
            "sources_analyzed": len(source_data),
            "write_result": write_result,
        }

    def execute_goal(self, goal, on_step=None):
        """
        Автономное выполнение цели на естественном языке.

        LLM сам решает, какие файловые операции выполнить.
        Использует OpenAI function calling в цикле (до 15 итераций).

        on_step — callback(step_info) для отображения прогресса.
        """
        # Описание доступных функций для OpenAI
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "file_read",
                    "description": "Прочитать файл проекта",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Путь к файлу"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_write",
                    "description": "Записать/создать файл",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_patch",
                    "description": "Найти и заменить текст в файле. Возвращает diff.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                            "all": {"type": "boolean"},
                        },
                        "required": ["path", "find", "replace"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_tree",
                    "description": "Дерево файлов проекта",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string"},
                            "max_depth": {"type": "integer"},
                            "extensions": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_grep",
                    "description": "Поиск по содержимому файлов (regex)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "directory": {"type": "string"},
                            "extensions": {"type": "array", "items": {"type": "string"}},
                            "context_lines": {"type": "integer"},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_glob",
                    "description": "Найти файлы по glob-паттерну",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "directory": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_diff",
                    "description": "Unified diff между двумя файлами",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path_a": {"type": "string"},
                            "path_b": {"type": "string"},
                        },
                        "required": ["path_a"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_analyze",
                    "description": "Структурный анализ файла (Python: классы/функции, MD: заголовки)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_git_changes",
                    "description": (
                        "Git изменения. Режимы: "
                        "'all' — ПОЛНЫЙ обзор (status + log + changed_files список + per_file_diffs для каждого файла), ЛУЧШИЙ выбор для обзора всех изменений; "
                        "'status' — только git status; "
                        "'diff' — unstaged diff; "
                        "'staged' — staged diff; "
                        "'log' — последние коммиты; "
                        "'diff_branch' — diff от ветки; "
                        "'diff_file' — diff конкретного файла (нужен параметр file)"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["all", "status", "diff", "staged", "log", "diff_branch", "diff_file"]},
                            "branch": {"type": "string", "description": "Ветка для сравнения (default: main)"},
                            "n": {"type": "integer", "description": "Кол-во коммитов для log"},
                            "file": {"type": "string", "description": "Путь к файлу (для diff_file)"},
                        },
                        "required": ["mode"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_multi_read",
                    "description": "Прочитать несколько файлов за раз",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["paths"],
                    },
                },
            },
        ]

        system_prompt = (
            "Ты — AI-ассистент для работы с файлами проекта. "
            "У тебя есть набор инструментов для чтения, записи, поиска, анализа файлов и git. "
            "Пользователь описывает цель на естественном языке — ты сам планируешь и выполняешь шаги.\n\n"
            "ПРАВИЛА:\n"
            "1. АКТИВНО используй инструменты! Не спрашивай пользователя — действуй сам.\n"
            "2. Для обзора ВСЕХ изменений используй file_git_changes(mode='all') — он вернёт "
            "status, коммиты, список изменённых файлов с типом (added/modified/deleted) И diff для каждого файла.\n"
            "3. Если нужны diff конкретного файла — file_git_changes(mode='diff_file', file='path').\n"
            "4. Для изучения проекта — file_tree, file_grep, file_analyze.\n"
            "5. Для создания файлов — file_write. Для изменения — file_patch.\n"
            "6. НЕ пропускай файлы! Если задача 'найди все diff' — используй режим 'all' и обработай ВСЕ файлы из changed_files.\n"
            "7. Отвечай на русском. В финальном ответе опиши что было сделано."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        steps = []
        max_iterations = 15

        for iteration in range(max_iterations):
            resp = self._llm.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                messages=messages,
                tools=tools,
            )

            choice = resp.choices[0]

            # Если нет tool_calls — финальный ответ
            if not choice.message.tool_calls:
                final_text = choice.message.content or ""
                return {
                    "answer": final_text,
                    "steps": steps,
                    "iterations": iteration + 1,
                    "tokens": {
                        "input": resp.usage.prompt_tokens,
                        "output": resp.usage.completion_tokens,
                    },
                }

            # Добавляем assistant message
            messages.append({
                "role": "assistant",
                "content": choice.message.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            })

            # Выполняем каждый tool call
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                step_info = {"tool": tool_name, "args": tool_args}
                if on_step:
                    on_step(step_info)
                steps.append(step_info)

                # Вызов MCP
                tool_result = self._call(tool_name, tool_args)

                # Ограничиваем размер результата для контекста
                result_text = json.dumps(tool_result, ensure_ascii=False)
                if len(result_text) > 8000:
                    result_text = result_text[:8000] + "\n... (обрезано)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        return {
            "answer": "[Превышен лимит итераций]",
            "steps": steps,
            "iterations": max_iterations,
        }


# Для удобства импорта в тестах
import re
