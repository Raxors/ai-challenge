#!/usr/bin/env python3
"""
CLI для AI-ассистента по работе с файлами проекта.

Сценарии (цель → ассистент сам работает с файлами):
  /find <символ>           — найти все использования символа в проекте
  /readme [dir]            — сгенерировать README на основе кода
  /changelog [branch]      — сгенерировать changelog из git
  /check <правила>         — проверить файлы на соответствие правилам
  /update-docs <doc> <src> — обновить документацию по коду

Файловые команды:
  /tree [dir]              — дерево файлов
  /read <path>             — прочитать файл
  /grep <pattern>          — поиск по содержимому
  /analyze <path>          — структурный анализ файла
  /diff <a> <b>            — diff двух файлов
  /git [mode]              — git-изменения (status/diff/log)

  /help                    — справка
  /quit                    — выход
"""

import os
import sys
import json
import textwrap

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from file_assistant.file_agent import FileAgent

W = 76


def header(title):
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def section(title):
    print(f"\n  {'─' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'─' * (W - 4)}")


def wrap_print(text, indent=2):
    for line in text.split("\n"):
        if line.strip():
            wrapped = textwrap.fill(line, width=W - indent, initial_indent=" " * indent,
                                    subsequent_indent=" " * indent)
            print(wrapped)
        else:
            print()


def print_help():
    print("""
  Сценарии (ассистент сам работает с файлами):
    /find <символ>                — найти все использования в проекте
    /readme [dir]                 — сгенерировать README
    /changelog [branch]           — сгенерировать changelog из git
    /check <правило1; правило2>   — проверить файлы по правилам
    /update-docs <doc> <src1 src2> — обновить документацию по коду

  Файловые команды:
    /tree [dir] [depth]           — дерево файлов
    /read <path> [start:end]      — прочитать файл
    /grep <pattern> [ext]         — поиск по содержимому
    /analyze <path>               — структурный анализ
    /diff <file_a> <file_b>       — diff двух файлов
    /git [status|diff|log|staged] — git-изменения

    /help    — справка
    /quit    — выход
""")


def main():
    header("AI-АССИСТЕНТ ДЛЯ РАБОТЫ С ФАЙЛАМИ ПРОЕКТА")
    print("  Подключение к серверу...")

    agent = FileAgent()
    try:
        info = agent.connect()
        print(f"  Подключён. Инструментов: {info.get('tools', 0)}")
    except Exception as e:
        print(f"  Ошибка подключения: {e}")
        return

    print_help()
    print(f"{'─' * W}")

    try:
        while True:
            try:
                user_input = input(f"\n  > ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "/q"):
                break
            if user_input.lower() == "/help":
                print_help()
                continue

            # ── Сценарий: find usages ──
            if user_input.lower().startswith("/find "):
                symbol = user_input[6:].strip()
                section(f"Поиск использований: {symbol}")
                print("  Сканирование проекта...")
                try:
                    result = agent.find_usages(symbol)
                    print(f"  Найдено: {result.get('total_matches', 0)} совпадений в {result.get('files_found', 0)} файлах")
                    print(f"  Просканировано файлов: {result.get('files_searched', 0)}")
                    for u in result.get("usages", []):
                        print(f"    {u['file']}: строки {u['lines']}")
                    section("Анализ LLM")
                    wrap_print(result.get("summary", ""))
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # ── Сценарий: generate README ──
            if user_input.lower().startswith("/readme"):
                parts = user_input.split(maxsplit=1)
                directory = parts[1].strip() if len(parts) > 1 else ""
                section(f"Генерация README{' для ' + directory if directory else ''}")
                print("  Анализ проекта...")
                try:
                    result = agent.generate_readme(directory=directory)
                    print(f"  Проанализировано файлов: {result.get('files_analyzed', 0)}")
                    print(f"  Записано в: {result.get('written_to', '?')}")
                    section("Предпросмотр README")
                    wrap_print(result.get("readme", "")[:2000])
                    if len(result.get("readme", "")) > 2000:
                        print(f"\n  ... (ещё {len(result['readme']) - 2000} символов)")
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # ── Сценарий: generate changelog ──
            if user_input.lower().startswith("/changelog"):
                parts = user_input.split(maxsplit=1)
                branch = parts[1].strip() if len(parts) > 1 else "main"
                section(f"Генерация CHANGELOG (от {branch})")
                print("  Анализ git-истории...")
                try:
                    result = agent.generate_changelog(branch=branch)
                    print(f"  Коммитов проанализировано: {result.get('commits_analyzed', 0)}")
                    print(f"  Записано в: {result.get('written_to', '?')}")
                    section("Предпросмотр CHANGELOG")
                    wrap_print(result.get("changelog", "")[:2000])
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # ── Сценарий: check invariants ──
            if user_input.lower().startswith("/check "):
                rules_str = user_input[7:].strip()
                rules = [r.strip() for r in rules_str.split(";") if r.strip()]
                if not rules:
                    print("  Укажите правила через ';'. Например:")
                    print("  /check Все Python-файлы имеют docstring; Нет print() в коде")
                    continue
                section(f"Проверка правил ({len(rules)})")
                print("  Проверка файлов...")
                try:
                    result = agent.check_invariants(rules)
                    print(f"  Файлов проверено: {result.get('files_checked', 0)}")
                    print(f"  Нарушений: {result.get('violation_count', 0)}")
                    if result.get("violations"):
                        section("Нарушения")
                        for v in result["violations"][:20]:
                            print(f"    [{v['file']}] {v['rule']}")
                            print(f"      {v['detail']}")
                    section("Сводка LLM")
                    wrap_print(result.get("summary", ""))
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # ── Сценарий: update docs ──
            if user_input.lower().startswith("/update-docs "):
                parts = user_input[13:].strip().split()
                if len(parts) < 2:
                    print("  Формат: /update-docs <документ.md> <файл1.py> [файл2.py ...]")
                    continue
                doc_path = parts[0]
                source_paths = parts[1:]
                section(f"Обновление {doc_path}")
                print(f"  Источники: {', '.join(source_paths)}")
                try:
                    result = agent.update_docs(doc_path, source_paths)
                    if "error" in result:
                        print(f"  Ошибка: {result['error']}")
                    else:
                        print(f"  Проанализировано источников: {result.get('sources_analyzed', 0)}")
                        section("Diff")
                        diff = result.get("diff", "")
                        if diff:
                            print(diff[:3000])
                        else:
                            print("  (нет изменений)")
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # ── Файловые команды ──

            # /tree [dir] [depth]
            if user_input.lower().startswith("/tree"):
                parts = user_input.split()
                directory = parts[1] if len(parts) > 1 else ""
                depth = int(parts[2]) if len(parts) > 2 else 3
                result = agent.tree(directory=directory, max_depth=depth)
                if "error" in result:
                    print(f"  Ошибка: {result['error']}")
                else:
                    print(result.get("tree", ""))
                    print(f"\n  Файлов: {result.get('files', 0)}, Директорий: {result.get('directories', 0)}")
                continue

            # /read <path> [start:end]
            if user_input.lower().startswith("/read "):
                parts = user_input[6:].strip().split()
                path = parts[0]
                start = end = None
                if len(parts) > 1 and ":" in parts[1]:
                    s, e = parts[1].split(":", 1)
                    start = int(s) if s else None
                    end = int(e) if e else None
                result = agent.read(path, start_line=start, end_line=end)
                if "error" in result:
                    print(f"  Ошибка: {result['error']}")
                else:
                    print(result.get("content", ""))
                    print(f"\n  [{result.get('total_lines', '?')} строк]")
                continue

            # /grep <pattern> [ext]
            if user_input.lower().startswith("/grep "):
                parts = user_input[6:].strip().split()
                pattern = parts[0]
                exts = [parts[1]] if len(parts) > 1 else None
                result = agent.grep(pattern, extensions=exts, context_lines=1)
                if "error" in result:
                    print(f"  Ошибка: {result['error']}")
                else:
                    for m in result.get("results", [])[:30]:
                        print(f"  {m['file']}:{m['line']}  {m['match']}")
                    print(f"\n  Найдено: {result.get('count', 0)} (файлов: {result.get('files_searched', 0)})")
                continue

            # /analyze <path>
            if user_input.lower().startswith("/analyze "):
                path = user_input[9:].strip()
                result = agent.analyze(path)
                if "error" in result:
                    print(f"  Ошибка: {result['error']}")
                else:
                    print(f"  Файл: {result.get('path')}")
                    print(f"  Строк: {result.get('total_lines', 0)} (пустых: {result.get('blank_lines', 0)})")
                    print(f"  Размер: {result.get('size_bytes', 0)} байт")
                    s = result.get("structure", {})
                    stype = s.get("type", "?")
                    print(f"  Тип: {stype}")
                    if stype == "python":
                        print(f"  Docstring модуля: {'Да' if s.get('module_docstring') else 'Нет'}")
                        print(f"  Импортов: {len(s.get('imports', []))}")
                        print(f"  Классов: {s.get('total_classes', 0)}, Функций: {s.get('total_functions', 0)}")
                        for cls in s.get("classes", []):
                            print(f"    class {cls['name']} (строка {cls['line']}, методов: {len(cls.get('methods', []))})")
                        for func in s.get("functions", []):
                            print(f"    def {func['name']}({', '.join(func.get('args', []))}) — строка {func['line']}")
                    elif stype == "markdown":
                        for h in s.get("headers", []):
                            indent = "  " * h["level"]
                            print(f"  {indent}{'#' * h['level']} {h['title']} (строка {h['line']})")
                continue

            # /diff <a> <b>
            if user_input.lower().startswith("/diff "):
                parts = user_input[6:].strip().split()
                if len(parts) < 2:
                    print("  Формат: /diff <файл_a> <файл_b>")
                    continue
                result = agent.diff(parts[0], path_b=parts[1])
                print(result.get("diff", ""))
                continue

            # /git [mode]
            if user_input.lower().startswith("/git"):
                parts = user_input.split()
                mode = parts[1] if len(parts) > 1 else "status"
                result = agent.git_changes(mode=mode)
                if "error" in result:
                    print(f"  Ошибка: {result['error']}")
                elif mode == "status":
                    for f in result.get("files", []):
                        print(f"  {f['status']} {f['path']}")
                    print(f"\n  Файлов: {result.get('count', 0)}")
                elif mode == "log":
                    for c in result.get("commits", []):
                        print(f"  {c['hash']} {c['message']}")
                else:
                    print(result.get("stat", ""))
                    diff = result.get("diff", "")
                    if diff:
                        print(diff[:3000])
                continue

            # ── Свободный запрос → автономный агент ──
            section(f"Цель: {user_input}")
            print("  Планирование и выполнение...")

            step_num = [0]

            def on_step(info):
                step_num[0] += 1
                tool = info.get("tool", "?")
                args = info.get("args", {})
                # Краткое описание аргументов
                brief = ""
                if "path" in args:
                    brief = args["path"]
                elif "pattern" in args:
                    brief = args["pattern"]
                elif "paths" in args:
                    brief = f"{len(args['paths'])} файлов"
                elif "mode" in args:
                    brief = args["mode"]
                print(f"    шаг {step_num[0]}: {tool}({brief})")

            try:
                result = agent.execute_goal(user_input, on_step=on_step)

                section("Результат")
                wrap_print(result.get("answer", "Нет ответа"))

                steps = result.get("steps", [])
                iters = result.get("iterations", 0)
                tokens = result.get("tokens", {})
                print(f"\n  Шагов: {len(steps)}, Итераций LLM: {iters}")
                if tokens:
                    print(f"  Токены: {tokens.get('input', 0)} вход / {tokens.get('output', 0)} выход")
            except Exception as e:
                print(f"  Ошибка: {e}")

    except KeyboardInterrupt:
        print("\n")
    finally:
        agent.close()
        print("  До свидания!")


if __name__ == "__main__":
    main()
