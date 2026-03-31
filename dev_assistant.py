"""
Ассистент разработчика с RAG + Git MCP.

Индексирует README.md и docs/ для ответов на вопросы о проекте.
Подключается к git_mcp для получения информации о репозитории.

Команды:
  /help <вопрос>  — задать вопрос о проекте (RAG + git-контекст)
  /index          — переиндексировать документацию
  /stats          — статистика индекса
  /git            — показать текущую ветку и статус
  exit            — выход
"""

import os
import sys
import json
import subprocess
import textwrap

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── RAG ──────────────────────────────────────────────────
from indexing.pipeline import IndexingPipeline

DB_PATH = os.path.join(PROJECT_ROOT, "dev_assistant_index.db")

# ── Git MCP (вызов напрямую без полного MCP-протокола) ───
sys.path.insert(0, PROJECT_ROOT)
from git_mcp.git_mcp_server import (
    _git_branch,
    _git_branches,
    _git_status,
    _git_log,
    _git_diff,
    _git_list_files,
)


def get_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def print_wrapped(text, width):
    for paragraph in text.split("\n"):
        if paragraph:
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def index_docs(pipeline):
    """Индексировать README.md, CLAUDE.md и docs/."""
    files = []

    # README.md
    readme = os.path.join(PROJECT_ROOT, "README.md")
    if os.path.exists(readme):
        with open(readme, "r", encoding="utf-8") as f:
            files.append((readme, f.read()))

    # CLAUDE.md
    claude_md = os.path.join(PROJECT_ROOT, "CLAUDE.md")
    if os.path.exists(claude_md):
        with open(claude_md, "r", encoding="utf-8") as f:
            files.append((claude_md, f.read()))

    # docs/
    docs_dir = os.path.join(PROJECT_ROOT, "docs")
    if os.path.isdir(docs_dir):
        scanned = pipeline.scan_directory(docs_dir)
        files.extend(scanned)

    if not files:
        print("[Нет файлов для индексации]")
        return

    result = pipeline.index_files(files, strategy="structural", embed=True, force=True)
    print(f"[Проиндексировано: {result['files_processed']} файлов, "
          f"{result['total_chunks']} чанков, {result['total_embedded']} эмбеддингов]")


def get_git_context():
    """Собрать контекст из git для обогащения ответов."""
    parts = []
    try:
        branch = _git_branch()
        parts.append(f"Текущая ветка: {branch}")
    except Exception:
        parts.append("Git: не удалось получить ветку")

    try:
        status = _git_status()
        if status:
            parts.append(f"Изменённые файлы:\n{status}")
        else:
            parts.append("Рабочее дерево чистое")
    except Exception:
        pass

    try:
        log = _git_log(5)
        if log:
            parts.append(f"Последние коммиты:\n{log}")
    except Exception:
        pass

    try:
        file_list = _git_list_files()
        if file_list:
            files = file_list.split("\n")
            # Группируем по папкам верхнего уровня
            dirs = set()
            root_files = []
            for f in files:
                if "/" in f:
                    dirs.add(f.split("/")[0] + "/")
                else:
                    root_files.append(f)
            structure = sorted(dirs) + sorted(root_files)
            parts.append(f"Структура проекта ({len(files)} файлов):\n" + "\n".join(structure))
    except Exception:
        pass

    return "\n\n".join(parts)


def handle_help(question, pipeline):
    """Ответить на вопрос о проекте через RAG + git-контекст."""
    if not question:
        print("Использование: /help <вопрос о проекте>")
        print("Примеры:")
        print("  /help Какие стратегии контекста есть?")
        print("  /help Как добавить новый MCP сервер?")
        print("  /help Какая структура проекта?")
        return

    # Собираем git-контекст
    git_context = get_git_context()

    # Ищем в RAG
    results = pipeline.search(question, top_k=5)
    has_docs = bool(results) and max(c.get("similarity", 0) for c in results) > 0.25

    # Формируем контекст из чанков
    doc_context = ""
    sources = []
    if has_docs:
        doc_parts = []
        for i, chunk in enumerate(results, 1):
            source = chunk.get("source_file", "unknown")
            sim = chunk.get("similarity", 0)
            doc_parts.append(f"[Документ {i}] (файл: {os.path.basename(source)}, "
                           f"релевантность: {sim:.3f})\n{chunk['text']}")
            sources.append(f"  {os.path.basename(source)} (сходство: {sim:.3f})")
        doc_context = "\n\n---\n\n".join(doc_parts)

    # Формируем промпт для LLM
    system_prompt = (
        "Ты — ассистент разработчика. Отвечай на вопросы о проекте, "
        "используя предоставленную документацию и информацию о репозитории.\n\n"
        "Отвечай на русском языке, конкретно и по делу.\n\n"
    )

    if doc_context:
        system_prompt += f"Документация проекта:\n\n{doc_context}\n\n"

    system_prompt += f"Информация о репозитории:\n\n{git_context}"

    # Вызываем LLM
    from openai import OpenAI
    client = OpenAI()

    print("\n  Думаю...")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=2048,
    )

    answer = resp.choices[0].message.content
    width = get_width()
    print()
    print_wrapped(answer, width)

    if sources:
        print(f"\n  Источники:")
        for s in sources[:3]:
            print(f"  {s}")

    tokens_in = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens
    print(f"\n  Токены: {tokens_in} вход / {tokens_out} выход")


def handle_git():
    """Показать git-информацию."""
    try:
        print(f"\n  Ветка: {_git_branch()}")
    except Exception as e:
        print(f"  Ошибка: {e}")

    try:
        branches = _git_branches()
        print(f"\n  Все ветки:\n{branches}")
    except Exception:
        pass

    try:
        status = _git_status()
        if status:
            print(f"\n  Статус:\n{status}")
        else:
            print("\n  Рабочее дерево чистое")
    except Exception:
        pass

    try:
        log = _git_log(5)
        print(f"\n  Последние коммиты:\n{log}")
    except Exception:
        pass


def main():
    print("=" * 60)
    print("  Ассистент разработчика")
    print("  RAG + Git MCP")
    print("=" * 60)

    pipeline = IndexingPipeline(db_path=DB_PATH)

    # Автоматическая индексация при первом запуске
    stats_list = pipeline.get_stats()
    total_chunks = sum(s.get("chunks", 0) for s in stats_list)
    total_files = sum(s.get("files", 0) for s in stats_list)
    if total_chunks == 0:
        print("\nПервый запуск — индексирую документацию...")
        index_docs(pipeline)
    else:
        print(f"\nИндекс: {total_chunks} чанков, {total_files} файлов")

    # Показываем git-контекст
    try:
        branch = _git_branch()
        print(f"Git: ветка '{branch}'")
    except Exception:
        print("Git: не удалось подключиться")

    print("\nКоманды:")
    print("  /help <вопрос>  — спросить о проекте")
    print("  /index          — переиндексировать документацию")
    print("  /stats          — статистика индекса")
    print("  /git            — информация о репозитории")
    print("  exit            — выход")
    print()

    while True:
        try:
            user_input = input("dev> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Выход.")
            break

        if user_input.lower() == "/index":
            index_docs(pipeline)
            continue

        if user_input.lower() == "/stats":
            stats_list = pipeline.get_stats()
            total_chunks = sum(s.get("chunks", 0) for s in stats_list)
            total_files = sum(s.get("files", 0) for s in stats_list)
            print(f"\n  Файлов:  {total_files}")
            print(f"  Чанков:  {total_chunks}")
            for s in stats_list:
                print(f"    {s['strategy']}: {s['chunks']} чанков, {s['files']} файлов")
            print(f"  БД: {DB_PATH}")
            continue

        if user_input.lower() == "/git":
            handle_git()
            continue

        if user_input.lower().startswith("/help"):
            question = user_input[5:].strip()
            handle_help(question, pipeline)
            continue

        # Если не команда — тоже обрабатываем как вопрос
        handle_help(user_input, pipeline)

    pipeline.close()


if __name__ == "__main__":
    main()
