"""
Ассистент разработчика с RAG + Git.

Индексирует README.md и docs/ для ответов на вопросы о проекте.
Использует git для получения информации о репозитории.

Команды:
  /help <вопрос>  — задать вопрос о проекте (RAG + git-контекст)
  /review [base]  — AI code review текущей ветки (default base: main)
  /docs [file]    — сгенерировать документацию (или сохранить в file)
  /index          — переиндексировать документацию
  /stats          — статистика индекса
  /git            — показать текущую ветку и статус
  exit            — выход
"""

import os
import sys
import textwrap
import readline

from dotenv import load_dotenv

load_dotenv()

# Корень проекта — на уровень выше от папки dev_assistant/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Добавляем корень проекта в sys.path чтобы dev_assistant был доступен как пакет
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dev_assistant.indexing.pipeline import IndexingPipeline
from dev_assistant.git_tools import (
    git_branch,
    git_branches,
    git_status,
    git_log,
    git_diff,
    git_list_files,
)

DB_PATH = os.path.join(PROJECT_ROOT, "dev_assistant_index.db")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), ".dev_history")


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

    for name in ("README.md", "CLAUDE.md"):
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                files.append((path, f.read()))

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
        parts.append(f"Текущая ветка: {git_branch()}")
    except Exception:
        parts.append("Git: не удалось получить ветку")

    try:
        status = git_status()
        if status:
            parts.append(f"Изменённые файлы:\n{status}")
        else:
            parts.append("Рабочее дерево чистое")
    except Exception:
        pass

    try:
        log = git_log(5)
        if log:
            parts.append(f"Последние коммиты:\n{log}")
    except Exception:
        pass

    try:
        file_list = git_list_files()
        if file_list:
            all_files = file_list.split("\n")
            dirs = set()
            root_files = []
            for f in all_files:
                if "/" in f:
                    dirs.add(f.split("/")[0] + "/")
                else:
                    root_files.append(f)
            structure = sorted(dirs) + sorted(root_files)
            parts.append(f"Структура проекта ({len(all_files)} файлов):\n" + "\n".join(structure))
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

    git_context = get_git_context()

    results = pipeline.search(question, top_k=5)
    has_docs = bool(results) and max(c.get("similarity", 0) for c in results) > 0.25

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

    system_prompt = (
        "Ты — ассистент разработчика. Отвечай на вопросы о проекте, "
        "используя предоставленную документацию и информацию о репозитории.\n\n"
        "Отвечай на русском языке, конкретно и по делу.\n\n"
    )

    if doc_context:
        system_prompt += f"Документация проекта:\n\n{doc_context}\n\n"

    system_prompt += f"Информация о репозитории:\n\n{git_context}"

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
        print(f"\n  Ветка: {git_branch()}")
    except Exception as e:
        print(f"  Ошибка: {e}")

    try:
        branches = git_branches()
        print(f"\n  Все ветки:\n{branches}")
    except Exception:
        pass

    try:
        status = git_status()
        if status:
            print(f"\n  Статус:\n{status}")
        else:
            print("\n  Рабочее дерево чистое")
    except Exception:
        pass

    try:
        log = git_log(5)
        print(f"\n  Последние коммиты:\n{log}")
    except Exception:
        pass


def main():
    print("=" * 60)
    print("  Ассистент разработчика")
    print("  RAG + Git")
    print("=" * 60)

    # История команд (стрелки вверх/вниз)
    readline.set_history_length(500)
    if os.path.exists(HISTORY_FILE):
        readline.read_history_file(HISTORY_FILE)

    pipeline = IndexingPipeline(db_path=DB_PATH)

    stats_list = pipeline.get_stats()
    total_chunks = sum(s.get("chunks", 0) for s in stats_list)
    total_files = sum(s.get("files", 0) for s in stats_list)
    if total_chunks == 0:
        print("\nПервый запуск — индексирую документацию...")
        index_docs(pipeline)
    else:
        print(f"\nИндекс: {total_chunks} чанков, {total_files} файлов")

    try:
        print(f"Git: ветка '{git_branch()}'")
    except Exception:
        print("Git: не удалось подключиться")

    print("\nКоманды:")
    print("  /help <вопрос>  — спросить о проекте")
    print("  /review [base]  — AI code review (default: main)")
    print("  /docs [file]    — сгенерировать документацию проекта")
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

        if user_input.lower().startswith("/docs"):
            parts = user_input.split()
            output_file = parts[1] if len(parts) > 1 else None
            from dev_assistant.docs_gen import generate_docs
            print("\n  Сканирую проект и генерирую документацию...")
            try:
                result = generate_docs(pipeline=pipeline)
                width = get_width()
                if output_file:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(result["docs"])
                    print(f"  Документация сохранена в {output_file}")
                else:
                    print()
                    print_wrapped(result["docs"], width)
                scanned = result["files_scanned"]
                t = result["tokens"]
                print(f"\n  Файлов: код={scanned['code']}, конфиги={scanned['config']}, доки={scanned['docs']}")
                print(f"  Токены: {t['input']} вход / {t['output']} выход")
            except Exception as e:
                print(f"  Ошибка: {e}")
            continue

        if user_input.lower().startswith("/review"):
            parts = user_input.split()
            base = parts[1] if len(parts) > 1 else "main"
            from dev_assistant.review import run_review
            print(f"\n  Анализирую изменения (vs {base})...")
            try:
                result = run_review(base=base, pipeline=pipeline)
                width = get_width()
                print()
                print_wrapped(result["review"], width)
                t = result["tokens"]
                print(f"\n  Токены: {t['input']} вход / {t['output']} выход")
            except Exception as e:
                print(f"  Ошибка: {e}")
            continue

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

        handle_help(user_input, pipeline)

    readline.write_history_file(HISTORY_FILE)
    pipeline.close()


if __name__ == "__main__":
    main()
