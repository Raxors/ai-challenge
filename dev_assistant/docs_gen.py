"""
Генерация документации проекта через LLM.

Сканирует код, группирует по модулям, генерирует описание через LLM.
Поддерживает любые языки программирования.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dev_assistant.git_tools import git_list_files

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt", ".swift",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".rb", ".php", ".lua", ".sh", ".bash",
    ".sql", ".r", ".scala", ".ex", ".exs",
}

CONFIG_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".env", ".env.example",
}

DOC_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".txt",
}

SKIP_DIRS = {
    "venv", "node_modules", "__pycache__", ".git",
    "dist", "build", ".tox", ".mypy_cache",
}

DOCS_SYSTEM_PROMPT = """\
Ты — технический писатель. Сгенерируй документацию для проекта на основе его кода.

Формат ответа — Markdown. Структура:

# Название проекта
Краткое описание (1-2 предложения) на основе README или кода.

## Структура проекта
Дерево директорий с кратким описанием каждой папки/модуля.

## Модули
Для каждого модуля/папки:
### <название модуля>
- **Назначение**: что делает модуль
- **Ключевые файлы**: список с описанием
- **Основные классы/функции**: имя + что делает (1 строка)
- **Зависимости**: от каких других модулей зависит

## API / Интерфейсы
Публичные интерфейсы, эндпоинты, CLI-команды — если есть.

## Конфигурация
Какие конфиги и переменные окружения используются.

Правила:
- Пиши на русском языке
- Будь конкретным — описывай что код ДЕЛАЕТ, а не что он "может делать"
- Не выдумывай то, чего нет в коде
- Группируй логически, не перечисляй файлы по алфавиту
"""


def scan_project_files():
    """Собрать файлы проекта через git ls-files + прямое чтение."""
    files_by_type = {"code": [], "config": [], "docs": []}

    try:
        tracked = git_list_files().split("\n")
    except Exception:
        tracked = []
        for root, dirs, fnames in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in fnames:
                rel = os.path.relpath(os.path.join(root, fname), PROJECT_ROOT)
                tracked.append(rel)

    for rel_path in tracked:
        rel_path = rel_path.strip()
        if not rel_path:
            continue

        parts = rel_path.split("/")
        if any(p in SKIP_DIRS or p.startswith(".") for p in parts):
            continue

        ext = os.path.splitext(rel_path)[1].lower()
        basename = os.path.basename(rel_path)

        if ext in CODE_EXTENSIONS:
            files_by_type["code"].append(rel_path)
        elif ext in CONFIG_EXTENSIONS or basename.startswith(".env"):
            files_by_type["config"].append(rel_path)
        elif ext in DOC_EXTENSIONS:
            files_by_type["docs"].append(rel_path)

    return files_by_type


def read_file_safe(rel_path, max_chars=4000):
    """Прочитать файл, обрезать если слишком большой."""
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (обрезано) ..."
        return content
    except (OSError, UnicodeDecodeError):
        return None


def build_project_snapshot(files_by_type, max_total_chars=60000):
    """Собрать снимок проекта для LLM."""
    parts = []
    total = 0

    # Документация — целиком
    for path in files_by_type["docs"]:
        content = read_file_safe(path, max_chars=3000)
        if content:
            parts.append(f"### {path}\n```\n{content}\n```")
            total += len(content)

    # Конфиги — целиком
    for path in files_by_type["config"]:
        content = read_file_safe(path, max_chars=1000)
        if content:
            parts.append(f"### {path}\n```\n{content}\n```")
            total += len(content)

    # Код — с лимитом на общий объём
    for path in sorted(files_by_type["code"]):
        if total > max_total_chars:
            parts.append(f"\n... и ещё {len(files_by_type['code'])} файлов кода (обрезано по лимиту)")
            break
        content = read_file_safe(path, max_chars=3000)
        if content:
            parts.append(f"### {path}\n```\n{content}\n```")
            total += len(content)

    return "\n\n".join(parts)


def generate_docs(pipeline=None, model="gpt-4o"):
    """Сгенерировать документацию проекта."""
    from openai import OpenAI
    client = OpenAI()

    files_by_type = scan_project_files()
    snapshot = build_project_snapshot(files_by_type)

    # RAG-контекст
    rag_context = ""
    if pipeline:
        try:
            results = pipeline.search("архитектура структура модули проект", top_k=3)
            if results and max(c.get("similarity", 0) for c in results) > 0.2:
                rag_parts = [chunk["text"] for chunk in results]
                rag_context = "\n\nДополнительный контекст из документации:\n" + "\n---\n".join(rag_parts)
        except Exception:
            pass

    stats = (
        f"Файлов кода: {len(files_by_type['code'])}, "
        f"конфигов: {len(files_by_type['config'])}, "
        f"документов: {len(files_by_type['docs'])}"
    )

    user_prompt = f"Сгенерируй документацию для проекта.\n\n{stats}\n\n{snapshot}{rag_context}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DOCS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
    )

    return {
        "docs": resp.choices[0].message.content,
        "files_scanned": {k: len(v) for k, v in files_by_type.items()},
        "tokens": {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        },
    }


def main():
    """CLI: python -m dev_assistant.docs_gen [--output FILE]"""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate project documentation")
    parser.add_argument("--output", default=None, help="Файл для записи (default: stdout)")
    parser.add_argument("--model", default="gpt-4o", help="Модель (default: gpt-4o)")
    parser.add_argument("--no-rag", action="store_true", help="Без RAG")
    args = parser.parse_args()

    pipeline = None
    if not args.no_rag:
        try:
            from dev_assistant.indexing.pipeline import IndexingPipeline
            db_path = os.path.join(PROJECT_ROOT, "dev_assistant_index.db")
            pipeline = IndexingPipeline(db_path=db_path)
        except Exception:
            pass

    print("Сканирую проект...", file=sys.stderr)
    result = generate_docs(pipeline=pipeline, model=args.model)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["docs"])
        print(f"Документация сохранена в {args.output}", file=sys.stderr)
    else:
        print(result["docs"])

    scanned = result["files_scanned"]
    t = result["tokens"]
    print(f"Файлов: код={scanned['code']}, конфиги={scanned['config']}, "
          f"доки={scanned['docs']}", file=sys.stderr)
    print(f"Токены: {t['input']} вход / {t['output']} выход", file=sys.stderr)

    if pipeline:
        pipeline.close()


if __name__ == "__main__":
    main()
