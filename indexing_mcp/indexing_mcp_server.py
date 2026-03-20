#!/usr/bin/env python3
"""
MCP-сервер индексации документов.

Инструменты:
  - index_directory  — проиндексировать директорию (scan → chunk → embed → store)
  - index_search     — семантический поиск по индексу
  - index_ask        — RAG: вопрос → top-k чанков → LLM ответ с контекстом
  - index_ask_no_rag — ответ LLM без RAG (для сравнения)
  - index_stats      — статистика хранилища
  - index_compare    — сравнить стратегии чанкинга на директории
"""

import sys
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "indexing-mcp"
SERVER_VERSION = "1.0.0"

# ── Инструменты ──────────────────────────────────────

TOOLS = [
    {
        "name": "index_directory",
        "description": (
            "Проиндексировать директорию: сканирование файлов → разбиение на чанки → "
            "генерация эмбеддингов → сохранение в SQLite. "
            "Поддерживает: .pdf, .md, .txt, .py, .js, .ts, .json, .yaml, .html, .rst, .css."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Путь к директории с документами",
                },
                "strategy": {
                    "type": "string",
                    "description": "Стратегия чанкинга: 'fixed_size', 'structural' или 'both' (по умолчанию 'both')",
                    "enum": ["fixed_size", "structural", "both"],
                },
                "embed": {
                    "type": "boolean",
                    "description": "Генерировать эмбеддинги (по умолчанию true). Если false — только чанкинг без API-вызовов.",
                },
                "db_path": {
                    "type": "string",
                    "description": "Путь к файлу БД индекса (по умолчанию index.db в корне проекта)",
                },
                "force": {
                    "type": "boolean",
                    "description": "Переиндексировать файлы даже если они уже в индексе (по умолчанию false)",
                },
            },
            "required": ["directory"],
        },
    },
    {
        "name": "index_search",
        "description": (
            "Семантический поиск по проиндексированным документам. "
            "Возвращает наиболее релевантные чанки с текстом и метаданными."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос (текст на естественном языке)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Количество результатов (по умолчанию 5)",
                },
                "strategy": {
                    "type": "string",
                    "description": "Фильтр по стратегии чанкинга (опционально)",
                    "enum": ["fixed_size", "structural"],
                },
                "db_path": {
                    "type": "string",
                    "description": "Путь к файлу БД индекса (по умолчанию index.db)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "index_ask",
        "description": (
            "RAG-ответ на вопрос по проиндексированным документам. "
            "Автоматически: эмбеддинг запроса → поиск top-k релевантных чанков → "
            "LLM генерирует ответ (JSON: answer + citations + sources_used). "
            "Если релевантность ниже min_similarity — возвращает 'не знаю' (dont_know=true). "
            "Всегда содержит поля: answer, citations (цитаты из чанков), sources (источники), dont_know."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Вопрос пользователя на естественном языке",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Сколько релевантных фрагментов использовать (по умолчанию 5)",
                },
                "strategy": {
                    "type": "string",
                    "description": "Фильтр по стратегии чанкинга (опционально)",
                    "enum": ["fixed_size", "structural"],
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Минимальный порог релевантности (0.0–1.0). Если максимальная similarity найденных чанков ниже — возвращает 'не знаю' (по умолчанию 0.3)",
                },
                "db_path": {
                    "type": "string",
                    "description": "Путь к файлу БД индекса (по умолчанию index.db)",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "index_ask_enhanced",
        "description": (
            "Улучшенный RAG с реранкингом, фильтрацией и структурированным ответом. "
            "Пайплайн: query rewrite → широкий поиск → порог similarity → LLM-реранкинг → ответ (JSON: answer + citations + sources). "
            "Если релевантность ниже min_similarity — возвращает 'не знаю' (dont_know=true). "
            "Даёт более точные ответы за счёт отсечения нерелевантных чанков."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Вопрос пользователя",
                },
                "threshold": {
                    "type": "number",
                    "description": "Порог similarity для отсечения при реранкинге (по умолчанию 0.45)",
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Минимальный порог релевантности для режима 'не знаю' (по умолчанию 0.3)",
                },
                "top_k_initial": {
                    "type": "integer",
                    "description": "Сколько чанков искать на первом этапе (по умолчанию 10)",
                },
                "top_k_final": {
                    "type": "integer",
                    "description": "Сколько чанков оставить после реранкинга (по умолчанию 5)",
                },
                "rewrite": {
                    "type": "boolean",
                    "description": "Переформулировать запрос через LLM перед поиском (по умолчанию false)",
                },
                "strategy": {
                    "type": "string",
                    "description": "Фильтр по стратегии чанкинга",
                    "enum": ["fixed_size", "structural"],
                },
                "db_path": {
                    "type": "string",
                    "description": "Путь к БД индекса",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "index_ask_no_rag",
        "description": (
            "Ответ LLM БЕЗ RAG — только собственные знания модели, без контекста из документов. "
            "Используется для сравнения качества ответа с RAG и без RAG."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Вопрос пользователя",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "index_stats",
        "description": "Статистика индекса: количество чанков, токенов, файлов по каждой стратегии.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "db_path": {
                    "type": "string",
                    "description": "Путь к файлу БД индекса (по умолчанию index.db)",
                },
            },
        },
    },
    {
        "name": "index_compare",
        "description": (
            "Сравнить две стратегии чанкинга (fixed_size vs structural) на документах "
            "из указанной директории. Возвращает метрики и текстовый отчёт."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Путь к директории с документами для сравнения",
                },
            },
            "required": ["directory"],
        },
    },
]


# ── Helpers ──────────────────────────────────────────

def _default_db():
    return os.path.join(_PROJECT_ROOT, "index.db")


def _get_pipeline(db_path=None):
    from dotenv import load_dotenv
    load_dotenv()
    from indexing.pipeline import IndexingPipeline
    return IndexingPipeline(db_path=db_path or _default_db())


# ── Обработчики ──────────────────────────────────────

def _index_directory(args):
    directory = args["directory"]
    if not os.path.isabs(directory):
        directory = os.path.join(_PROJECT_ROOT, directory)

    if not os.path.isdir(directory):
        raise ValueError(f"Директория не найдена: {directory}")

    strategy = args.get("strategy", "both")
    embed = args.get("embed", True)
    db_path = args.get("db_path")
    force = args.get("force", False)

    pipeline = _get_pipeline(db_path)
    try:
        result = pipeline.index_directory(directory, strategy=strategy, embed=embed, force=force)
    finally:
        pipeline.close()

    return result


def _index_search(args):
    query = args["query"]
    top_k = args.get("top_k", 5)
    strategy = args.get("strategy")
    db_path = args.get("db_path")

    pipeline = _get_pipeline(db_path)
    try:
        results = pipeline.search(query, top_k=top_k, strategy=strategy)
    finally:
        pipeline.close()

    return {"query": query, "results": results, "count": len(results)}


def _index_ask(args):
    question = args["question"]
    top_k = args.get("top_k", 5)
    strategy = args.get("strategy")
    db_path = args.get("db_path")
    min_similarity = args.get("min_similarity", 0.3)

    pipeline = _get_pipeline(db_path)
    try:
        result = pipeline.ask(question, top_k=top_k, strategy=strategy, min_similarity=min_similarity)
    finally:
        pipeline.close()

    return result


def _index_ask_enhanced(args):
    question = args["question"]
    threshold = args.get("threshold", 0.45)
    min_similarity = args.get("min_similarity", 0.3)
    top_k_initial = args.get("top_k_initial", 10)
    top_k_final = args.get("top_k_final", 5)
    rewrite = args.get("rewrite", False)
    strategy = args.get("strategy")
    db_path = args.get("db_path")

    pipeline = _get_pipeline(db_path)
    try:
        result = pipeline.ask_enhanced(
            question,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            threshold=threshold,
            rewrite=rewrite,
            strategy=strategy,
            min_similarity=min_similarity,
        )
    finally:
        pipeline.close()

    return result


def _index_ask_no_rag(args):
    question = args["question"]
    pipeline = _get_pipeline()
    try:
        result = pipeline.ask_no_rag(question)
    finally:
        pipeline.close()
    return result


def _index_stats(args):
    db_path = args.get("db_path")
    pipeline = _get_pipeline(db_path)
    try:
        stats = pipeline.get_stats()
    finally:
        pipeline.close()
    return {"stats": stats}


def _index_compare(args):
    directory = args["directory"]
    if not os.path.isabs(directory):
        directory = os.path.join(_PROJECT_ROOT, directory)

    if not os.path.isdir(directory):
        raise ValueError(f"Директория не найдена: {directory}")

    from indexing.pipeline import IndexingPipeline
    from indexing.compare import compare_strategies, generate_report

    pipeline = IndexingPipeline.__new__(IndexingPipeline)
    # Только сканирование, без создания store/embedder
    from indexing.pipeline import SUPPORTED_EXTENSIONS
    files = []
    for root, _dirs, fnames in os.walk(directory):
        parts = root.split(os.sep)
        if any((p.startswith(".") and p != "." and p != "..") or p in ("venv", "node_modules", "__pycache__") for p in parts):
            continue
        for fname in sorted(fnames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text.strip():
                    files.append((filepath, text))
            except (OSError, UnicodeDecodeError):
                continue

    if not files:
        return {"error": "Нет файлов для сравнения в указанной директории"}

    comparison = compare_strategies(files)
    report = generate_report(comparison)

    return {
        "files_scanned": len(files),
        "comparison": comparison,
        "report": report,
    }


# ── Диспетчер ────────────────────────────────────────

_TOOL_HANDLERS = {
    "index_directory": _index_directory,
    "index_search": _index_search,
    "index_ask": _index_ask,
    "index_ask_enhanced": _index_ask_enhanced,
    "index_ask_no_rag": _index_ask_no_rag,
    "index_stats": _index_stats,
    "index_compare": _index_compare,
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
