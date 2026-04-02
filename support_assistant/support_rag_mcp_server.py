#!/usr/bin/env python3
"""
MCP-сервер поддержки с RAG — индексирует FAQ и отвечает на вопросы
с использованием семантического поиска по базе знаний.
"""

import sys
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer
from dev_assistant.indexing.pipeline import IndexingPipeline

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "support-rag-mcp"
SERVER_VERSION = "1.0.0"

_FAQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faq")
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "support_index.db")

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = IndexingPipeline(db_path=_DB_PATH)
    return _pipeline


# ── Инструменты ──────────────────────────────────────

TOOLS = [
    {
        "name": "support_index_faq",
        "description": "Проиндексировать FAQ-документацию для семантического поиска. Вызывается один раз при первом запуске.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Переиндексировать принудительно (по умолчанию false)",
                },
            },
        },
    },
    {
        "name": "support_search",
        "description": "Семантический поиск по базе знаний FAQ. Возвращает релевантные фрагменты документации.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
                "top_k": {"type": "integer", "description": "Количество результатов (по умолчанию 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "support_ask",
        "description": "Ответить на вопрос используя RAG: поиск по FAQ + генерация ответа через LLM. Можно передать контекст пользователя/тикета.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Вопрос пользователя"},
                "user_context": {"type": "string", "description": "Контекст пользователя (план, статус, тикет)"},
                "top_k": {"type": "integer", "description": "Количество фрагментов для контекста (по умолчанию 5)"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "support_stats",
        "description": "Статистика индекса базы знаний.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Обработчики ──────────────────────────────────────

def _support_index_faq(args):
    pipeline = _get_pipeline()
    force = args.get("force", False)

    files = pipeline.scan_directory(_FAQ_DIR, extensions={".md"})

    if not files:
        return {"error": "FAQ-файлы не найдены", "directory": _FAQ_DIR}

    result = pipeline.index_files(files, strategy="structural", force=force)
    return {
        "status": "indexed",
        "files": [os.path.basename(f[0]) for f in files],
        **result,
    }


def _support_search(args):
    pipeline = _get_pipeline()
    query = args["query"]
    top_k = args.get("top_k", 5)

    # Автоиндексируем если база пустая
    stats = pipeline.get_stats()
    if not stats:
        _support_index_faq({"force": False})

    results = pipeline.search(query, top_k=top_k)

    return {
        "query": query,
        "results": [
            {
                "text": r["text"],
                "source": os.path.basename(r["source_file"]),
                "title": r.get("title", ""),
                "similarity": r["similarity"],
            }
            for r in results
        ],
        "count": len(results),
    }


def _support_ask(args):
    pipeline = _get_pipeline()
    question = args["question"]
    user_context = args.get("user_context", "")
    top_k = args.get("top_k", 5)

    # Автоиндексируем если база пустая
    stats = pipeline.get_stats()
    if not stats:
        _support_index_faq({"force": False})

    # Семантический поиск
    results = pipeline.search(question, top_k=top_k)

    if not results:
        return {
            "answer": "К сожалению, я не нашёл релевантной информации в базе знаний по вашему вопросу. Обратитесь в поддержку: support@example.com",
            "sources": [],
            "has_context": False,
        }

    # Фильтруем по минимальной релевантности
    min_similarity = 0.3
    relevant = [r for r in results if r["similarity"] >= min_similarity]
    if not relevant:
        relevant = results[:2]  # берём хотя бы 2 лучших

    # Формируем контекст для LLM
    context_parts = []
    sources = []
    for r in relevant:
        source = os.path.basename(r["source_file"])
        context_parts.append(f"[Источник: {source}]\n{r['text']}")
        if source not in sources:
            sources.append(source)

    context_text = "\n\n---\n\n".join(context_parts)

    # Генерируем ответ через OpenAI
    from openai import OpenAI
    client = OpenAI()

    system_prompt = (
        "Ты — AI-ассистент службы поддержки Example Platform. "
        "Отвечай на русском языке, вежливо и по существу. "
        "Используй информацию из предоставленного контекста: базу знаний FAQ и данные CRM (тикеты, пользователи). "
        "Если в контексте есть данные о тикетах и пользователях — используй их для точного ответа: "
        "кто создал тикет, кто занимается проблемой, какой статус, какая история действий. "
        "Если информации недостаточно — честно скажи об этом и предложи обратиться в поддержку. "
        "Не выдумывай информацию. Упоминай конкретные шаги и ссылки из документации."
    )

    user_prompt_parts = [f"Вопрос пользователя: {question}"]
    if user_context:
        user_prompt_parts.append(f"\nКонтекст пользователя:\n{user_context}")
    user_prompt_parts.append(f"\nБаза знаний:\n{context_text}")
    user_prompt_parts.append(
        "\nОтветь на вопрос пользователя, учитывая его контекст и данные из базы знаний."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_parts)},
        ],
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources,
        "fragments_used": len(relevant),
        "has_context": bool(user_context),
        "tokens": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    }


def _support_stats(args):
    pipeline = _get_pipeline()
    stats = pipeline.get_stats()
    return {"stats": stats, "db_path": _DB_PATH, "faq_dir": _FAQ_DIR}


_TOOL_HANDLERS = {
    "support_index_faq": _support_index_faq,
    "support_search": _support_search,
    "support_ask": _support_ask,
    "support_stats": _support_stats,
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
