#!/usr/bin/env python3
"""
MCP-сервер CRM — доступ к данным пользователей и тикетов.

Предоставляет инструменты для поиска пользователей, тикетов,
обновления статусов тикетов и добавления комментариев.
"""

import sys
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.jsonrpc import JSONRPCServer

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "crm-mcp"
SERVER_VERSION = "1.0.0"

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_USERS_PATH = os.path.join(_DATA_DIR, "users.json")
_TICKETS_PATH = os.path.join(_DATA_DIR, "tickets.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ── Инструменты ──────────────────────────────────────

TOOLS = [
    {
        "name": "crm_get_user",
        "description": "Получить данные пользователя по ID или email.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "description": "ID пользователя"},
                "email": {"type": "string", "description": "Email пользователя"},
            },
        },
    },
    {
        "name": "crm_list_users",
        "description": "Список всех пользователей. Можно фильтровать по плану или статусу.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "Фильтр по плану (free/pro/enterprise)"},
                "status": {"type": "string", "description": "Фильтр по статусу (active/suspended)"},
            },
        },
    },
    {
        "name": "crm_get_ticket",
        "description": "Получить тикет по ID (например TK-001).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "ID тикета (например TK-001)"},
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "crm_list_tickets",
        "description": "Список тикетов. Можно фильтровать по user_id, статусу, категории, приоритету.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "description": "Фильтр по ID пользователя"},
                "status": {"type": "string", "description": "Фильтр по статусу (open/in_progress/closed)"},
                "category": {"type": "string", "description": "Фильтр по категории (auth/billing/api/general)"},
                "priority": {"type": "string", "description": "Фильтр по приоритету (low/medium/high/critical)"},
            },
        },
    },
    {
        "name": "crm_search_tickets",
        "description": "Поиск тикетов по ключевым словам в теме и описании.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "crm_add_ticket_comment",
        "description": "Добавить комментарий к тикету.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "ID тикета"},
                "note": {"type": "string", "description": "Текст комментария"},
            },
            "required": ["ticket_id", "note"],
        },
    },
    {
        "name": "crm_update_ticket_status",
        "description": "Обновить статус тикета.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "ID тикета"},
                "status": {"type": "string", "description": "Новый статус (open/in_progress/closed)"},
            },
            "required": ["ticket_id", "status"],
        },
    },
    {
        "name": "crm_get_user_context",
        "description": "Полный контекст пользователя: профиль + все его тикеты. Полезно для ответа на вопрос с учётом истории.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "description": "ID пользователя"},
                "email": {"type": "string", "description": "Email пользователя"},
            },
        },
    },
]


# ── Обработчики ──────────────────────────────────────

def _crm_get_user(args):
    users = _load_json(_USERS_PATH)
    user_id = args.get("user_id")
    email = args.get("email")

    for u in users:
        if (user_id and u["id"] == user_id) or (email and u["email"] == email):
            return u

    return {"error": "Пользователь не найден"}


def _crm_list_users(args):
    users = _load_json(_USERS_PATH)
    plan = args.get("plan")
    status = args.get("status")

    if plan:
        users = [u for u in users if u["plan"] == plan]
    if status:
        users = [u for u in users if u["status"] == status]

    return {"users": users, "count": len(users)}


def _crm_get_ticket(args):
    tickets = _load_json(_TICKETS_PATH)
    ticket_id = args["ticket_id"]

    for t in tickets:
        if t["id"] == ticket_id:
            return t

    return {"error": f"Тикет {ticket_id} не найден"}


def _crm_list_tickets(args):
    tickets = _load_json(_TICKETS_PATH)
    user_id = args.get("user_id")
    status = args.get("status")
    category = args.get("category")
    priority = args.get("priority")

    if user_id:
        tickets = [t for t in tickets if t["user_id"] == user_id]
    if status:
        tickets = [t for t in tickets if t["status"] == status]
    if category:
        tickets = [t for t in tickets if t["category"] == category]
    if priority:
        tickets = [t for t in tickets if t["priority"] == priority]

    summary = [
        {
            "id": t["id"],
            "subject": t["subject"],
            "status": t["status"],
            "priority": t["priority"],
            "category": t["category"],
            "user_id": t["user_id"],
        }
        for t in tickets
    ]
    return {"tickets": summary, "count": len(summary)}


def _normalize_word(word):
    """Простая нормализация русского слова — обрезка типичных окончаний."""
    import re
    word = re.sub(r'[^\w]', '', word.lower())
    if len(word) <= 3:
        return word
    # Отрезаем типичные русские окончания (грубая стемматизация)
    for suffix in ("ация", "яции", "ации", "ений", "ения", "ться", "ного",
                   "ному", "ской", "ским", "ских", "ного", "ному",
                   "ость", "ости", "ные", "ных", "ной", "ное",
                   "ами", "ями", "ием", "ией", "ого", "ому",
                   "ей", "ий", "ой", "ые", "ых", "ом", "ем",
                   "ов", "ев", "ам", "ям", "ах", "ях",
                   "ую", "юю", "ая", "яя", "ое", "ее",
                   "ит", "ет", "ут", "ют", "ат", "ят",
                   "ть", "ся",
                   "а", "о", "у", "е", "и", "ы", "я", "ю"):
        if len(word) - len(suffix) >= 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def _crm_search_tickets(args):
    tickets = _load_json(_TICKETS_PATH)
    raw_query = args["query"].lower()

    # Стоп-слова
    stop_words = {
        "кто", "что", "как", "где", "когда", "почему", "зачем",
        "для", "при", "без", "над", "под", "это", "этот", "эта",
        "все", "вся", "всё", "мой", "мне", "меня", "они", "она", "его",
        "был", "была", "были", "быть", "будет", "есть", "нет",
        "уже", "ещё", "еще", "тоже", "также", "очень", "только",
        "или", "либо", "если", "чтобы", "потому",
        "занят", "занята", "заняты", "делает", "делают",
        "работает", "работают", "занимается", "занимаются",
        "задачей", "задачу", "задача", "задачи",
        "проблема", "проблему", "проблемой", "проблемы",
        "починка", "починки", "починкой", "починку",
    }

    import re
    words = re.findall(r'[а-яёa-z0-9]+', raw_query)
    # Значимые слова: >= 3 символов и не стоп-слова
    query_words = [w for w in words if len(w) >= 3 and w not in stop_words]
    query_stems = [_normalize_word(w) for w in query_words]

    # Синонимы / связанные слова для расширения поиска
    _synonyms = {
        "авториз": ["аутентифик", "auth", "login", "вход", "логин", "пароль", "oauth", "2fa"],
        "аутентифик": ["авториз", "auth", "login", "вход", "логин", "пароль", "oauth", "2fa"],
        "оплат": ["биллинг", "billing", "подписк", "тариф", "план"],
        "биллинг": ["оплат", "billing", "подписк", "тариф"],
        "api": ["запрос", "endpoint", "rate", "лимит", "429"],
        "лимит": ["api", "rate", "429", "запрос"],
        "вебхук": ["webhook", "интеграц", "уведомлен"],
        "блокир": ["suspend", "заблокир", "блокировк"],
    }

    # Расширяем стемы синонимами
    expanded_stems = set(query_stems)
    for stem in query_stems:
        for key, syns in _synonyms.items():
            if stem.startswith(key) or key.startswith(stem):
                expanded_stems.update(syns)

    scored = []
    for t in tickets:
        text = f"{t['subject']} {t['description']} {' '.join(t.get('tags', []))} {t.get('category', '')}".lower()
        text_normalized = " ".join(_normalize_word(w) for w in re.findall(r'[а-яёa-z0-9]+', text))

        score = 0

        # Полное вхождение запроса
        if raw_query in text:
            score += 10

        # Совпадение по стемам (прямые)
        for stem in query_stems:
            if stem and stem in text_normalized:
                score += 3
            elif stem and stem in text:
                score += 2

        # Совпадение по расширенным стемам (синонимы)
        for stem in expanded_stems - set(query_stems):
            if stem and stem in text_normalized:
                score += 1
            elif stem and stem in text:
                score += 1

        # Совпадение по оригинальным словам
        for word in query_words:
            if word in text:
                score += 1

        if score > 0:
            scored.append((t, score))

    # Сортировка по релевантности
    scored.sort(key=lambda x: x[1], reverse=True)
    results = [t for t, _ in scored]

    return {"tickets": results, "count": len(results), "query": raw_query}


def _crm_add_ticket_comment(args):
    tickets = _load_json(_TICKETS_PATH)
    ticket_id = args["ticket_id"]
    note = args["note"]

    for t in tickets:
        if t["id"] == ticket_id:
            import datetime
            now = datetime.datetime.now().isoformat(timespec="seconds")
            t["history"].append({
                "timestamp": now,
                "action": "comment",
                "note": note,
            })
            t["updated"] = now
            _save_json(_TICKETS_PATH, tickets)
            return {"success": True, "ticket_id": ticket_id, "comment": note}

    return {"error": f"Тикет {ticket_id} не найден"}


def _crm_update_ticket_status(args):
    tickets = _load_json(_TICKETS_PATH)
    ticket_id = args["ticket_id"]
    new_status = args["status"]

    if new_status not in ("open", "in_progress", "closed"):
        return {"error": f"Недопустимый статус: {new_status}"}

    for t in tickets:
        if t["id"] == ticket_id:
            old_status = t["status"]
            t["status"] = new_status
            import datetime
            now = datetime.datetime.now().isoformat(timespec="seconds")
            t["history"].append({
                "timestamp": now,
                "action": "status_change",
                "note": f"Статус изменён: {old_status} → {new_status}",
            })
            t["updated"] = now
            _save_json(_TICKETS_PATH, tickets)
            return {"success": True, "ticket_id": ticket_id, "old_status": old_status, "new_status": new_status}

    return {"error": f"Тикет {ticket_id} не найден"}


def _crm_get_user_context(args):
    users = _load_json(_USERS_PATH)
    tickets = _load_json(_TICKETS_PATH)

    user_id = args.get("user_id")
    email = args.get("email")

    user = None
    for u in users:
        if (user_id and u["id"] == user_id) or (email and u["email"] == email):
            user = u
            break

    if not user:
        return {"error": "Пользователь не найден"}

    user_tickets = [t for t in tickets if t["user_id"] == user["id"]]

    return {
        "user": user,
        "tickets": user_tickets,
        "tickets_count": len(user_tickets),
        "open_tickets": len([t for t in user_tickets if t["status"] == "open"]),
    }


_TOOL_HANDLERS = {
    "crm_get_user": _crm_get_user,
    "crm_list_users": _crm_list_users,
    "crm_get_ticket": _crm_get_ticket,
    "crm_list_tickets": _crm_list_tickets,
    "crm_search_tickets": _crm_search_tickets,
    "crm_add_ticket_comment": _crm_add_ticket_comment,
    "crm_update_ticket_status": _crm_update_ticket_status,
    "crm_get_user_context": _crm_get_user_context,
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
