"""
Агент поддержки пользователей.

Оркестрирует три MCP-сервера:
  - crm (пользователи, тикеты)
  - support_rag (FAQ индекс, семантический поиск, RAG-ответы)
  - youtrack (задачи в YouTrack — создание, поиск, обновление)

Алгоритм:
  1. Определить пользователя/тикет из вопроса
  2. Загрузить контекст из CRM
  3. (Опционально) загрузить связанные задачи из YouTrack
  4. Найти релевантные FAQ через RAG
  5. Сгенерировать ответ с учётом контекста
"""

import os
import sys
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp_client import MCPClient


class SupportAgent:
    """AI-ассистент поддержки пользователей."""

    def __init__(self, youtrack=True):
        self._server_dir = os.path.dirname(os.path.abspath(__file__))
        self._crm_client = None
        self._rag_client = None
        self._yt_client = None
        self._youtrack_enabled = youtrack
        self._indexed = False

    def connect(self):
        """Подключиться к MCP-серверам."""
        crm_cmd = [sys.executable, os.path.join(self._server_dir, "crm_mcp_server.py")]
        rag_cmd = [sys.executable, os.path.join(self._server_dir, "support_rag_mcp_server.py")]

        self._crm_client = MCPClient(server_cmd=crm_cmd)
        self._crm_client.connect()

        self._rag_client = MCPClient(server_cmd=rag_cmd)
        self._rag_client.connect()

        # YouTrack (опционально — требует .env.youtrack)
        if self._youtrack_enabled:
            try:
                yt_cmd = [sys.executable, os.path.join(
                    _PROJECT_ROOT, "youtrack_mcp", "youtrack_mcp_server.py")]
                self._yt_client = MCPClient(server_cmd=yt_cmd)
                self._yt_client.connect()
            except Exception:
                self._yt_client = None

        # Индексируем FAQ при первом подключении
        if not self._indexed:
            result = self._rag_client.call_tool("support_index_faq", {})
            self._indexed = True
            return self._extract_text(result)

        return {"status": "connected"}

    @property
    def youtrack_connected(self):
        return self._yt_client is not None

    def close(self):
        """Закрыть соединения."""
        if self._crm_client:
            self._crm_client.close()
        if self._rag_client:
            self._rag_client.close()
        if self._yt_client:
            self._yt_client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _extract_text(result):
        """Извлечь текст из MCP tool result."""
        content = result.get("content", [])
        if content:
            text = content[0].get("text", "{}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
        return result

    # ── CRM-методы ────────────────────────────────────

    def get_user(self, user_id=None, email=None):
        """Получить данные пользователя."""
        args = {}
        if user_id:
            args["user_id"] = user_id
        if email:
            args["email"] = email
        result = self._crm_client.call_tool("crm_get_user", args)
        return self._extract_text(result)

    def get_ticket(self, ticket_id):
        """Получить тикет."""
        result = self._crm_client.call_tool("crm_get_ticket", {"ticket_id": ticket_id})
        return self._extract_text(result)

    def get_user_context(self, user_id=None, email=None):
        """Полный контекст пользователя с тикетами."""
        args = {}
        if user_id:
            args["user_id"] = user_id
        if email:
            args["email"] = email
        result = self._crm_client.call_tool("crm_get_user_context", args)
        return self._extract_text(result)

    def search_tickets(self, query):
        """Поиск тикетов."""
        result = self._crm_client.call_tool("crm_search_tickets", {"query": query})
        return self._extract_text(result)

    def list_tickets(self, **filters):
        """Список тикетов с фильтрами."""
        result = self._crm_client.call_tool("crm_list_tickets", filters)
        return self._extract_text(result)

    def add_comment(self, ticket_id, note):
        """Добавить комментарий к тикету."""
        result = self._crm_client.call_tool("crm_add_ticket_comment", {
            "ticket_id": ticket_id,
            "note": note,
        })
        return self._extract_text(result)

    def update_ticket_status(self, ticket_id, status):
        """Обновить статус тикета."""
        result = self._crm_client.call_tool("crm_update_ticket_status", {
            "ticket_id": ticket_id,
            "status": status,
        })
        return self._extract_text(result)

    # ── RAG-методы ────────────────────────────────────

    def search_faq(self, query, top_k=5):
        """Семантический поиск по FAQ."""
        result = self._rag_client.call_tool("support_search", {
            "query": query,
            "top_k": top_k,
        })
        return self._extract_text(result)

    def ask_faq(self, question, user_context="", top_k=5):
        """RAG-ответ на вопрос с контекстом."""
        result = self._rag_client.call_tool("support_ask", {
            "question": question,
            "user_context": user_context,
            "top_k": top_k,
        })
        return self._extract_text(result)

    # ── YouTrack-методы ────────────────────────────────

    def _ensure_youtrack(self):
        if not self._yt_client:
            raise RuntimeError("YouTrack не подключён. Проверьте youtrack_mcp/.env.youtrack")

    def yt_get_projects(self, top=10):
        """Список проектов YouTrack."""
        self._ensure_youtrack()
        result = self._yt_client.call_tool("youtrack_get_projects", {"top": top})
        return self._extract_text(result)

    def yt_get_issues(self, query=None, project=None, top=10):
        """Поиск задач YouTrack."""
        self._ensure_youtrack()
        args = {"top": top}
        if query:
            args["query"] = query
        if project:
            args["project"] = project
        result = self._yt_client.call_tool("youtrack_get_issues", args)
        return self._extract_text(result)

    def yt_get_issue(self, issue_id):
        """Получить задачу YouTrack по ID."""
        self._ensure_youtrack()
        result = self._yt_client.call_tool("youtrack_get_issue", {"issue_id": issue_id})
        return self._extract_text(result)

    def yt_create_issue(self, project_id, summary, description=None):
        """Создать задачу в YouTrack."""
        self._ensure_youtrack()
        args = {"project_id": project_id, "summary": summary}
        if description:
            args["description"] = description
        result = self._yt_client.call_tool("youtrack_create_issue", args)
        return self._extract_text(result)

    def yt_update_issue(self, issue_id, summary=None, description=None):
        """Обновить задачу YouTrack."""
        self._ensure_youtrack()
        args = {"issue_id": issue_id}
        if summary:
            args["summary"] = summary
        if description:
            args["description"] = description
        result = self._yt_client.call_tool("youtrack_update_issue", args)
        return self._extract_text(result)

    # ── Главный метод ─────────────────────────────────

    def _format_ticket_context(self, ticket, include_user=True):
        """Форматировать тикет для контекста LLM."""
        parts = []
        parts.append(
            f"Тикет [{ticket['id']}]: {ticket['subject']}"
        )
        parts.append(f"  Описание: {ticket['description']}")
        parts.append(
            f"  Категория: {ticket['category']}, "
            f"Приоритет: {ticket['priority']}, "
            f"Статус: {ticket['status']}"
        )
        if ticket.get("tags"):
            parts.append(f"  Теги: {', '.join(ticket['tags'])}")
        if ticket.get("history"):
            parts.append("  История:")
            for h in ticket["history"][-5:]:
                parts.append(f"    {h['timestamp']} — {h['action']}: {h['note']}")
        # Подгрузка пользователя тикета
        if include_user:
            user = self.get_user(user_id=ticket.get("user_id"))
            if "error" not in user:
                parts.append(
                    f"  Пользователь: {user.get('name', '?')} "
                    f"(email: {user.get('email', '?')}, "
                    f"план: {user.get('plan', '?')}, статус: {user.get('status', '?')})"
                )
        return "\n".join(parts)

    def answer(self, question, user_id=None, email=None, ticket_id=None, youtrack_issue=None):
        """
        Ответить на вопрос пользователя с учётом контекста.

        1. Если указан user_id/email — загружает профиль и тикеты
        2. Если указан ticket_id — загружает конкретный тикет
        3. Если указан youtrack_issue — загружает задачу из YouTrack
        4. Автоматически ищет релевантные тикеты в CRM по вопросу
        5. Ищет ответ в FAQ через RAG
        6. Генерирует ответ с учётом всех данных
        """
        # Собираем контекст пользователя
        context_parts = []

        user_data = None
        if user_id or email:
            ctx = self.get_user_context(user_id=user_id, email=email)
            if "error" not in ctx:
                user_data = ctx.get("user", {})
                context_parts.append(
                    f"Пользователь: {user_data.get('name', '?')} "
                    f"(план: {user_data.get('plan', '?')}, "
                    f"статус: {user_data.get('status', '?')}, "
                    f"2FA: {'да' if user_data.get('two_factor') else 'нет'}, "
                    f"API-ключ: {'активен' if user_data.get('api_key_active') else 'нет'})"
                )
                tickets = ctx.get("tickets", [])
                if tickets:
                    context_parts.append(f"Открытых тикетов: {ctx.get('open_tickets', 0)}")
                    for t in tickets:
                        context_parts.append(
                            f"  - [{t['id']}] {t['subject']} "
                            f"(статус: {t['status']}, приоритет: {t['priority']})"
                        )

        ticket_data = None
        if ticket_id:
            ticket_data = self.get_ticket(ticket_id)
            if "error" not in ticket_data:
                context_parts.append(
                    f"\nТекущий тикет:"
                )
                context_parts.append(self._format_ticket_context(
                    ticket_data, include_user=(not user_data)
                ))
                # Подгружаем пользователя тикета если не загружен
                if not user_data:
                    tid_user = self.get_user(user_id=ticket_data.get("user_id"))
                    if "error" not in tid_user:
                        user_data = tid_user

        # Автоматический поиск релевантных тикетов по вопросу
        # CRM search сам разбивает запрос на слова, нормализует и ранжирует
        related_tickets = []
        try:
            search_result = self.search_tickets(question)
            found = search_result.get("tickets", [])
            for t in found:
                if ticket_data and t.get("id") == ticket_data.get("id"):
                    continue
                related_tickets.append(t)
        except Exception:
            pass

        if related_tickets:
            context_parts.append(f"\nРелевантные тикеты из CRM ({len(related_tickets)}):")
            for t in related_tickets[:5]:
                context_parts.append(self._format_ticket_context(t, include_user=True))
                context_parts.append("")

        # YouTrack задача
        yt_issue_data = None
        if youtrack_issue and self._yt_client:
            try:
                yt_issue_data = self.yt_get_issue(youtrack_issue)
                if isinstance(yt_issue_data, dict) and "error" not in str(yt_issue_data).lower():
                    context_parts.append(
                        f"\nЗадача YouTrack [{yt_issue_data.get('idReadable', youtrack_issue)}]: "
                        f"{yt_issue_data.get('summary', '?')}"
                    )
                    if yt_issue_data.get("description"):
                        context_parts.append(f"Описание: {yt_issue_data['description']}")
                    for cf in yt_issue_data.get("customFields", []):
                        val = cf.get("value")
                        if isinstance(val, dict):
                            val = val.get("name", str(val))
                        if val:
                            context_parts.append(f"  {cf.get('name', '?')}: {val}")
                    comments = yt_issue_data.get("comments", [])
                    if comments:
                        context_parts.append("Комментарии YouTrack:")
                        for c in comments[-5:]:
                            author = c.get("author", {}).get("login", "?")
                            context_parts.append(f"  [{author}]: {c.get('text', '')[:200]}")
            except Exception:
                pass

        user_context = "\n".join(context_parts) if context_parts else ""

        # RAG-ответ
        rag_result = self.ask_faq(question, user_context=user_context)

        return {
            "answer": rag_result.get("answer", "Не удалось получить ответ"),
            "sources": rag_result.get("sources", []),
            "user": user_data,
            "ticket": ticket_data,
            "youtrack_issue": yt_issue_data,
            "related_tickets": len(related_tickets),
            "fragments_used": rag_result.get("fragments_used", 0),
            "tokens": rag_result.get("tokens", {}),
        }
