#!/usr/bin/env python3
"""
CLI для AI-ассистента поддержки пользователей.

Команды:
  /user <id|email>       — показать данные пользователя
  /ticket <TK-XXX>       — показать тикет
  /tickets [фильтры]     — список тикетов
  /search <запрос>        — поиск по тикетам
  /faq <запрос>           — поиск по FAQ
  /context <id|email>     — полный контекст пользователя
  /set user <id|email>    — установить текущего пользователя
  /set ticket <TK-XXX>    — установить текущий тикет
  /set yt <PROJ-123>      — установить текущую задачу YouTrack
  /yt projects            — список проектов YouTrack
  /yt issues [запрос]     — поиск задач YouTrack
  /yt issue <PROJ-123>    — показать задачу YouTrack
  /yt create <project_id> <summary>  — создать задачу
  /clear                  — сбросить текущий контекст
  /help                   — справка
  /quit                   — выход

Без команды — задать вопрос ассистенту (с учётом текущего контекста).
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

from support_assistant.support_agent import SupportAgent

W = 70


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


def print_user(user):
    if "error" in user:
        print(f"  Ошибка: {user['error']}")
        return
    print(f"  ID:        {user['id']}")
    print(f"  Имя:       {user['name']}")
    print(f"  Email:     {user['email']}")
    print(f"  План:      {user['plan']}")
    print(f"  Статус:    {user['status']}")
    print(f"  2FA:       {'Да' if user.get('two_factor') else 'Нет'}")
    print(f"  API-ключ:  {'Активен' if user.get('api_key_active') else 'Нет'}")
    print(f"  Регистр.:  {user.get('registered', '?')}")
    print(f"  Посл.вход: {user.get('last_login', '?')}")


def print_ticket(ticket):
    if "error" in ticket:
        print(f"  Ошибка: {ticket['error']}")
        return
    print(f"  ID:        {ticket['id']}")
    print(f"  Тема:      {ticket['subject']}")
    print(f"  Статус:    {ticket['status']}")
    print(f"  Приоритет: {ticket['priority']}")
    print(f"  Категория: {ticket['category']}")
    print(f"  Создан:    {ticket['created']}")
    print(f"  Обновлён:  {ticket['updated']}")
    print(f"  Описание:")
    wrap_print(ticket.get("description", ""), indent=4)
    if ticket.get("history"):
        print(f"  История:")
        for h in ticket["history"]:
            print(f"    {h['timestamp']} | {h['action']}: {h['note']}")


def print_yt_issue(issue):
    if isinstance(issue, dict) and "error" in str(issue).lower():
        print(f"  Ошибка: {issue}")
        return
    if not isinstance(issue, dict):
        print(f"  {issue}")
        return
    print(f"  ID:        {issue.get('idReadable', issue.get('id', '?'))}")
    print(f"  Тема:      {issue.get('summary', '?')}")
    if issue.get("description"):
        print(f"  Описание:")
        wrap_print(issue.get("description", ""), indent=4)
    if issue.get("reporter"):
        print(f"  Автор:     {issue['reporter'].get('login', '?')}")
    for cf in issue.get("customFields", []):
        val = cf.get("value")
        if isinstance(val, dict):
            val = val.get("name", str(val))
        if val:
            print(f"  {cf.get('name', '?')}: {val}")
    comments = issue.get("comments", [])
    if comments:
        print(f"  Комментарии ({len(comments)}):")
        for c in comments[-5:]:
            author = c.get("author", {}).get("login", "?")
            print(f"    [{author}]: {c.get('text', '')[:150]}")


def print_help(yt_connected=False):
    yt_status = "подключён" if yt_connected else "не подключён"
    print(f"""
  Команды:
    /user <id|email>       — данные пользователя
    /ticket <TK-XXX>       — данные тикета
    /tickets               — список всех тикетов
    /search <запрос>       — поиск по тикетам
    /faq <запрос>          — поиск по FAQ (без LLM)
    /context <id|email>    — полный контекст пользователя

    /set user <id|email>   — установить текущего пользователя
    /set ticket <TK-XXX>   — установить текущий тикет
    /set yt <PROJ-123>     — установить задачу YouTrack
    /clear                 — сбросить контекст

  YouTrack ({yt_status}):
    /yt projects           — список проектов
    /yt issues [запрос]    — поиск задач
    /yt issue <PROJ-123>   — показать задачу
    /yt create <proj> <тема> — создать задачу

    /help                  — эта справка
    /quit                  — выход

  Просто введите вопрос — ассистент ответит с учётом
  текущего пользователя/тикета/задачи YouTrack.
""")


def main():
    header("AI-АССИСТЕНТ ПОДДЕРЖКИ ПОЛЬЗОВАТЕЛЕЙ")
    print("  Подключение к серверам...")

    agent = SupportAgent(youtrack=True)
    try:
        index_result = agent.connect()
        print(f"  CRM + RAG подключены. FAQ проиндексирован.")
        if isinstance(index_result, dict):
            files = index_result.get("files", [])
            chunks = index_result.get("total_chunks", 0)
            if files:
                print(f"  Файлы: {', '.join(files)} ({chunks} чанков)")
        if agent.youtrack_connected:
            print(f"  YouTrack: подключён")
        else:
            print(f"  YouTrack: не подключён (нет .env.youtrack — работает без него)")
    except Exception as e:
        print(f"  Ошибка подключения: {e}")
        return

    # Текущий контекст
    current_user_id = None
    current_email = None
    current_ticket_id = None
    current_yt_issue = None

    print_help(yt_connected=agent.youtrack_connected)
    print(f"{'─' * W}")

    try:
        while True:
            try:
                prompt_parts = []
                if current_user_id or current_email:
                    prompt_parts.append(f"user:{current_user_id or current_email}")
                if current_ticket_id:
                    prompt_parts.append(f"ticket:{current_ticket_id}")
                if current_yt_issue:
                    prompt_parts.append(f"yt:{current_yt_issue}")
                ctx_str = f" [{', '.join(prompt_parts)}]" if prompt_parts else ""
                user_input = input(f"\n  Вы{ctx_str}> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                break

            if user_input.lower() == "/help":
                print_help()
                continue

            if user_input.lower() == "/clear":
                current_user_id = None
                current_email = None
                current_ticket_id = None
                current_yt_issue = None
                print("  Контекст сброшен.")
                continue

            # /set user <id|email>
            if user_input.lower().startswith("/set user "):
                val = user_input[10:].strip()
                try:
                    current_user_id = int(val)
                    current_email = None
                    user = agent.get_user(user_id=current_user_id)
                    if "error" in user:
                        print(f"  Пользователь не найден: {val}")
                        current_user_id = None
                    else:
                        print(f"  Текущий пользователь: {user['name']} ({user['email']})")
                except ValueError:
                    current_email = val
                    current_user_id = None
                    user = agent.get_user(email=current_email)
                    if "error" in user:
                        print(f"  Пользователь не найден: {val}")
                        current_email = None
                    else:
                        current_user_id = user["id"]
                        print(f"  Текущий пользователь: {user['name']} ({user['email']})")
                continue

            # /set ticket <TK-XXX>
            if user_input.lower().startswith("/set ticket "):
                current_ticket_id = user_input[12:].strip()
                ticket = agent.get_ticket(current_ticket_id)
                if "error" in ticket:
                    print(f"  Тикет не найден: {current_ticket_id}")
                    current_ticket_id = None
                else:
                    print(f"  Текущий тикет: [{ticket['id']}] {ticket['subject']}")
                    # Автоматически подгружаем пользователя тикета
                    if not current_user_id:
                        current_user_id = ticket.get("user_id")
                        user = agent.get_user(user_id=current_user_id)
                        if "error" not in user:
                            print(f"  Пользователь тикета: {user['name']}")
                continue

            # /user <id|email>
            if user_input.lower().startswith("/user "):
                val = user_input[6:].strip()
                section(f"Пользователь: {val}")
                try:
                    user = agent.get_user(user_id=int(val))
                except ValueError:
                    user = agent.get_user(email=val)
                print_user(user)
                continue

            # /ticket <TK-XXX>
            if user_input.lower().startswith("/ticket "):
                tid = user_input[8:].strip()
                section(f"Тикет: {tid}")
                ticket = agent.get_ticket(tid)
                print_ticket(ticket)
                continue

            # /tickets
            if user_input.lower().startswith("/tickets"):
                section("Все тикеты")
                result = agent.list_tickets()
                for t in result.get("tickets", []):
                    status_icon = {"open": "●", "in_progress": "◐", "closed": "○"}.get(t["status"], "?")
                    prio_icon = {"critical": "‼", "high": "!", "medium": "~", "low": "·"}.get(t["priority"], " ")
                    print(f"  {status_icon} [{t['id']}] {prio_icon} {t['subject']}")
                    print(f"      статус: {t['status']}, приоритет: {t['priority']}, категория: {t['category']}")
                print(f"\n  Всего: {result.get('count', 0)}")
                continue

            # /search <query>
            if user_input.lower().startswith("/search "):
                query = user_input[8:].strip()
                section(f"Поиск тикетов: {query}")
                result = agent.search_tickets(query)
                for t in result.get("tickets", []):
                    print(f"  [{t['id']}] {t['subject']} (статус: {t['status']})")
                print(f"\n  Найдено: {result.get('count', 0)}")
                continue

            # /faq <query>
            if user_input.lower().startswith("/faq "):
                query = user_input[5:].strip()
                section(f"Поиск FAQ: {query}")
                result = agent.search_faq(query)
                for r in result.get("results", []):
                    sim = r.get("similarity", 0)
                    src = r.get("source", "?")
                    print(f"\n  [{src}] (релевантность: {sim:.2f})")
                    wrap_print(r.get("text", "")[:300], indent=4)
                continue

            # /context <id|email>
            if user_input.lower().startswith("/context "):
                val = user_input[9:].strip()
                section(f"Контекст пользователя: {val}")
                try:
                    ctx = agent.get_user_context(user_id=int(val))
                except ValueError:
                    ctx = agent.get_user_context(email=val)

                if "error" in ctx:
                    print(f"  Ошибка: {ctx['error']}")
                else:
                    print_user(ctx.get("user", {}))
                    print(f"\n  Тикетов: {ctx.get('tickets_count', 0)} (открытых: {ctx.get('open_tickets', 0)})")
                    for t in ctx.get("tickets", []):
                        print(f"    [{t['id']}] {t['subject']} — {t['status']}")
                continue

            # /set yt <PROJ-123>
            if user_input.lower().startswith("/set yt "):
                current_yt_issue = user_input[8:].strip()
                if agent.youtrack_connected:
                    try:
                        issue = agent.yt_get_issue(current_yt_issue)
                        print(f"  YouTrack задача: [{issue.get('idReadable', current_yt_issue)}] {issue.get('summary', '?')}")
                    except Exception as e:
                        print(f"  Ошибка загрузки задачи: {e}")
                        current_yt_issue = None
                else:
                    print(f"  YouTrack не подключён. Задача сохранена как контекст: {current_yt_issue}")
                continue

            # /yt projects
            if user_input.lower() == "/yt projects":
                if not agent.youtrack_connected:
                    print("  YouTrack не подключён.")
                    continue
                section("Проекты YouTrack")
                try:
                    projects = agent.yt_get_projects()
                    if isinstance(projects, list):
                        for p in projects:
                            print(f"  [{p.get('shortName', '?')}] {p.get('name', '?')}")
                            if p.get("description"):
                                wrap_print(p["description"][:100], indent=4)
                        print(f"\n  Всего: {len(projects)}")
                    else:
                        print(f"  {projects}")
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # /yt issues [запрос]
            if user_input.lower().startswith("/yt issues"):
                if not agent.youtrack_connected:
                    print("  YouTrack не подключён.")
                    continue
                query = user_input[10:].strip() or None
                section(f"Задачи YouTrack{': ' + query if query else ''}")
                try:
                    issues = agent.yt_get_issues(query=query)
                    if isinstance(issues, list):
                        for iss in issues:
                            readable = iss.get("idReadable", iss.get("id", "?"))
                            print(f"  [{readable}] {iss.get('summary', '?')}")
                        print(f"\n  Всего: {len(issues)}")
                    else:
                        print(f"  {issues}")
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # /yt issue <PROJ-123>
            if user_input.lower().startswith("/yt issue "):
                if not agent.youtrack_connected:
                    print("  YouTrack не подключён.")
                    continue
                issue_id = user_input[10:].strip()
                section(f"Задача YouTrack: {issue_id}")
                try:
                    issue = agent.yt_get_issue(issue_id)
                    print_yt_issue(issue)
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # /yt create <project_id> <summary>
            if user_input.lower().startswith("/yt create "):
                if not agent.youtrack_connected:
                    print("  YouTrack не подключён.")
                    continue
                parts = user_input[11:].strip().split(" ", 1)
                if len(parts) < 2:
                    print("  Формат: /yt create <project_id> <тема>")
                    continue
                proj_id, summary = parts
                section(f"Создание задачи в {proj_id}")
                try:
                    result = agent.yt_create_issue(proj_id, summary)
                    print(f"  Создана: [{result.get('idReadable', '?')}] {result.get('summary', summary)}")
                except Exception as e:
                    print(f"  Ошибка: {e}")
                continue

            # Вопрос ассистенту
            print(f"\n  Думаю...")
            try:
                result = agent.answer(
                    question=user_input,
                    user_id=current_user_id,
                    email=current_email,
                    ticket_id=current_ticket_id,
                    youtrack_issue=current_yt_issue,
                )

                section("Ответ ассистента")
                wrap_print(result.get("answer", "Нет ответа"))

                sources = result.get("sources", [])
                related = result.get("related_tickets", 0)
                if sources or related:
                    info_parts = []
                    if sources:
                        info_parts.append(f"FAQ: {', '.join(sources)}")
                    if related:
                        info_parts.append(f"тикетов CRM: {related}")
                    print(f"\n  Источники: {' | '.join(info_parts)}")

                tokens = result.get("tokens", {})
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
