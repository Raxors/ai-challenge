import os
import sys
import textwrap
import time

from dotenv import load_dotenv
from llm import OpenAIModel
from agent import Agent
from core import LLMError
from strategies import get_strategy_names

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

try:
    import select
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

NOTIFICATION_POLL_INTERVAL = 10  # секунд между проверками

_input_history = []

load_dotenv()

MODEL_NAME = "gpt-4o"


def print_wrapped(text, width):
    for paragraph in text.split("\n"):
        if paragraph:
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def get_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def print_metrics(metrics, width):
    m = metrics
    bar_len = 30
    filled = min(int(bar_len * m["usage_percent"] / 100), bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)

    if m["usage_percent"] > 90:
        status = "!! CRITICAL"
    elif m["usage_percent"] > 70:
        status = "! WARNING"
    else:
        status = "OK"

    si = m.get("strategy_info", {})
    ts = m.get("task_state")

    print("-" * width)
    print(f"  Запрос #{m['request_number']}  |  Стратегия: {si.get('strategy', '?')}")
    print(f"    Input tokens:    {m['request_input_tokens']}")
    print(f"    Output tokens:   {m['request_output_tokens']}")
    print(f"    Стоимость:       ${m['request_cost']:.6f}")
    print()
    print(f"  Отправлено в LLM: {m['sent_tokens']} токенов  |  Полная история: {m['full_history_tokens']} токенов")
    print(f"    [{bar}] {m['usage_percent']:.1f}% {status}")

    if "dropped_messages" in si:
        print(f"    Отброшено сообщений: {si['dropped_messages']}")
    if "facts_count" in si:
        print(f"    Фактов в памяти: {si['facts_count']}  |  Токены на извлечение: {si.get('facts_extraction_tokens', 0)}")
    if "branches" in si and "current_branch" in si:
        print(f"    Ветка: {si.get('current_branch', 'main')}  |  Чекпоинты: {si.get('checkpoints', [])}")
    if "current_task" in si:
        wm_keys = si.get("working_memory_keys", [])
        print(f"    Задача: {si['current_task']}  |  Working keys: {len(wm_keys)}  |  Long-term: {si.get('long_term_count', 0)} фактов")
        print(f"    Токены на классификацию: {si.get('classification_tokens', 0)}")
        if si.get("task_change_suggested"):
            print(f"    ⚡ Обнаружена смена задачи! Используйте 'task <name>' для переключения.")

    inv_count = m.get("invariants_count", 0)
    if inv_count:
        print(f"    Инварианты: {inv_count}")

    mcp_tools = m.get("mcp_tools", 0)
    mcp_iters = m.get("mcp_tool_iterations", 0)
    if mcp_tools:
        print(f"    MCP инструменты: {mcp_tools}  |  Вызовов инструментов: {mcp_iters}")

    if ts:
        print(f"    Задача (FSM): {ts['task_name']}  |  Фаза: {ts['phase']}")
        if ts.get("current_step"):
            print(f"    Шаг: {ts['current_step']}")

    print()
    print(f"  Итого за сессию:")
    print(f"    Input:           {m['session_total_input']} токенов")
    print(f"    Output:          {m['session_total_output']} токенов")
    print(f"    Стоимость:       ${m['session_total_cost']:.6f}")
    print("-" * width)


def print_help():
    print("""
Команды:
  exit              — выход
  reset             — сброс диалога
  stats             — статистика сессии
  strategy <name>   — сменить стратегию (sliding_window / sticky_facts / branching / memory_layers)
  facts             — показать текущие факты (для sticky_facts)
  checkpoint <name> — сохранить чекпоинт (для branching)
  branch <name>     — создать ветку от текущего состояния (для branching)
  branch <name> from <checkpoint> — создать ветку от чекпоинта
  switch <name>     — переключиться на ветку (или 'main')
  branches          — список веток и чекпоинтов

  Команды Memory Layers (для стратегии memory_layers):
  memory            — показать все слои памяти
  memory short      — показать short-term (последние сообщения)
  memory working    — показать рабочую память текущей задачи
  memory long       — показать долгосрочную память
  task <name>       — переключиться на задачу (создаёт новую если не существует)
  tasks             — список всех задач
  forget            — полный сброс памяти (working + long-term)

  Профиль пользователя:
  profile           — показать текущий профиль
  profile show      — то же самое
  profile set <field> <value> — установить поле (например: profile set response_language English)
  profile clear     — очистить все поля профиля

  Task State Machine (управление жизненным циклом задачи):
  state             — показать текущее состояние задачи
  state start <name> — начать новую задачу (фаза: planning)
  state advance     — перейти к следующей фазе
  state phase <name> — перейти к конкретной фазе (planning/execution/validation/done)
  state step <text> [| expected action] — установить текущий шаг
  state pause       — приостановить задачу
  state resume      — возобновить задачу
  state clear       — очистить состояние задачи
  state history     — показать историю переходов

  Инварианты проекта (жёсткие ограничения):
  invariant         — показать все инварианты
  invariant show    — то же самое
  invariant add <категория> <правило> — добавить инвариант
  invariant remove <id> — удалить инвариант по ID
  invariant clear   — удалить все инварианты

  MCP серверы (подключение внешних инструментов):
  mcp               — показать все серверы и статус
  mcp show          — то же самое
  mcp add <name> <cmd...> — добавить stdio-сервер (например: mcp add myserver python server.py)
  mcp add-sse <name> <url> — добавить SSE-сервер (например: mcp add-sse remote http://host:8080/sse)
  mcp remove <name>  — удалить сервер
  mcp connect        — подключиться ко всем серверам
  mcp connect <name> — подключиться к конкретному серверу
  mcp disconnect <name> — отключиться от сервера
  mcp tools          — показать все доступные инструменты

  help              — эта справка
""")


def _can_use_cbreak():
    """Проверить, поддерживает ли терминал cbreak-режим."""
    if not _HAS_TERMIOS or not sys.stdin.isatty():
        return False
    try:
        termios.tcgetattr(sys.stdin.fileno())
        return True
    except (termios.error, OSError):
        return False


def _input_cbreak(prompt, agent, poll_interval):
    """Ввод через cbreak + select: уведомления не ломают строку ввода.

    Поддерживает: UTF-8, Backspace, Enter, Ctrl+C/D/U,
    стрелки вверх/вниз для истории.
    """
    global _input_history

    sys.stdout.write(prompt)
    sys.stdout.flush()

    buf = []
    history_pos = len(_input_history)
    saved_buf = None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        while True:
            if select.select([sys.stdin], [], [], poll_interval)[0]:
                raw = os.read(fd, 256)
                if not raw:
                    raise EOFError

                # Escape-последовательности (стрелки и т.п.)
                if raw[0] == 27:
                    if len(raw) >= 3 and raw[1] == 91:
                        arrow = raw[2]
                        if arrow == 65:  # Up
                            if history_pos > 0:
                                if history_pos == len(_input_history):
                                    saved_buf = "".join(buf)
                                history_pos -= 1
                                buf[:] = list(_input_history[history_pos])
                                sys.stdout.write("\r\033[K" + prompt + "".join(buf))
                                sys.stdout.flush()
                        elif arrow == 66:  # Down
                            if history_pos < len(_input_history):
                                history_pos += 1
                                if history_pos == len(_input_history):
                                    buf[:] = list(saved_buf or "")
                                else:
                                    buf[:] = list(_input_history[history_pos])
                                sys.stdout.write("\r\033[K" + prompt + "".join(buf))
                                sys.stdout.flush()
                    continue

                # Декодируем UTF-8
                text = raw.decode("utf-8", errors="ignore")
                for ch in text:
                    c = ord(ch)
                    if c in (10, 13):  # Enter
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        result = "".join(buf)
                        if result.strip():
                            _input_history.append(result)
                        return result
                    elif c in (127, 8):  # Backspace
                        if buf:
                            buf.pop()
                            sys.stdout.write("\b \b")
                    elif c == 3:  # Ctrl+C
                        raise KeyboardInterrupt
                    elif c == 4:  # Ctrl+D
                        if not buf:
                            raise EOFError
                    elif c == 21:  # Ctrl+U — очистить строку
                        sys.stdout.write("\r\033[K" + prompt)
                        buf.clear()
                    elif c >= 32:  # Печатный символ
                        buf.append(ch)
                        sys.stdout.write(ch)
                sys.stdout.flush()
            else:
                # Таймаут — проверяем уведомления
                try:
                    notes = agent.check_scheduler_notifications()
                except Exception:
                    notes = []
                if notes:
                    current = "".join(buf)
                    sys.stdout.write("\r\033[K")
                    _print_notifications(notes)
                    sys.stdout.write(prompt + current)
                    sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _print_notifications(notes):
    """Отформатировать и вывести уведомления планировщика."""
    for note in notes:
        for line in note.split("\n"):
            sys.stdout.write(f"  [Scheduler] {line}\n")
        sys.stdout.write("\n")
    sys.stdout.flush()


def _input_fallback(prompt, agent, poll_interval):
    """Fallback: однопоточный ввод через fcntl non-blocking + polling уведомлений.

    В cooked-режиме терминал сам буферизует и отображает (echo) символы.
    Мы периодически проверяем, появилась ли готовая строка (после Enter),
    а между проверками опрашиваем уведомления планировщика.
    Текст пользователя не теряется — терминал хранит и отображает его сам.
    """
    # Синхронная проверка перед показом prompt
    try:
        notes = agent.check_scheduler_notifications()
    except Exception:
        notes = []
    if notes:
        _print_notifications(notes)

    # Если fcntl недоступен (Windows) — обычный input()
    if _fcntl is None:
        return input(prompt)

    sys.stdout.write(prompt)
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_flags = _fcntl.fcntl(fd, _fcntl.F_GETFL)
    _fcntl.fcntl(fd, _fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

    line_buf = b""
    last_check = time.time()
    try:
        while True:
            # Попытка прочитать данные (неблокирующая)
            try:
                chunk = os.read(fd, 4096)
                if not chunk:
                    raise EOFError
                line_buf += chunk
                if b"\n" in line_buf:
                    text = line_buf.decode("utf-8", errors="ignore")
                    return text.split("\n", 1)[0]
            except BlockingIOError:
                pass

            # Периодическая проверка уведомлений
            now = time.time()
            if now - last_check >= poll_interval:
                last_check = now
                try:
                    notes = agent.check_scheduler_notifications()
                except Exception:
                    notes = []
                if notes:
                    # Перенос на новую строку, чтобы не стирать набранный текст
                    sys.stdout.write("\n")
                    _print_notifications(notes)
                    sys.stdout.write(prompt)
                    sys.stdout.flush()

            time.sleep(0.1)
    except KeyboardInterrupt:
        raise
    finally:
        _fcntl.fcntl(fd, _fcntl.F_SETFL, old_flags)


def _input_with_notifications(prompt, agent, poll_interval):
    """Ввод с проверкой уведомлений. Выбирает лучший доступный режим."""
    if _can_use_cbreak():
        return _input_cbreak(prompt, agent, poll_interval)
    return _input_fallback(prompt, agent, poll_interval)


def main():
    llm = OpenAIModel(model=MODEL_NAME)

    default_strategy = "sliding_window"
    agent = Agent(
        model=llm,
        model_name=MODEL_NAME,
        max_tokens=1024,
        system_prompt="You are a helpful assistant. Answer concisely and clearly.",
        strategy_name=default_strategy,
        window_size=10,
    )

    # Auto-connect MCP servers
    mcp_servers = agent.mcp_hub.get_server_names()
    if mcp_servers:
        errors = agent.mcp_hub.connect_all()
        connected = agent.mcp_hub.get_connected_names()
        total_tools = len(agent.mcp_hub.list_all_tools())
        if connected:
            print(f"MCP: подключено {len(connected)} серверов, {total_tools} инструментов.")
        if errors:
            for name, err in errors.items():
                print(f"MCP ошибка '{name}': {err}")

    width = get_width()
    msg_count = agent.get_message_count()
    strategy_name = agent.strategy.get_name()
    if msg_count > 0:
        print(f"Агент запущен. История: {msg_count} сообщений. Стратегия: {strategy_name}")
    else:
        print(f"Агент запущен. Новый диалог. Стратегия: {strategy_name}")
    print("Введите 'help' для списка команд.")
    print("=" * width)

    try:
        while True:
            try:
                # Show notifications from strategy
                for note in agent.strategy.get_notifications():
                    print(f"\n  {note}")

                # Build prompt with strategy info
                prompt_info = agent.strategy.get_prompt_info()
                branch_info = f" [{prompt_info}]" if prompt_info else ""
                prompt_str = f"Вы{branch_info}: "
                print()  # отступ перед prompt
                user_input = _input_with_notifications(
                    prompt_str, agent, NOTIFICATION_POLL_INTERVAL,
                ).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nВыход.")
                break

            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()
            args = [a.lower() for a in parts[1:]]
            args_raw = parts[1:]  # preserve original case for values

            if cmd == "exit":
                print("Выход.")
                break

            if cmd == "help":
                print_help()
                continue

            if cmd == "reset":
                agent.reset()
                print("[Диалог сброшен. Счётчики обнулены.]")
                continue

            if cmd == "stats":
                print(f"\n  Стратегия:          {agent.strategy.get_name()}")
                print(f"  Запросов:           {agent.request_number}")
                print(f"  Input токенов:      {agent.session_input_tokens}")
                print(f"  Output токенов:     {agent.session_output_tokens}")
                print(f"  Общая стоимость:    ${agent.session_cost:.6f}")
                si = agent.strategy.get_state_info()
                if "facts" in si and si["facts"]:
                    print(f"  Факты: {len(si['facts'])} ключей")
                if "facts_extraction_tokens" in si:
                    print(f"  Токены на факты:    {si['facts_extraction_tokens']}")
                if "current_task" in si:
                    print(f"  Текущая задача:     {si['current_task']}")
                    print(f"  Working memory:     {len(si.get('working_memory_keys', []))} ключей")
                    print(f"  Long-term memory:   {si.get('long_term_count', 0)} фактов")
                    print(f"  Токены на класс.:   {si.get('classification_tokens', 0)}")
                continue

            if cmd == "strategy":
                if not args:
                    print(f"[Текущая: {agent.strategy.get_name()}]")
                    print(f"[Доступные: {', '.join(get_strategy_names())}]")
                    continue
                try:
                    name = agent.set_strategy(args[0], window_size=10)
                    print(f"[Стратегия изменена на: {name}]")
                except ValueError as e:
                    print(f"[Ошибка: {e}]")
                continue

            if cmd == "facts":
                si = agent.strategy.get_state_info()
                if "facts" in si:
                    if si["facts"]:
                        print("\n  Ключевые факты:")
                        for k, v in si["facts"].items():
                            print(f"    {k}: {v}")
                    else:
                        print("[Фактов пока нет]")
                else:
                    print("[Факты доступны только в стратегии sticky_facts]")
                continue

            if cmd == "profile":
                if not args or args[0] == "show":
                    if agent.profile.is_empty():
                        print("[Профиль пуст. Используйте 'profile set <field> <value>'.]")
                    else:
                        print(f"\n  {agent.profile.to_prompt()}")
                    print(f"\n  Доступные поля: {', '.join(agent.profile.get_settable_fields())}")
                elif args[0] == "clear":
                    agent.profile.clear()
                    agent.profile.save(agent.profile_path)
                    print("[Профиль очищен.]")
                elif args[0] == "set":
                    if len(args_raw) < 3:
                        print("[Использование: profile set <field> <value>]")
                        print(f"  Доступные поля: {', '.join(agent.profile.get_settable_fields())}")
                    else:
                        field_name = args[1]  # lowercase for field name
                        value = " ".join(args_raw[2:])  # preserve case for value
                        if agent.profile.set_field(field_name, value):
                            agent.profile.save(agent.profile_path)
                            print(f"[Профиль обновлён: {field_name} = {value}]")
                        else:
                            print(f"[Неизвестное поле: {field_name}]")
                            print(f"  Доступные поля: {', '.join(agent.profile.get_settable_fields())}")
                else:
                    print("[Неизвестная подкоманда. Используйте: profile, profile show, profile set, profile clear]")
                continue

            if cmd == "state":
                if not args or args[0] == "show":
                    if agent.task_state.is_empty():
                        print("[Нет активной задачи. Используйте 'state start <name>'.]")
                    else:
                        print(f"\n  {agent.task_state.to_prompt()}")
                elif args[0] == "start":
                    if len(args_raw) < 2:
                        print("[Использование: state start <task name>]")
                    else:
                        task_name = " ".join(args_raw[1:])
                        agent.task_state.start(task_name)
                        agent.save_task_state()
                        print(f"[Задача '{task_name}' начата. Фаза: planning]")
                elif args[0] == "advance":
                    try:
                        new_phase = agent.task_state.advance()
                        agent.save_task_state()
                        print(f"[Фаза изменена на: {new_phase}]")
                    except ValueError as e:
                        print(f"[Ошибка: {e}]")
                elif args[0] == "phase":
                    if len(args) < 2:
                        print("[Использование: state phase <name>]")
                        print(f"  Доступные переходы: {agent.task_state.get_valid_transitions()}")
                    else:
                        try:
                            new_phase = agent.task_state.transition(args[1])
                            agent.save_task_state()
                            print(f"[Фаза изменена на: {new_phase}]")
                        except ValueError as e:
                            print(f"[Ошибка: {e}]")
                elif args[0] == "step":
                    if len(args_raw) < 2:
                        print("[Использование: state step <text> [| expected action]]")
                    else:
                        step_text = " ".join(args_raw[1:])
                        if "|" in step_text:
                            parts_step = step_text.split("|", 1)
                            step = parts_step[0].strip()
                            expected = parts_step[1].strip()
                        else:
                            step = step_text
                            expected = ""
                        agent.task_state.set_step(step, expected)
                        agent.save_task_state()
                        msg = f"[Шаг: {step}]"
                        if expected:
                            msg += f" [Ожидаемое действие: {expected}]"
                        print(msg)
                elif args[0] == "pause":
                    try:
                        agent.task_state.pause()
                        agent.save_task_state()
                        print("[Задача приостановлена.]")
                    except ValueError as e:
                        print(f"[Ошибка: {e}]")
                elif args[0] == "resume":
                    try:
                        restored = agent.task_state.resume()
                        agent.save_task_state()
                        print(f"[Задача возобновлена. Фаза: {restored}]")
                    except ValueError as e:
                        print(f"[Ошибка: {e}]")
                elif args[0] == "clear":
                    agent.task_state.clear()
                    agent.save_task_state()
                    print("[Состояние задачи очищено.]")
                elif args[0] == "history":
                    if agent.task_state.is_empty() and not agent.task_state.history:
                        print("[Нет истории переходов.]")
                    else:
                        print(f"\n  История переходов (задача: {agent.task_state.task_name}):")
                        for i, entry in enumerate(agent.task_state.history):
                            step_info = f" — {entry['step']}" if entry.get("step") else ""
                            print(f"    {i+1}. {entry['phase']}{step_info}  ({entry['timestamp']})")
                        if not agent.task_state.history:
                            print("    (пусто)")
                else:
                    print("[Неизвестная подкоманда. Используйте 'help' для списка команд.]")
                continue

            if cmd == "invariant":
                if not args or args[0] == "show":
                    if agent.invariants.is_empty():
                        print("[Инвариантов нет. Используйте 'invariant add <категория> <правило>'.]")
                    else:
                        print("\n  Инварианты проекта:")
                        for inv in agent.invariants.get_all():
                            print(f"    #{inv['id']} [{inv['category']}] {inv['rule']}")
                elif args[0] == "add":
                    if len(args_raw) < 3:
                        print("[Использование: invariant add <категория> <правило>]")
                    else:
                        category = args_raw[1]
                        rule = " ".join(args_raw[2:])
                        inv_id = agent.invariants.add(category, rule)
                        agent.save_invariants()
                        print(f"[Инвариант #{inv_id} добавлен: [{category}] {rule}]")
                elif args[0] == "remove":
                    if len(args) < 2:
                        print("[Использование: invariant remove <id>]")
                    else:
                        try:
                            inv_id = int(args[1])
                            if agent.invariants.remove(inv_id):
                                agent.save_invariants()
                                print(f"[Инвариант #{inv_id} удалён.]")
                            else:
                                print(f"[Инвариант #{inv_id} не найден.]")
                        except ValueError:
                            print("[ID должен быть числом.]")
                elif args[0] == "clear":
                    agent.invariants.clear()
                    agent.save_invariants()
                    print("[Все инварианты удалены.]")
                else:
                    print("[Неизвестная подкоманда. Используйте: invariant, invariant show, invariant add, invariant remove, invariant clear]")
                continue

            if cmd == "mcp":
                if not args or args[0] == "show":
                    configured = agent.mcp_hub.get_server_names()
                    connected = agent.mcp_hub.get_connected_names()
                    if not configured:
                        print("[Нет сконфигурированных MCP-серверов. Используйте 'mcp add <name> <cmd...>'.]")
                    else:
                        print("\n  MCP серверы:")
                        for name in configured:
                            status = "подключён" if name in connected else "отключён"
                            info = agent.mcp_hub.get_server_info(name)
                            if info:
                                print(f"    {name}: {info['transport']} — {status} ({info['tools_count']} инструментов)")
                            else:
                                cfg = agent.mcp_hub.get_server_config(name)
                                transport = cfg.get("transport", "stdio")
                                print(f"    {name}: {transport} — {status}")
                elif args[0] == "add":
                    if len(args_raw) < 3:
                        print("[Использование: mcp add <name> <command...>]")
                        print("  Пример: mcp add myserver python server.py")
                    else:
                        name = args_raw[1]
                        command = args_raw[2:]
                        agent.mcp_hub.add_stdio_server(name, command)
                        print(f"[Сервер '{name}' добавлен (stdio). Используйте 'mcp connect' для подключения.]")
                elif args[0] == "add-sse":
                    if len(args_raw) < 3:
                        print("[Использование: mcp add-sse <name> <url>]")
                        print("  Пример: mcp add-sse remote http://localhost:8080/sse")
                    else:
                        name = args_raw[1]
                        url = args_raw[2]
                        agent.mcp_hub.add_sse_server(name, url)
                        print(f"[Сервер '{name}' добавлен (SSE: {url}). Используйте 'mcp connect' для подключения.]")
                elif args[0] == "remove":
                    if len(args_raw) < 2:
                        print("[Использование: mcp remove <name>]")
                    else:
                        name = args_raw[1]
                        if agent.mcp_hub.remove_server(name):
                            print(f"[Сервер '{name}' удалён.]")
                        else:
                            print(f"[Сервер '{name}' не найден.]")
                elif args[0] == "connect":
                    if len(args_raw) >= 2:
                        name = args_raw[1]
                        try:
                            agent.mcp_hub.connect(name)
                            info = agent.mcp_hub.get_server_info(name)
                            tools_count = info["tools_count"] if info else 0
                            print(f"[Подключён к '{name}'. Инструментов: {tools_count}]")
                        except Exception as e:
                            print(f"[Ошибка подключения к '{name}': {e}]")
                    else:
                        errors = agent.mcp_hub.connect_all()
                        connected = agent.mcp_hub.get_connected_names()
                        if connected:
                            total_tools = len(agent.mcp_hub.list_all_tools())
                            print(f"[Подключено серверов: {len(connected)}. Всего инструментов: {total_tools}]")
                        if errors:
                            for name, err in errors.items():
                                print(f"  Ошибка '{name}': {err}")
                        if not connected and not errors:
                            print("[Нет серверов для подключения. Используйте 'mcp add'.]")
                elif args[0] == "disconnect":
                    if len(args_raw) < 2:
                        print("[Использование: mcp disconnect <name>]")
                    else:
                        name = args_raw[1]
                        agent.mcp_hub.disconnect(name)
                        print(f"[Сервер '{name}' отключён.]")
                elif args[0] == "tools":
                    tools = agent.mcp_hub.list_all_tools()
                    if not tools:
                        print("[Нет доступных инструментов. Подключите серверы: 'mcp connect'.]")
                    else:
                        print(f"\n  MCP инструменты ({len(tools)}):")
                        for t in tools:
                            desc = t["description"][:60] + "..." if len(t["description"]) > 60 else t["description"]
                            print(f"    [{t['server']}] {t['name']} — {desc}")
                else:
                    print("[Неизвестная подкоманда. Используйте 'help' для списка команд.]")
                continue

            # Delegate strategy-specific commands
            if agent.strategy.handle_command(cmd, args, agent):
                continue

            # Regular message
            try:
                result = agent.ask(user_input)
            except LLMError as e:
                print(f"\nОшибка: {e}")
                continue

            width = get_width()
            print()
            print_wrapped(result["text"], width)
            print()
            print_metrics(result["metrics"], width)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
