import os
import textwrap

from dotenv import load_dotenv
from openai_model import OpenAIModel
from agent import Agent
from context_strategies import BranchingStrategy
from llm_interface import LLMError

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

    print("-" * width)
    print(f"  Запрос #{m['request_number']}  |  Стратегия: {si.get('strategy', '?')}")
    print(f"    Input tokens:    {m['request_input_tokens']}")
    print(f"    Output tokens:   {m['request_output_tokens']}")
    print(f"    Стоимость:       ${m['request_cost']:.6f}")
    print()
    print(f"  Отправлено в LLM: {m['sent_tokens']} токенов  |  Полная история: {m['full_history_tokens']} токенов")
    print(f"    [{bar}] {m['usage_percent']:.1f}% {status}")

    # Дополнительная инфо от стратегии
    if "dropped_messages" in si:
        print(f"    Отброшено сообщений: {si['dropped_messages']}")
    if "facts_count" in si:
        print(f"    Фактов в памяти: {si['facts_count']}  |  Токены на извлечение: {si.get('facts_extraction_tokens', 0)}")
    if "branches" in si:
        print(f"    Ветка: {si.get('current_branch', 'main')}  |  Чекпоинты: {si.get('checkpoints', [])}")

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
  strategy <name>   — сменить стратегию (sliding_window / sticky_facts / branching)
  facts             — показать текущие факты (для sticky_facts)
  checkpoint <name> — сохранить чекпоинт (для branching)
  branch <name>     — создать ветку от текущего состояния (для branching)
  branch <name> from <checkpoint> — создать ветку от чекпоинта
  switch <name>     — переключиться на ветку (или 'main')
  branches          — список веток и чекпоинтов
  help              — эта справка
""")


def handle_branching_commands(agent, cmd, args):
    """Обрабатывает команды ветвления. Возвращает True если команда обработана."""
    strategy = agent.strategy
    if not isinstance(strategy, BranchingStrategy):
        print("[Команда доступна только в стратегии branching]")
        return True

    if cmd == "checkpoint":
        if not args:
            print("[Укажите имя чекпоинта: checkpoint <name>]")
            return True
        strategy.create_checkpoint(args[0], agent.history)
        print(f"[Чекпоинт '{args[0]}' сохранён ({len(agent.history)} сообщений)]")
        return True

    elif cmd == "branch":
        if not args:
            print("[Укажите имя ветки: branch <name> [from <checkpoint>]]")
            return True
        branch_name = args[0]
        if len(args) >= 3 and args[1] == "from":
            cp_name = args[2]
            if cp_name not in strategy.checkpoints:
                print(f"[Чекпоинт '{cp_name}' не найден]")
                return True
            strategy.create_branch(branch_name, checkpoint_name=cp_name)
            print(f"[Ветка '{branch_name}' создана от чекпоинта '{cp_name}']")
        else:
            strategy.create_branch(branch_name, history=agent.history)
            print(f"[Ветка '{branch_name}' создана от текущего состояния]")
        return True

    elif cmd == "switch":
        if not args:
            print("[Укажите имя ветки: switch <name>]")
            return True
        if strategy.switch_branch(args[0]):
            print(f"[Переключено на ветку '{args[0]}']")
        else:
            print(f"[Ветка '{args[0]}' не найдена]")
        return True

    elif cmd == "branches":
        print("\n  Чекпоинты:")
        for name, info in strategy.list_checkpoints().items():
            print(f"    {name}: {info}")
        print("\n  Ветки:")
        for name, info in strategy.list_branches().items():
            marker = " <--" if (name == (strategy.current_branch or "main")) else ""
            print(f"    {name}: {info}{marker}")
        print()
        return True

    return False


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
                branch_info = ""
                if isinstance(agent.strategy, BranchingStrategy):
                    branch = agent.strategy.current_branch or "main"
                    branch_info = f" [{branch}]"
                user_input = input(f"\nВы{branch_info}: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nВыход.")
                break

            if not user_input:
                continue

            parts = user_input.lower().split()
            cmd = parts[0]
            args = parts[1:]

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
                continue

            if cmd == "strategy":
                if not args:
                    print(f"[Текущая: {agent.strategy.get_name()}]")
                    print(f"[Доступные: {', '.join(Agent.STRATEGIES)}]")
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

            if cmd in ("checkpoint", "branch", "switch", "branches"):
                handle_branching_commands(agent, cmd, args)
                continue

            # Обычное сообщение
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
