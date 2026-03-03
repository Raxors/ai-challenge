import copy
from strategies.base import ContextStrategy


class BranchingStrategy(ContextStrategy):

    def __init__(self, **kwargs):
        self.checkpoints = {}
        self.branches = {}
        self.current_branch = None

    def get_name(self) -> str:
        branch = self.current_branch or "main"
        return f"Branching(branch={branch})"

    def prepare_messages(self, history: list) -> list:
        if self.current_branch and self.current_branch in self.branches:
            return list(self.branches[self.current_branch])
        return list(history)

    def on_user_message(self, history, message):
        if self.current_branch and self.current_branch in self.branches:
            self.branches[self.current_branch].append({"role": "user", "content": message})

    def on_assistant_message(self, history, message):
        if self.current_branch and self.current_branch in self.branches:
            self.branches[self.current_branch].append({"role": "assistant", "content": message})

    def create_checkpoint(self, name: str, history: list):
        source = history
        if self.current_branch and self.current_branch in self.branches:
            source = self.branches[self.current_branch]
        self.checkpoints[name] = copy.deepcopy(source)

    def create_branch(self, branch_name: str, checkpoint_name: str = None, history: list = None):
        if checkpoint_name and checkpoint_name in self.checkpoints:
            self.branches[branch_name] = copy.deepcopy(self.checkpoints[checkpoint_name])
        elif history:
            source = history
            if self.current_branch and self.current_branch in self.branches:
                source = self.branches[self.current_branch]
            self.branches[branch_name] = copy.deepcopy(source)
        else:
            self.branches[branch_name] = []

    def switch_branch(self, branch_name: str) -> bool:
        if branch_name == "main":
            self.current_branch = None
            return True
        if branch_name in self.branches:
            self.current_branch = branch_name
            return True
        return False

    def list_branches(self) -> dict:
        result = {"main": "active" if self.current_branch is None else "inactive"}
        for name, msgs in self.branches.items():
            non_system = [m for m in msgs if m["role"] != "system"]
            status = "active" if name == self.current_branch else "inactive"
            result[name] = f"{status} ({len(non_system)} msgs)"
        return result

    def list_checkpoints(self) -> dict:
        result = {}
        for name, msgs in self.checkpoints.items():
            non_system = [m for m in msgs if m["role"] != "system"]
            result[name] = f"{len(non_system)} messages"
        return result

    def get_branch_history(self) -> list:
        if self.current_branch and self.current_branch in self.branches:
            return self.branches[self.current_branch]
        return None

    def reset(self):
        self.checkpoints = {}
        self.branches = {}
        self.current_branch = None

    # === CLI hooks ===

    def handle_command(self, cmd, args, agent) -> bool:
        if cmd == "checkpoint":
            if not args:
                print("[Укажите имя чекпоинта: checkpoint <name>]")
                return True
            self.create_checkpoint(args[0], agent.history)
            print(f"[Чекпоинт '{args[0]}' сохранён ({len(agent.history)} сообщений)]")
            return True

        elif cmd == "branch":
            if not args:
                print("[Укажите имя ветки: branch <name> [from <checkpoint>]]")
                return True
            branch_name = args[0]
            if len(args) >= 3 and args[1] == "from":
                cp_name = args[2]
                if cp_name not in self.checkpoints:
                    print(f"[Чекпоинт '{cp_name}' не найден]")
                    return True
                self.create_branch(branch_name, checkpoint_name=cp_name)
                print(f"[Ветка '{branch_name}' создана от чекпоинта '{cp_name}']")
            else:
                self.create_branch(branch_name, history=agent.history)
                print(f"[Ветка '{branch_name}' создана от текущего состояния]")
            return True

        elif cmd == "switch":
            if not args:
                print("[Укажите имя ветки: switch <name>]")
                return True
            if self.switch_branch(args[0]):
                print(f"[Переключено на ветку '{args[0]}']")
            else:
                print(f"[Ветка '{args[0]}' не найдена]")
            return True

        elif cmd == "branches":
            print("\n  Чекпоинты:")
            for name, info in self.list_checkpoints().items():
                print(f"    {name}: {info}")
            print("\n  Ветки:")
            for name, info in self.list_branches().items():
                marker = " <--" if (name == (self.current_branch or "main")) else ""
                print(f"    {name}: {info}{marker}")
            print()
            return True

        return False

    def get_prompt_info(self) -> str:
        return self.current_branch or "main"

    def get_state_info(self) -> dict:
        return {
            "strategy": self.get_name(),
            "current_branch": self.current_branch or "main",
            "checkpoints": list(self.checkpoints.keys()),
            "branches": self.list_branches(),
        }
