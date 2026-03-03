import json
from core import LLMModel, split_messages
from storage import MemoryStore
from strategies.base import ContextStrategy
from strategies.memory_classifier import MemoryClassifier


class MemoryLayerStrategy(ContextStrategy):

    def __init__(self, model: LLMModel, window_size=6, db_path="memory.db", **kwargs):
        self.model = model
        self.window_size = window_size
        self.db_path = db_path

        self.store = MemoryStore(db_path)
        self.classifier = MemoryClassifier(model)

        self.current_task_id = "default"
        self.working_memory = self.store.load_working(self.current_task_id)
        self.task_change_suggested = False

    def get_name(self) -> str:
        return f"MemoryLayers(N={self.window_size})"

    def prepare_messages(self, history: list) -> list:
        system_msgs, recent = split_messages(history, self.window_size)
        result = list(system_msgs)

        long_term = self.store.get_long_term_dict()
        if long_term:
            lt_text = "[Долгосрочная память — устойчивые факты о пользователе]:\n"
            for category, items in long_term.items():
                lt_text += f"  [{category}]\n"
                for key, value in items.items():
                    lt_text += f"    {key}: {value}\n"
            result.append({"role": "system", "content": lt_text})

        if self.working_memory:
            wm_text = f"[Рабочая память — текущая задача: {self.current_task_id}]:\n"
            for key, value in self.working_memory.items():
                if isinstance(value, (list, dict)):
                    wm_text += f"  {key}: {json.dumps(value, ensure_ascii=False)}\n"
                else:
                    wm_text += f"  {key}: {value}\n"
            result.append({"role": "system", "content": wm_text})

        result.extend(recent)
        return result

    def on_assistant_message(self, history, message):
        recent = history[-4:] if len(history) > 4 else history
        long_term = self.store.get_long_term_dict()

        classification = self.classifier.classify(
            recent_messages=recent,
            working_memory=self.working_memory,
            long_term=long_term,
        )

        if classification["working_memory"]:
            self.working_memory = classification["working_memory"]
            self.store.save_working(self.current_task_id, self.working_memory)

        for update in classification["long_term_updates"]:
            self.store.save_long_term(
                category=update["category"],
                key=update["key"],
                value=update["value"],
            )

        self.task_change_suggested = classification["task_change_detected"]

    # -- Task management --

    def switch_task(self, task_id: str):
        self.store.save_working(self.current_task_id, self.working_memory)
        self.current_task_id = task_id
        self.working_memory = self.store.load_working(task_id)
        self.task_change_suggested = False

    def clear_working_memory(self):
        self.working_memory = {}
        self.store.save_working(self.current_task_id, {})

    def list_tasks(self) -> list:
        return self.store.list_tasks()

    def forget_all(self):
        self.store.delete_long_term()
        self.store.clear_all_working()
        self.working_memory = {}
        self.current_task_id = "default"

    # -- Lifecycle --

    def reset(self):
        self.clear_working_memory()
        self.task_change_suggested = False

    def close(self):
        self.store.save_working(self.current_task_id, self.working_memory)
        self.store.close()

    # -- CLI hooks --

    def handle_command(self, cmd, args, agent) -> bool:
        if cmd == "memory":
            if not args:
                print("\n  === Short-term (последние сообщения) ===")
                non_system = [m for m in agent.history if m["role"] != "system"]
                recent = non_system[-self.window_size:]
                for m in recent:
                    role = "User" if m["role"] == "user" else "Assistant"
                    text = m["content"][:80] + "..." if len(m["content"]) > 80 else m["content"]
                    print(f"    {role}: {text}")

                print(f"\n  === Working memory (задача: {self.current_task_id}) ===")
                if self.working_memory:
                    for k, v in self.working_memory.items():
                        print(f"    {k}: {v}")
                else:
                    print("    (пусто)")

                print("\n  === Long-term memory ===")
                lt = self.store.get_long_term_dict()
                if lt:
                    for cat, items in lt.items():
                        print(f"    [{cat}]")
                        for k, v in items.items():
                            print(f"      {k}: {v}")
                else:
                    print("    (пусто)")
                print()
                return True

            layer = args[0]
            if layer == "short":
                print("\n  Short-term (последние сообщения):")
                non_system = [m for m in agent.history if m["role"] != "system"]
                recent = non_system[-self.window_size:]
                for m in recent:
                    role = "User" if m["role"] == "user" else "Assistant"
                    print(f"    {role}: {m['content']}")
                print()

            elif layer == "working":
                print(f"\n  Working memory (задача: {self.current_task_id}):")
                if self.working_memory:
                    for k, v in self.working_memory.items():
                        print(f"    {k}: {v}")
                else:
                    print("    (пусто)")
                print()

            elif layer == "long":
                print("\n  Long-term memory:")
                lt = self.store.get_long_term_dict()
                if lt:
                    for cat, items in lt.items():
                        print(f"    [{cat}]")
                        for k, v in items.items():
                            print(f"      {k}: {v}")
                else:
                    print("    (пусто)")
                print()

            else:
                print("[Укажите слой: memory short / working / long]")
            return True

        elif cmd == "task":
            if not args:
                print(f"[Текущая задача: {self.current_task_id}]")
                print("[Переключить: task <name>]")
                return True
            task_name = " ".join(args)
            old_task = self.current_task_id
            self.switch_task(task_name)
            wm = self.working_memory
            if wm:
                print(f"[Переключено: {old_task} → {task_name} (загружено {len(wm)} ключей)]")
            else:
                print(f"[Переключено: {old_task} → {task_name} (новая задача)]")
            return True

        elif cmd == "tasks":
            tasks = self.list_tasks()
            if tasks:
                print("\n  Задачи:")
                for t in tasks:
                    marker = " <-- текущая" if t["task_id"] == self.current_task_id else ""
                    print(f"    {t['task_id']}: {len(t['keys'])} ключей (обновлено: {t['updated_at']}){marker}")
            else:
                print("[Нет сохранённых задач]")
            print()
            return True

        elif cmd == "forget":
            self.forget_all()
            print("[Вся память очищена (working + long-term)]")
            return True

        return False

    def get_prompt_info(self) -> str:
        return self.current_task_id

    def get_notifications(self) -> list[str]:
        if self.task_change_suggested:
            return ["[!] Обнаружена возможная смена задачи. Используйте 'task <name>' для переключения."]
        return []

    def get_state_info(self) -> dict:
        long_term = self.store.get_long_term()
        return {
            "strategy": self.get_name(),
            "current_task": self.current_task_id,
            "working_memory_keys": list(self.working_memory.keys()),
            "working_memory": dict(self.working_memory),
            "long_term_count": len(long_term),
            "long_term": long_term,
            "classification_tokens": self.classifier.classification_tokens,
            "task_change_suggested": self.task_change_suggested,
        }
