"""
Три стратегии управления контекстом:
  1. SlidingWindow — только последние N сообщений
  2. StickyFacts  — key-value факты + последние N сообщений
  3. Branching    — чекпоинты и ветки диалога
"""

import copy
import json
from abc import ABC, abstractmethod
from llm_interface import LLMModel, LLMError


class ContextStrategy(ABC):
    """Базовый интерфейс стратегии управления контекстом."""

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def prepare_messages(self, history: list) -> list:
        """Подготавливает сообщения для отправки в LLM."""
        pass

    @abstractmethod
    def on_user_message(self, history: list, message: str):
        """Вызывается после добавления сообщения пользователя."""
        pass

    @abstractmethod
    def on_assistant_message(self, history: list, message: str):
        """Вызывается после получения ответа ассистента."""
        pass

    @abstractmethod
    def get_state_info(self) -> dict:
        """Возвращает информацию о состоянии стратегии для метрик."""
        pass


# ─────────────────────────────────────────────
# Стратегия 1: Sliding Window
# ─────────────────────────────────────────────

class SlidingWindowStrategy(ContextStrategy):
    """Хранит только последние N сообщений, остальное отбрасывается."""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.dropped_count = 0

    def get_name(self) -> str:
        return f"SlidingWindow(N={self.window_size})"

    def prepare_messages(self, history: list) -> list:
        system_msgs = [m for m in history if m["role"] == "system"]
        non_system = [m for m in history if m["role"] != "system"]

        if len(non_system) > self.window_size:
            dropped = len(non_system) - self.window_size
            self.dropped_count += dropped
            non_system = non_system[-self.window_size:]

        return system_msgs + non_system

    def on_user_message(self, history, message):
        pass

    def on_assistant_message(self, history, message):
        pass

    def get_state_info(self) -> dict:
        return {
            "strategy": self.get_name(),
            "window_size": self.window_size,
            "dropped_messages": self.dropped_count,
        }


# ─────────────────────────────────────────────
# Стратегия 2: Sticky Facts / Key-Value Memory
# ─────────────────────────────────────────────

FACTS_EXTRACTION_PROMPT = """\
Ты — экстрактор фактов из диалога. Проанализируй последнее сообщение пользователя \
в контексте диалога и обнови JSON-объект с ключевыми фактами.

Текущие факты:
{current_facts}

Категории фактов для извлечения:
- goal: главная цель/задача пользователя
- constraints: ограничения и требования
- preferences: предпочтения пользователя
- decisions: принятые решения
- tech_stack: технологии, инструменты
- requirements: функциональные требования (список)
- agreements: договорённости
- names: важные имена, названия
- numbers: важные числа, даты, суммы
- context: дополнительный контекст

Верни ТОЛЬКО валидный JSON (без markdown, без ```). Сохрани все старые факты, \
обнови изменившиеся, добавь новые. Удали неактуальные. Если новой информации нет — \
верни текущие факты без изменений."""


class StickyFactsStrategy(ContextStrategy):
    """Извлекает ключевые факты из диалога и хранит их как key-value блок."""

    def __init__(self, model: LLMModel, window_size=6):
        self.model = model
        self.window_size = window_size
        self.facts = {}
        self.facts_extraction_tokens = 0

    def get_name(self) -> str:
        return f"StickyFacts(N={self.window_size})"

    def prepare_messages(self, history: list) -> list:
        system_msgs = [m for m in history if m["role"] == "system"]
        non_system = [m for m in history if m["role"] != "system"]

        # Берём последние N сообщений
        recent = non_system[-self.window_size:] if len(non_system) > self.window_size else non_system

        result = list(system_msgs)

        # Вставляем блок фактов, если есть
        if self.facts:
            facts_text = "[Ключевые факты из диалога]:\n"
            for key, value in self.facts.items():
                if isinstance(value, list):
                    facts_text += f"  {key}: {', '.join(str(v) for v in value)}\n"
                else:
                    facts_text += f"  {key}: {value}\n"
            result.append({"role": "system", "content": facts_text})

        result.extend(recent)
        return result

    def on_user_message(self, history, message):
        """Обновляет факты после каждого сообщения пользователя."""
        # Берём последние несколько сообщений для контекста
        recent = history[-4:] if len(history) > 4 else history
        context_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent if m["role"] != "system"
        )

        prompt = FACTS_EXTRACTION_PROMPT.format(
            current_facts=json.dumps(self.facts, ensure_ascii=False, indent=2) if self.facts else "{}"
        )

        try:
            result = self.model.generate(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Контекст диалога:\n{context_text}\n\nОбнови факты."},
                ],
                max_tokens=512,
            )
            self.facts_extraction_tokens += result["total_tokens"]
            # Парсим JSON из ответа
            text = result["text"].strip()
            # Убираем возможные markdown-обёртки
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            self.facts = json.loads(text)
        except (LLMError, json.JSONDecodeError):
            pass  # Если извлечение не удалось — оставляем старые факты

    def on_assistant_message(self, history, message):
        pass

    def get_state_info(self) -> dict:
        return {
            "strategy": self.get_name(),
            "window_size": self.window_size,
            "facts_count": len(self.facts),
            "facts": dict(self.facts),
            "facts_extraction_tokens": self.facts_extraction_tokens,
        }


# ─────────────────────────────────────────────
# Стратегия 3: Branching (ветки диалога)
# ─────────────────────────────────────────────

class BranchingStrategy(ContextStrategy):
    """Поддержка чекпоинтов и веток диалога."""

    def __init__(self):
        self.checkpoints = {}      # name -> list of messages (snapshot)
        self.branches = {}         # name -> list of messages (full history)
        self.current_branch = None # имя текущей ветки (None = основной диалог)

    def get_name(self) -> str:
        branch = self.current_branch or "main"
        return f"Branching(branch={branch})"

    def prepare_messages(self, history: list) -> list:
        # Если мы в ветке — используем историю ветки
        if self.current_branch and self.current_branch in self.branches:
            return list(self.branches[self.current_branch])
        return list(history)

    def on_user_message(self, history, message):
        # Если мы в ветке — добавляем сообщение в ветку
        if self.current_branch and self.current_branch in self.branches:
            self.branches[self.current_branch].append({"role": "user", "content": message})

    def on_assistant_message(self, history, message):
        # Если мы в ветке — добавляем ответ в ветку
        if self.current_branch and self.current_branch in self.branches:
            self.branches[self.current_branch].append({"role": "assistant", "content": message})

    def create_checkpoint(self, name: str, history: list):
        """Сохраняет текущее состояние диалога как чекпоинт."""
        source = history
        if self.current_branch and self.current_branch in self.branches:
            source = self.branches[self.current_branch]
        self.checkpoints[name] = copy.deepcopy(source)

    def create_branch(self, branch_name: str, checkpoint_name: str = None, history: list = None):
        """Создаёт ветку от чекпоинта или от текущего состояния."""
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
        """Переключается на ветку. Возвращает True при успехе."""
        if branch_name == "main":
            self.current_branch = None
            return True
        if branch_name in self.branches:
            self.current_branch = branch_name
            return True
        return False

    def list_branches(self) -> dict:
        """Возвращает информацию о ветках."""
        result = {"main": "active" if self.current_branch is None else "inactive"}
        for name, msgs in self.branches.items():
            non_system = [m for m in msgs if m["role"] != "system"]
            status = "active" if name == self.current_branch else "inactive"
            result[name] = f"{status} ({len(non_system)} msgs)"
        return result

    def list_checkpoints(self) -> dict:
        """Возвращает информацию о чекпоинтах."""
        result = {}
        for name, msgs in self.checkpoints.items():
            non_system = [m for m in msgs if m["role"] != "system"]
            result[name] = f"{len(non_system)} messages"
        return result

    def get_branch_history(self) -> list:
        """Возвращает историю текущей ветки."""
        if self.current_branch and self.current_branch in self.branches:
            return self.branches[self.current_branch]
        return None

    def get_state_info(self) -> dict:
        return {
            "strategy": self.get_name(),
            "current_branch": self.current_branch or "main",
            "checkpoints": list(self.checkpoints.keys()),
            "branches": self.list_branches(),
        }
