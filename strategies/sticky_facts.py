import json
from core import LLMModel, LLMError, parse_llm_json, split_messages
from strategies.base import ContextStrategy


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

    def __init__(self, model: LLMModel, window_size=6, **kwargs):
        self.model = model
        self.window_size = window_size
        self.facts = {}
        self.facts_extraction_tokens = 0

    def get_name(self) -> str:
        return f"StickyFacts(N={self.window_size})"

    def prepare_messages(self, history: list) -> list:
        system_msgs, recent = split_messages(history, self.window_size)
        result = list(system_msgs)

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
            self.facts = parse_llm_json(result["text"])
        except (LLMError, json.JSONDecodeError):
            pass

    def reset(self):
        self.facts = {}
        self.facts_extraction_tokens = 0

    def get_state_info(self) -> dict:
        return {
            "strategy": self.get_name(),
            "window_size": self.window_size,
            "facts_count": len(self.facts),
            "facts": dict(self.facts),
            "facts_extraction_tokens": self.facts_extraction_tokens,
        }
