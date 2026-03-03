import json
from core import LLMModel, LLMError, parse_llm_json
from storage import LONG_TERM_CATEGORIES


CLASSIFICATION_PROMPT = """\
Ты — классификатор памяти диалогового агента. Твоя задача — проанализировать последние \
сообщения и решить, какую информацию нужно сохранить.

## Слои памяти

1. **working_memory** — информация о ТЕКУЩЕЙ задаче (цели, контекст, промежуточные решения, \
требования). Это оперативная память для активной работы.

2. **long_term_memory** — устойчивые факты о пользователе и его окружении, которые полезны \
МЕЖДУ задачами. Категории:
   - profile: имя, роль, компания, опыт
   - preferences: стиль кода, предпочтения инструментов, язык общения
   - decisions: архитектурные решения, выбранные технологии
   - knowledge: технический стек, домен, ключевые понятия
   - projects: названия проектов, их контекст
   - contacts: имена коллег, их роли

## Текущее состояние

Working memory (текущая задача):
{working_memory}

Long-term memory:
{long_term_memory}

## Инструкции

Проанализируй последние сообщения и верни JSON:
{{
  "working_memory": {{...}},
  "long_term_updates": [
    {{"category": "...", "key": "...", "value": "..."}},
    ...
  ],
  "task_change_detected": false
}}

Правила:
- working_memory: полный обновлённый словарь рабочей памяти (merge с текущим)
- long_term_updates: ТОЛЬКО новые или изменённые факты (не дублируй уже сохранённые)
- task_change_detected: true если пользователь явно переключился на другую тему/задачу
- Если нет новой информации — верни пустые обновления
- Верни ТОЛЬКО валидный JSON, без markdown-обёрток"""


class MemoryClassifier:

    def __init__(self, model: LLMModel):
        self.model = model
        self.classification_tokens = 0

    def classify(self, recent_messages: list, working_memory: dict, long_term: dict) -> dict:
        context_lines = []
        for m in recent_messages:
            if m["role"] == "system":
                continue
            role = "User" if m["role"] == "user" else "Assistant"
            context_lines.append(f"{role}: {m['content']}")
        context_text = "\n".join(context_lines)

        if not context_text.strip():
            return {
                "working_memory": working_memory,
                "long_term_updates": [],
                "task_change_detected": False,
            }

        wm_str = json.dumps(working_memory, ensure_ascii=False, indent=2) if working_memory else "{}"
        lt_str = json.dumps(long_term, ensure_ascii=False, indent=2) if long_term else "{}"

        prompt = CLASSIFICATION_PROMPT.format(
            working_memory=wm_str,
            long_term_memory=lt_str,
        )

        try:
            result = self.model.generate(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Последние сообщения:\n{context_text}\n\nКлассифицируй информацию."},
                ],
                max_tokens=1024,
            )
            self.classification_tokens += result["total_tokens"]

            parsed = parse_llm_json(result["text"])

            wm = parsed.get("working_memory", working_memory)
            if not isinstance(wm, dict):
                wm = working_memory

            lt_updates = parsed.get("long_term_updates", [])
            if not isinstance(lt_updates, list):
                lt_updates = []
            valid_updates = []
            for item in lt_updates:
                if (isinstance(item, dict)
                        and "category" in item
                        and "key" in item
                        and "value" in item
                        and item["category"] in LONG_TERM_CATEGORIES):
                    valid_updates.append(item)

            task_change = parsed.get("task_change_detected", False)

            return {
                "working_memory": wm,
                "long_term_updates": valid_updates,
                "task_change_detected": bool(task_change),
            }

        except (LLMError, json.JSONDecodeError, KeyError):
            return {
                "working_memory": working_memory,
                "long_term_updates": [],
                "task_change_detected": False,
            }
