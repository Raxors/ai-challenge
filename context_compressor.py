from llm_interface import LLMModel


class ContextCompressor:
    """Сжимает старые сообщения в summary, оставляя последние N сообщений как есть."""

    SUMMARY_PROMPT = (
        "Сделай краткое резюме следующего диалога. Сохрани ключевые факты, "
        "решения, имена, числа и контекст. Резюме должно быть на том же языке, "
        "что и диалог. Формат: сплошной текст, 3-5 предложений."
    )

    def __init__(self, model: LLMModel, keep_last=6):
        self.model = model
        self.keep_last = keep_last  # сколько последних сообщений хранить как есть
        self.summary = None

    def should_compress(self, history):
        """Проверяет, нужна ли компрессия (больше keep_last сообщений без system)."""
        non_system = [m for m in history if m["role"] != "system"]
        return len(non_system) > self.keep_last

    def compress(self, history):
        """Сжимает историю: старые сообщения -> summary, последние N -> как есть.

        Возвращает (новая_история, токены_потрачены_на_summary).
        """
        system_msgs = [m for m in history if m["role"] == "system"]
        non_system = [m for m in history if m["role"] != "system"]

        if len(non_system) <= self.keep_last:
            return history, 0

        # Разделяем: старые -> на сжатие, последние -> оставляем
        to_compress = non_system[:-self.keep_last]
        to_keep = non_system[-self.keep_last:]

        # Формируем текст для сжатия
        old_text = ""
        if self.summary:
            old_text += f"[Предыдущее резюме]: {self.summary}\n\n"
        for msg in to_compress:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            old_text += f"{role}: {msg['content']}\n\n"

        # Просим LLM сделать резюме
        summary_messages = [
            {"role": "system", "content": self.SUMMARY_PROMPT},
            {"role": "user", "content": old_text},
        ]
        result = self.model.generate(summary_messages, max_tokens=512)
        self.summary = result["text"]
        summary_tokens = result["total_tokens"]

        # Собираем новую историю: system + summary + последние N
        new_history = list(system_msgs)
        new_history.append({
            "role": "system",
            "content": f"[Резюме предыдущего диалога]: {self.summary}",
        })
        new_history.extend(to_keep)

        return new_history, summary_tokens
