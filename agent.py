from llm_interface import LLMModel, LLMError
from history_store import HistoryStore


class Agent:
    """Агент с историей диалога. Работает с любой реализацией LLMModel."""

    def __init__(self, model: LLMModel, max_tokens=1024, system_prompt=None,
                 db_path="chat_history.db"):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.store = HistoryStore(db_path)

        # Загружаем сохранённую историю или создаём новую
        saved = self.store.load()
        if saved:
            self.history = saved
        else:
            self.history = []
            if system_prompt:
                self.history.append({"role": "system", "content": system_prompt})
                self.store.add("system", system_prompt)

    def ask(self, user_message):
        """Отправляет сообщение в LLM. Выбрасывает LLMError при ошибке."""
        self.history.append({"role": "user", "content": user_message})
        self.store.add("user", user_message)

        try:
            result = self.model.generate(self.history, max_tokens=self.max_tokens)
        except LLMError:
            # Откатываем сообщение пользователя, т.к. ответа не было
            self.history.pop()
            self.store.remove_last()
            raise

        self.history.append({"role": "assistant", "content": result["text"]})
        self.store.add("assistant", result["text"])
        return result

    def reset(self):
        self.history = []
        self.store.clear()
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
            self.store.add("system", self.system_prompt)

    def get_message_count(self):
        return self.store.count("user")

    def close(self):
        self.store.close()
