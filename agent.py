from llm_interface import LLMModel, LLMError
from history_store import HistoryStore
from token_counter import TokenCounter


class Agent:
    """Агент с историей диалога, подсчётом токенов и защитой от переполнения."""

    def __init__(self, model: LLMModel, model_name="gpt-4o", max_tokens=1024,
                 system_prompt=None, db_path="chat_history.db"):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.store = HistoryStore(db_path)
        self.counter = TokenCounter(model_name)

        # Статистика за сессию
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cost = 0.0
        self.request_number = 0

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
        """Отправляет сообщение в LLM. Возвращает результат с метриками токенов."""
        self.history.append({"role": "user", "content": user_message})
        self.store.add("user", user_message)

        # Считаем токены истории ДО запроса
        history_tokens = self.counter.count_messages(self.history)
        context_limit = self.counter.get_limit()
        available = context_limit - history_tokens - self.max_tokens

        # Проверяем переполнение
        # if available < 0:
        #     self.history.pop()
        #     self.store.remove_last()
        #     raise LLMError(
        #         f"Context overflow! History: {history_tokens} tokens, "
        #         f"limit: {context_limit}, "
        #         f"available for response: {available}. "
        #         f"Use 'reset' to clear the dialog."
        #     )

        try:
            result = self.model.generate(self.history, max_tokens=self.max_tokens)
        except LLMError:
            self.history.pop()
            self.store.remove_last()
            raise

        self.history.append({"role": "assistant", "content": result["text"]})
        self.store.add("assistant", result["text"])

        # Обновляем статистику
        self.request_number += 1
        self.session_input_tokens += result["input_tokens"]
        self.session_output_tokens += result["output_tokens"]
        request_cost = self.counter.calc_cost(result["input_tokens"], result["output_tokens"])
        self.session_cost += request_cost

        # Метрики после запроса
        history_tokens_after = self.counter.count_messages(self.history)
        usage_percent = self.counter.get_usage_percent(self.history)

        result["metrics"] = {
            "request_number": self.request_number,
            "request_input_tokens": result["input_tokens"],
            "request_output_tokens": result["output_tokens"],
            "request_cost": request_cost,
            "history_tokens": history_tokens_after,
            "context_limit": context_limit,
            "usage_percent": usage_percent,
            "session_total_input": self.session_input_tokens,
            "session_total_output": self.session_output_tokens,
            "session_total_cost": self.session_cost,
        }

        return result

    def reset(self):
        self.history = []
        self.store.clear()
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cost = 0.0
        self.request_number = 0
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
            self.store.add("system", self.system_prompt)

    def get_message_count(self):
        return self.store.count("user")

    def close(self):
        self.store.close()
