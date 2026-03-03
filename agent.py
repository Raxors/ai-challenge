from core import LLMModel, LLMError
from storage import HistoryStore
from token_counter import TokenCounter
from strategies import create_strategy, get_strategy_names


class Agent:

    def __init__(self, model: LLMModel, model_name="gpt-4o", max_tokens=1024,
                 system_prompt=None, db_path="chat_history.db",
                 strategy_name="sliding_window", window_size=10):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.store = HistoryStore(db_path)
        self.counter = TokenCounter(model_name)

        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cost = 0.0
        self.request_number = 0

        saved = self.store.load()
        if saved:
            self.history = saved
        else:
            self.history = []
            if system_prompt:
                self.history.append({"role": "system", "content": system_prompt})
                self.store.add("system", system_prompt)

        self.strategy_name = strategy_name
        self.strategy = create_strategy(strategy_name, model=self.model, window_size=window_size)

    def set_strategy(self, name, window_size=10):
        self.strategy_name = name
        self.strategy = create_strategy(name, model=self.model, window_size=window_size)
        return self.strategy.get_name()

    def ask(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        self.store.add("user", user_message)

        self.strategy.on_user_message(self.history, user_message)

        messages_to_send = self.strategy.prepare_messages(self.history)

        history_tokens = self.counter.count_messages(messages_to_send)
        context_limit = self.counter.get_limit()
        available = context_limit - history_tokens - self.max_tokens

        if available < 0:
            self.history.pop()
            self.store.remove_last()
            raise LLMError(
                f"Context overflow! History: {history_tokens} tokens, "
                f"limit: {context_limit}, "
                f"available for response: {available}. "
                f"Use 'reset' to clear the dialog."
            )

        try:
            result = self.model.generate(messages_to_send, max_tokens=self.max_tokens)
        except LLMError:
            self.history.pop()
            self.store.remove_last()
            raise

        self.history.append({"role": "assistant", "content": result["text"]})
        self.store.add("assistant", result["text"])

        self.strategy.on_assistant_message(self.history, result["text"])

        self.request_number += 1
        self.session_input_tokens += result["input_tokens"]
        self.session_output_tokens += result["output_tokens"]
        request_cost = self.counter.calc_cost(result["input_tokens"], result["output_tokens"])
        self.session_cost += request_cost

        sent_tokens = self.counter.count_messages(messages_to_send)
        full_history_tokens = self.counter.count_messages(self.history)
        usage_percent = (sent_tokens / context_limit) * 100

        result["metrics"] = {
            "request_number": self.request_number,
            "request_input_tokens": result["input_tokens"],
            "request_output_tokens": result["output_tokens"],
            "request_cost": request_cost,
            "sent_tokens": sent_tokens,
            "full_history_tokens": full_history_tokens,
            "context_limit": context_limit,
            "usage_percent": usage_percent,
            "session_total_input": self.session_input_tokens,
            "session_total_output": self.session_output_tokens,
            "session_total_cost": self.session_cost,
            "strategy_info": self.strategy.get_state_info(),
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

        self.strategy.reset()

    def get_message_count(self):
        return self.store.count("user")

    def close(self):
        self.strategy.close()
        self.store.close()
