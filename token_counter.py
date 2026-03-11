import tiktoken


class TokenCounter:
    """Подсчёт токенов для моделей OpenAI."""

    # Лимиты контекстного окна (input + output)
    MODEL_LIMITS = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-4": 8_192,
        "gpt-3.5-turbo": 16_385,
    }

    # Цены за 1M токенов ($)
    MODEL_PRICES = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, model="gpt-4o"):
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text):
        """Считает токены в строке."""
        return len(self.encoding.encode(text))

    def count_messages(self, messages):
        """Считает токены во всём списке сообщений (формат OpenAI chat)."""
        total = 0
        for msg in messages:
            total += 4  # каждое сообщение: <im_start>{role}\n{content}<im_end>\n
            total += self.count_text(msg["role"])
            content = msg.get("content")
            if content:
                total += self.count_text(content)
            # tool role messages have tool_call_id and name
            if "tool_call_id" in msg:
                total += self.count_text(msg["tool_call_id"])
            if "name" in msg:
                total += self.count_text(msg["name"])
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_text(tc["function"]["name"])
                    total += self.count_text(tc["function"]["arguments"])
        total += 2  # <im_start>assistant
        return total

    def get_limit(self):
        """Возвращает лимит контекстного окна модели."""
        return self.MODEL_LIMITS.get(self.model, 128_000)

    def get_usage_percent(self, messages):
        """Процент использования контекстного окна."""
        used = self.count_messages(messages)
        limit = self.get_limit()
        return (used / limit) * 100

    def calc_cost(self, input_tokens, output_tokens):
        """Считает стоимость запроса в долларах."""
        prices = self.MODEL_PRICES.get(self.model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        return input_cost + output_cost
