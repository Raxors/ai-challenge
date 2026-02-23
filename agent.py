from llm_interface import LLMModel


class Agent:
    """Агент с историей диалога. Работает с любой реализацией LLMModel."""

    def __init__(self, model: LLMModel, max_tokens=1024, system_prompt=None):
        self.model = model
        self.max_tokens = max_tokens
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def ask(self, user_message):
        self.history.append({"role": "user", "content": user_message})

        result = self.model.generate(self.history, max_tokens=self.max_tokens)

        if "text" in result:
            self.history.append({"role": "assistant", "content": result["text"]})

        return result

    def reset(self):
        system = [m for m in self.history if m["role"] == "system"]
        self.history = system
