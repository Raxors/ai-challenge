from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from llm_interface import LLMModel


class OpenAIModel(LLMModel):
    """Реализация интерфейса LLMModel для OpenAI API."""

    def __init__(self, model="gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def generate(self, messages, max_tokens=1024):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return {
                "text": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        except RateLimitError:
            return {"error": "Quota exceeded. Check billing: https://platform.openai.com/settings/organization/billing"}
        except AuthenticationError:
            return {"error": "Invalid API key. Check OPENAI_API_KEY in .env file."}
        except APIError as e:
            return {"error": f"API error: {e.message}"}
