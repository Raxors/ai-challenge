import httpx
from openai import OpenAI, APIError, AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError
from core import LLMModel, LLMError


class OpenAIModel(LLMModel):

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
            raise LLMError("Quota exceeded. Check billing: https://platform.openai.com/settings/organization/billing")
        except AuthenticationError:
            raise LLMError("Invalid API key. Check OPENAI_API_KEY in .env file.")
        except APIConnectionError:
            raise LLMError("Connection error. Check your internet connection.")
        except APITimeoutError:
            raise LLMError("Request timed out. Try again later.")
        except APIError as e:
            raise LLMError(f"API error: {e.message}")
        except httpx.ConnectError:
            raise LLMError("No internet connection.")
