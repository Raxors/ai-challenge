import os
import sys
import textwrap
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from dotenv import load_dotenv

load_dotenv()


def main():
    client = OpenAI()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Enter your prompt: ")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80
        text = response.choices[0].message.content
        for paragraph in text.split("\n"):
            if paragraph:
                print(textwrap.fill(paragraph, width=width))
            else:
                print()
    except RateLimitError:
        print("Error: You exceeded your OpenAI quota. Please check your plan and billing at https://platform.openai.com/settings/organization/billing")
    except AuthenticationError:
        print("Error: Invalid API key. Please check your OPENAI_API_KEY in .env file.")
    except APIError as e:
        print(f"Error: OpenAI API error - {e.message}")


if __name__ == "__main__":
    main()
