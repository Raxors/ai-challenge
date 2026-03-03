import json


def parse_llm_json(text: str):
    """Strip markdown code fences and parse JSON from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def split_messages(history: list, window_size: int):
    """Split history into (system_msgs, recent_non_system) with windowing."""
    system_msgs = [m for m in history if m["role"] == "system"]
    non_system = [m for m in history if m["role"] != "system"]
    recent = non_system[-window_size:] if len(non_system) > window_size else non_system
    return system_msgs, recent
