import json

from core import LLMModel, LLMError, UserProfile, TaskState, ProjectInvariants
from storage import HistoryStore
from token_counter import TokenCounter
from strategies import create_strategy, get_strategy_names
from mcp_hub import MCPHub


class Agent:

    def __init__(self, model: LLMModel, model_name="gpt-4o", max_tokens=1024,
                 system_prompt=None, db_path="chat_history.db",
                 strategy_name="sliding_window", window_size=10,
                 profile_path="user_profile.json",
                 task_state_path="task_state.json",
                 invariants_path="invariants.json",
                 mcp_config_path="mcp_servers.json"):
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

        self.profile_path = profile_path
        self.profile = UserProfile.load(profile_path)

        self.task_state_path = task_state_path
        self.task_state = TaskState.load(task_state_path)

        self.invariants_path = invariants_path
        self.invariants = ProjectInvariants.load(invariants_path)

        self.mcp_hub = MCPHub(config_path=mcp_config_path)

        self.strategy_name = strategy_name
        self.strategy = create_strategy(strategy_name, model=self.model, window_size=window_size)

    def set_strategy(self, name, window_size=10):
        self.strategy_name = name
        self.strategy = create_strategy(name, model=self.model, window_size=window_size)
        return self.strategy.get_name()

    @staticmethod
    def _find_system_end(messages):
        """Find the index after the last leading system message."""
        pos = 0
        for i, m in enumerate(messages):
            if m["role"] == "system":
                pos = i + 1
            else:
                break
        return pos

    def _inject_system_context(self, messages):
        """Inject profile, task state, and invariants as system messages.

        Insertion order (after existing system messages):
          1. Profile
          2. Task State
          3. Invariants (last = highest recency weight)
        """
        extras = []
        if not self.profile.is_empty():
            extras.append({"role": "system", "content": self.profile.to_prompt()})
        if not self.task_state.is_empty():
            extras.append({"role": "system", "content": self.task_state.to_prompt()})
        if not self.invariants.is_empty():
            extras.append({"role": "system", "content": self.invariants.to_prompt()})

        if not extras:
            return messages

        insert_pos = self._find_system_end(messages)
        return messages[:insert_pos] + extras + messages[insert_pos:]

    def ask(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        self.store.add("user", user_message)

        self.strategy.on_user_message(self.history, user_message)

        messages_to_send = self.strategy.prepare_messages(self.history)
        messages_to_send = self._inject_system_context(messages_to_send)

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

        # MCP tools для OpenAI function calling
        openai_tools = None
        if not self.mcp_hub.is_empty():
            openai_tools = self.mcp_hub.get_openai_tools() or None

        # Tool use loop: LLM может вызывать инструменты многократно
        total_input_tokens = 0
        total_output_tokens = 0
        tool_iterations = 0
        max_tool_iterations = 10

        while True:
            try:
                result = self.model.generate(
                    messages_to_send,
                    max_tokens=self.max_tokens,
                    tools=openai_tools,
                )
            except LLMError:
                if tool_iterations == 0:
                    self.history.pop()
                    self.store.remove_last()
                raise

            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]

            # Если нет tool_calls — обычный текстовый ответ, выходим
            if "tool_calls" not in result or not result["tool_calls"]:
                break

            tool_iterations += 1
            if tool_iterations > max_tool_iterations:
                if not result["text"]:
                    result["text"] = "[Превышен лимит итераций вызова инструментов]"
                break

            # Добавить assistant message с tool_calls в контекст
            assistant_msg = {
                "role": "assistant",
                "content": result.get("text") or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in result["tool_calls"]
                ],
            }
            messages_to_send.append(assistant_msg)

            # Выполнить каждый tool call и добавить результат
            for tc in result["tool_calls"]:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                server_name, tool_name = self.mcp_hub.resolve_tool_call(func_name)

                try:
                    tool_result = self.mcp_hub.call_tool(server_name, tool_name, args)
                    content = tool_result.get("content", [])
                    if content:
                        text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                        tool_text = "\n".join(text_parts)
                    else:
                        tool_text = json.dumps(tool_result, ensure_ascii=False)
                except Exception as e:
                    tool_text = f"Error: {e}"

                messages_to_send.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_text,
                })

        # Подменяем токены на суммарные за все итерации
        result["input_tokens"] = total_input_tokens
        result["output_tokens"] = total_output_tokens
        result["total_tokens"] = total_input_tokens + total_output_tokens

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
            "task_state": {
                "task_name": self.task_state.task_name,
                "phase": self.task_state.phase,
                "current_step": self.task_state.current_step,
            } if not self.task_state.is_empty() else None,
            "invariants_count": len(self.invariants.invariants) if not self.invariants.is_empty() else 0,
            "mcp_tools": len(openai_tools) if openai_tools else 0,
            "mcp_tool_iterations": tool_iterations,
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

    def reload_profile(self):
        self.profile = UserProfile.load(self.profile_path)

    def save_invariants(self):
        self.invariants.save(self.invariants_path)

    def reload_invariants(self):
        self.invariants = ProjectInvariants.load(self.invariants_path)

    def save_task_state(self):
        self.task_state.save(self.task_state_path)

    def reload_task_state(self):
        self.task_state = TaskState.load(self.task_state_path)

    def check_scheduler_notifications(self):
        """Проверить недоставленные уведомления планировщика."""
        try:
            if "scheduler" not in self.mcp_hub.get_connected_names():
                return []
            tool_result = self.mcp_hub.call_tool("scheduler", "scheduler_check_notifications", {})
            content = tool_result.get("content", [])
            if not content:
                return []
            text = content[0].get("text", "[]")
            notifications = json.loads(text)
            result = []
            for n in notifications:
                task_name = n.get("task_name", "?")
                r = n.get("result", {})
                if isinstance(r, str):
                    try:
                        r = json.loads(r)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if not isinstance(r, dict):
                    result.append(f'"{task_name}": {r}')
                    continue
                ntype = r.get("type", "")
                message = r.get("message", "")
                lines = [f'"{task_name}": {message}']
                # Сводка по задачам если есть
                stats = r.get("stats")
                tasks_summary = r.get("tasks_summary")
                if stats:
                    lines.append(
                        f"  Задачи: {stats['active']} активных / "
                        f"{stats['total']} всего / {stats['fired']} выполнено"
                    )
                if tasks_summary:
                    for t in tasks_summary:
                        interval = ""
                        if t.get("interval_seconds"):
                            iv = t["interval_seconds"]
                            if iv >= 3600:
                                interval = f" каждые {iv // 3600}ч"
                            elif iv >= 60:
                                interval = f" каждые {iv // 60}мин"
                            else:
                                interval = f" каждые {iv}с"
                        lines.append(
                            f"    #{t['id']} {t['name']} ({t['type']}/{t['callback']}{interval})"
                        )
                # Summary-тип: показать data_points
                if ntype == "summary":
                    dp = r.get("data_points", 0)
                    lines = [f'"{task_name}": сводка ({dp} точек данных)']
                    latest = r.get("latest", [])
                    for item in latest[:3]:
                        lines.append(f"    {json.dumps(item, ensure_ascii=False)}")
                result.append("\n".join(lines))
            return result
        except Exception:
            return []

    def get_message_count(self):
        return self.store.count("user")

    def close(self):
        self.mcp_hub.close_all()
        self.strategy.close()
        self.store.close()
