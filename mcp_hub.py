"""
MCPHub — управление множественными MCP-серверами.

Подключает удалённые MCP-серверы (stdio и SSE транспорт),
агрегирует их инструменты, маршрутизирует вызовы,
конвертирует в формат OpenAI function calling.
"""

import json
import re
from pathlib import Path
from mcp_client import MCPClient, SSEMCPClient


class MCPHub:
    """Хаб для управления множеством MCP-серверов."""

    def __init__(self, config_path="mcp_servers.json"):
        self.config_path = config_path
        self.servers = {}  # name -> {"config": dict, "client": MCPClient|SSEMCPClient, "tools": list}
        self._config = {"servers": {}}
        self._load_config()

    # ── Конфигурация ──────────────────────────────────────

    def _load_config(self):
        p = Path(self.config_path)
        if not p.exists():
            return
        try:
            self._config = json.loads(p.read_text(encoding="utf-8"))
            if "servers" not in self._config:
                self._config = {"servers": {}}
        except (json.JSONDecodeError, TypeError):
            self._config = {"servers": {}}

    def save_config(self):
        Path(self.config_path).write_text(
            json.dumps(self._config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Добавление / удаление серверов ────────────────────

    def add_stdio_server(self, name, command, env=None):
        """Добавить stdio-сервер (запускается как subprocess)."""
        entry = {"transport": "stdio", "command": command}
        if env:
            entry["env"] = env
        self._config["servers"][name] = entry
        self.save_config()

    def add_sse_server(self, name, url):
        """Добавить SSE-сервер (удалённый HTTP)."""
        self._config["servers"][name] = {"transport": "sse", "url": url}
        self.save_config()

    def remove_server(self, name):
        """Удалить сервер из конфига и отключить."""
        if name in self.servers:
            self.disconnect(name)
        if name in self._config["servers"]:
            del self._config["servers"][name]
            self.save_config()
            return True
        return False

    # ── Подключение / отключение ──────────────────────────

    def connect_all(self):
        """Подключиться ко всем сконфигурированным серверам.

        Returns:
            dict: {name: error_message} для серверов, к которым не удалось подключиться.
        """
        errors = {}
        for name in self._config["servers"]:
            try:
                self.connect(name)
            except Exception as e:
                errors[name] = str(e)
        return errors

    def connect(self, name):
        """Подключиться к конкретному серверу."""
        config = self._config["servers"].get(name)
        if not config:
            raise ValueError(f"Unknown server: {name}")

        # Отключить, если уже подключён
        if name in self.servers:
            self.disconnect(name)

        transport = config.get("transport", "stdio")

        if transport == "stdio":
            client = MCPClient(
                server_cmd=config["command"],
                env=config.get("env"),
            )
        elif transport == "sse":
            client = SSEMCPClient(url=config["url"])
        else:
            raise ValueError(f"Unknown transport: {transport}")

        client.connect()
        tools = client.list_tools()

        self.servers[name] = {
            "config": config,
            "client": client,
            "tools": tools,
        }

    def disconnect(self, name):
        """Отключиться от конкретного сервера."""
        if name in self.servers:
            try:
                self.servers[name]["client"].close()
            except Exception:
                pass
            del self.servers[name]

    def close_all(self):
        """Отключиться от всех серверов."""
        for name in list(self.servers.keys()):
            self.disconnect(name)

    # ── Инструменты ───────────────────────────────────────

    def list_all_tools(self):
        """Получить все инструменты со всех подключённых серверов.

        Returns:
            list: [{server, name, description, inputSchema}, ...]
        """
        all_tools = []
        for name, server in self.servers.items():
            for tool in server["tools"]:
                all_tools.append({
                    "server": name,
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}),
                })
        return all_tools

    def call_tool(self, server_name, tool_name, arguments):
        """Вызвать инструмент на конкретном сервере.

        Returns:
            dict: результат {content: [...], isError: bool}
        """
        if server_name not in self.servers:
            raise ValueError(f"Server not connected: {server_name}")
        client = self.servers[server_name]["client"]
        return client.call_tool(tool_name, arguments)

    # ── OpenAI function calling ───────────────────────────

    def get_openai_tools(self):
        """Конвертировать MCP-инструменты в формат OpenAI function calling.

        Имя функции: {server}__{tool} (двойное подчёркивание как разделитель).
        """
        openai_tools = []
        for name, server in self.servers.items():
            for tool in server["tools"]:
                func_name = self._make_func_name(name, tool["name"])
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {
                            "type": "object",
                            "properties": {},
                        }),
                    },
                })
        return openai_tools

    def resolve_tool_call(self, function_name):
        """Разобрать имя функции OpenAI обратно в (server_name, tool_name)."""
        if "__" in function_name:
            server_name, tool_name = function_name.split("__", 1)
            return server_name, tool_name
        return None, function_name

    @staticmethod
    def _make_func_name(server_name, tool_name):
        """Создать валидное имя функции для OpenAI: server__tool."""
        name = f"{server_name}__{tool_name}"
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return name[:64]

    # ── Запросы ───────────────────────────────────────────

    def get_server_names(self):
        """Все сконфигурированные серверы."""
        return list(self._config["servers"].keys())

    def get_connected_names(self):
        """Только подключённые серверы."""
        return list(self.servers.keys())

    def is_empty(self):
        """True если нет подключённых серверов."""
        return len(self.servers) == 0

    def get_server_config(self, name):
        """Конфигурация сервера (из файла конфига)."""
        return self._config["servers"].get(name, {})

    def get_server_info(self, name):
        """Информация о подключённом сервере."""
        if name not in self.servers:
            return None
        srv = self.servers[name]
        return {
            "transport": srv["config"].get("transport", "stdio"),
            "tools_count": len(srv["tools"]),
            "server_info": getattr(srv["client"], "server_info", None),
        }
