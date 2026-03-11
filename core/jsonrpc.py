"""
JSON-RPC 2.0 базовые утилиты для MCP-серверов.

Общий код для всех MCP-серверов:
  - Формирование response/error
  - Диспетчеризация методов
  - Основной stdio цикл
"""

import sys
import json


def make_response(req_id, result):
    """Создать JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id, code, message):
    """Создать JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


class JSONRPCServer:
    """Базовый JSON-RPC 2.0 сервер для MCP протокола."""

    def __init__(self, name, version, protocol_version="2024-11-05"):
        self.name = name
        self.version = version
        self.protocol_version = protocol_version
        self._tools = []

    def set_tools(self, tools):
        self._tools = tools

    def handle_tool_call(self, name, arguments):
        """Override in subclass to handle tool calls.

        Returns:
            list: [{type: "text", text: "..."}]
        """
        raise NotImplementedError

    def handle_message(self, raw_message):
        """Обработать одно JSON-RPC сообщение. Вернуть ответ или None."""
        method = raw_message.get("method", "")
        params = raw_message.get("params", {})
        req_id = raw_message.get("id")

        if method == "initialize":
            result = {
                "protocolVersion": self.protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": self.version},
            }
            return make_response(req_id, result)

        elif method == "notifications/initialized":
            return None

        elif method == "tools/list":
            return make_response(req_id, {"tools": self._tools})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                content = self.handle_tool_call(tool_name, arguments)
                return make_response(req_id, {"content": content, "isError": False})
            except Exception as e:
                return make_response(req_id, {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True,
                })

        else:
            if req_id is not None:
                return make_error(req_id, -32601, f"Method not found: {method}")
            return None

    def run_stdio(self):
        """Читать JSON-RPC из stdin, обрабатывать, писать в stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                error_resp = make_error(None, -32700, f"Parse error: {e}")
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle_message(message)

            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
