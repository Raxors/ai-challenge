"""
MCP-клиенты — подключение к MCP-серверам через разные транспорты.

MCPClient      — stdio транспорт (subprocess)
SSEMCPClient   — SSE транспорт (HTTP, для удалённых серверов)

Оба реализуют одинаковый интерфейс: connect(), list_tools(), call_tool(), close().
"""

import subprocess
import json
import sys
import os
import time
import threading
import http.client
try:
    import queue
except ImportError:
    import Queue as queue
from urllib.parse import urlparse
import urllib.request


# ══════════════════════════════════════════════════════════
#  stdio транспорт
# ══════════════════════════════════════════════════════════

class MCPClient:
    """MCP-клиент поверх stdio (запускает сервер как subprocess)."""

    def __init__(self, server_cmd=None, env=None):
        """
        Args:
            server_cmd: команда запуска сервера (list).
            env: дополнительные переменные окружения (dict).
        """
        if server_cmd is None:
            server_cmd = [sys.executable, "mcp_server.py"]
        self.server_cmd = server_cmd
        self.env = env
        self.process = None
        self._next_id = 1
        self.server_info = None
        self.protocol_version = None

    def connect(self):
        """Запустить сервер и выполнить MCP handshake."""
        proc_env = dict(os.environ)
        # Подтянуть полный PATH из login shell (для Homebrew, nvm и т.д.)
        try:
            shell = os.environ.get("SHELL", "/bin/zsh")
            full_path = subprocess.check_output(
                [shell, "-l", "-c", "echo $PATH"],
                text=True, timeout=5,
            ).strip()
            if full_path:
                proc_env["PATH"] = full_path
        except Exception:
            pass
        if self.env:
            proc_env.update(self.env)

        self.process = subprocess.Popen(
            self.server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=proc_env,
        )

        init_result = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "context-agent-client", "version": "1.0.0"},
        })

        self.protocol_version = init_result.get("protocolVersion")
        self.server_info = init_result.get("serverInfo")
        self._notify("notifications/initialized")
        return init_result

    def list_tools(self):
        """Получить список инструментов."""
        result = self._request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name, arguments=None):
        """Вызвать инструмент."""
        result = self._request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        return result

    def close(self):
        """Завершить соединение."""
        if self.process:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self.process.wait(timeout=5)
            self.process = None

    def _send(self, message):
        line = json.dumps(message, ensure_ascii=False) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            stderr_output = ""
            if self.process.stderr:
                stderr_output = self.process.stderr.read()
            raise ConnectionError(
                f"MCP server closed unexpectedly. stderr: {stderr_output}"
            )
        return json.loads(line.strip())

    def _request(self, method, params=None):
        req_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        response = self._receive()
        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP error [{err.get('code')}]: {err.get('message')}")
        return response.get("result", {})

    def _notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ══════════════════════════════════════════════════════════
#  SSE транспорт (для удалённых HTTP-серверов)
# ══════════════════════════════════════════════════════════

class SSEMCPClient:
    """MCP-клиент поверх SSE (Server-Sent Events) для удалённых серверов."""

    def __init__(self, url):
        """
        Args:
            url: URL SSE-эндпоинта сервера (например http://localhost:8080/sse)
        """
        self.url = url
        self.messages_url = None
        self._response_queue = queue.Queue()
        self._reader_thread = None
        self._stop_event = threading.Event()
        self._next_id = 1
        self.server_info = None
        self.protocol_version = None
        self.process = None  # совместимость с MCPClient

    def connect(self):
        """Подключиться к SSE-серверу и выполнить MCP handshake."""
        self._reader_thread = threading.Thread(target=self._read_sse, daemon=True)
        self._reader_thread.start()

        # Ждём получения endpoint URL от сервера
        deadline = time.time() + 10
        while self.messages_url is None:
            if time.time() > deadline:
                raise ConnectionError("Timeout waiting for SSE endpoint event")
            time.sleep(0.1)

        init_result = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "context-agent-client", "version": "1.0.0"},
        })

        self.protocol_version = init_result.get("protocolVersion")
        self.server_info = init_result.get("serverInfo")
        self._notify("notifications/initialized")
        return init_result

    def list_tools(self):
        """Получить список инструментов."""
        result = self._request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name, arguments=None):
        """Вызвать инструмент."""
        result = self._request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        return result

    def close(self):
        """Закрыть SSE-соединение."""
        self._stop_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
        self.process = None

    def _read_sse(self):
        """Фоновый поток: читает SSE-события от сервера."""
        parsed = urlparse(self.url)

        if parsed.scheme == "https":
            import ssl
            conn = http.client.HTTPSConnection(
                parsed.hostname, parsed.port or 443,
                context=ssl.create_default_context(),
            )
        else:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80)

        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        try:
            conn.request("GET", path, headers={"Accept": "text/event-stream"})
            response = conn.getresponse()

            event_type = ""
            data_lines = []

            while not self._stop_event.is_set():
                line = response.readline()
                if not line:
                    break
                line = line.decode("utf-8").rstrip("\r\n")

                if line == "":
                    if data_lines:
                        data = "\n".join(data_lines)
                        if event_type == "endpoint":
                            if data.startswith("/"):
                                base = f"{parsed.scheme}://{parsed.hostname}"
                                if parsed.port:
                                    base += f":{parsed.port}"
                                self.messages_url = base + data
                            else:
                                self.messages_url = data
                        elif event_type == "message":
                            try:
                                self._response_queue.put(json.loads(data))
                            except json.JSONDecodeError:
                                pass
                        event_type = ""
                        data_lines = []
                elif line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
        except Exception:
            pass
        finally:
            conn.close()

    def _send(self, message):
        """Отправить JSON-RPC сообщение через POST."""
        data = json.dumps(message, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.messages_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=30)

    def _request(self, method, params=None):
        req_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

        stashed = []
        deadline = time.time() + 30
        try:
            while time.time() < deadline:
                try:
                    response = self._response_queue.get(timeout=1)
                    if response.get("id") == req_id:
                        if "error" in response:
                            err = response["error"]
                            raise RuntimeError(
                                f"MCP error [{err.get('code')}]: {err.get('message')}"
                            )
                        return response.get("result", {})
                    else:
                        stashed.append(response)
                except queue.Empty:
                    continue
        finally:
            for item in stashed:
                self._response_queue.put(item)

        raise TimeoutError(f"Timeout waiting for response to request {req_id}")

    def _notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
