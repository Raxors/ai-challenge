# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt    # openai, tiktoken, python-dotenv

# Run the CLI agent
python3 cli.py

# Run tests (custom runner, not pytest)
PYTHONPATH=. python3 tests/test_mcp.py
PYTHONPATH=. python3 tests/test_scheduler_mcp.py
PYTHONPATH=. python3 tests/test_pipeline_mcp.py
PYTHONPATH=. python3 tests/test_task_state.py
PYTHONPATH=. python3 tests/test_invariants.py
PYTHONPATH=. python3 tests/test_profiles.py
PYTHONPATH=. python3 tests/test_memory_layers.py   # requires OPENAI_API_KEY (makes real LLM calls)
PYTHONPATH=. python3 tests/test_indexing.py        # chunkers, store, embedder serialization
PYTHONPATH=. python3 tests/test_indexing_demo.py   # full demo: index PDF + search (requires OPENAI_API_KEY)
PYTHONPATH=. python3 tests/test_rag_comparison.py  # RAG vs No-RAG benchmark (requires OPENAI_API_KEY)
```

Tests use a custom `main()` runner with `check()` assertions — not unittest/pytest. Each test file is run directly with `PYTHONPATH=.` set.

## Configuration

- `.env` — `OPENAI_API_KEY`
- `mcp_servers.json` — MCP server registry (auto-loaded by MCPHub)
- `youtrack_mcp/.env.youtrack` — `YOUTRACK_URL`, `YOUTRACK_TOKEN`
- `scheduler_mcp/.env.scheduler` — `SCHEDULER_DB_PATH`, `SCHEDULER_CHECK_INTERVAL`

## Architecture

**Agent** (`agent.py`) is the central orchestrator. It composes:
- **LLM** (`llm/openai_model.py`) — OpenAI API wrapper
- **Strategy** (`strategies/`) — pluggable context management (4 implementations)
- **MCPHub** (`mcp_hub.py`) — multi-server MCP tool routing
- **HistoryStore** (`storage/history_store.py`) — SQLite chat persistence
- **TokenCounter** (`token_counter.py`) — token counting + cost calculation
- Plus: UserProfile, TaskState (FSM), ProjectInvariants — injected as system messages

### MCP Tool Flow

```
User message → Agent.ask()
  → Strategy.prepare_messages()
  → _inject_system_context() (profile, task_state, invariants, MCP tool list)
  → LLM.generate(messages, tools=openai_tools)
  → Tool use loop (max 10 iterations):
      LLM returns tool_calls → MCPHub.resolve_tool_call() → MCPHub.call_tool()
      → result appended to messages → LLM called again
  → Final text response + metrics
```

MCPHub converts MCP tools to OpenAI function calling format using `{server}__{tool}` naming. The agent routes calls back via `resolve_tool_call()`.

### MCP Servers

Each MCP server is a standalone JSON-RPC 2.0 process (stdio transport). All share `core/jsonrpc.py` (`JSONRPCServer` base class).

| Server | Directory | Purpose |
|--------|-----------|---------|
| youtrack_mcp | `youtrack_mcp/` | YouTrack issue tracking API |
| scheduler_mcp | `scheduler_mcp/` | Reminders + periodic tasks (SQLite + daemon thread) |
| prioritize_mcp | `prioritize_mcp/` | LLM-based task prioritization |
| filesaver_mcp | `filesaver_mcp/` | Save text to `pipeline_output/` |
| pipeline_mcp | `pipeline_mcp/` | Pipeline orchestrator (calls other 3 servers via MCPClient) |
| indexing_mcp | `indexing_mcp/` | Document indexing + RAG (PDF/MD/TXT → chunks → embeddings → search/ask) |

The agent itself also orchestrates MCP servers — it sees all tools and chains calls based on user intent.

### Context Strategies

All implement `strategies/base.py:ContextStrategy` ABC. Key method: `prepare_messages(history) → messages_to_send`.

- **sliding_window** — keeps last N messages, drops older ones
- **sticky_facts** — LLM extracts key facts, persisted across window slides
- **branching** — checkpoints + named branches of conversation
- **memory_layers** — 3-layer memory (short-term/working/long-term) with LLM classifier

### Adding a New MCP Server

1. Create `myserver_mcp/myserver_mcp_server.py` using `JSONRPCServer` base class
2. Define `TOOLS` list with `name`, `description`, `inputSchema`
3. Create `_TOOL_HANDLERS` dispatch dict
4. Add to `mcp_servers.json`:
   ```json
   "myserver": {"transport": "stdio", "command": ["python3", "myserver_mcp/myserver_mcp_server.py"]}
   ```

### Key Conventions

- MCP tool schemas must include `"items"` for array types (OpenAI validation requirement)
- All MCP servers add `_PROJECT_ROOT` to `sys.path` for importing `core`
- `handle_message()` and `handle_tool_call()` must be module-level functions (tests import them directly)
- Scheduler engine uses `PRAGMA auto_vacuum = FULL` and rotates old results (max 100 per task)
- CLI uses cbreak mode for input with scheduler notification polling; falls back to fcntl non-blocking
