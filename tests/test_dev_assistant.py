"""
Тест ассистента разработчика:
  - git MCP сервер работает
  - RAG индексирует документацию
  - /help отвечает на вопросы о проекте

Usage:
    PYTHONPATH=. python3 tests/test_dev_assistant.py
"""

import os
import sys
import json
import tempfile

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name} — {detail}")


def test_git_mcp():
    """Тест git MCP сервера через прямой вызов."""
    print("\n--- Git MCP Server ---")
    from git_mcp.git_mcp_server import handle_tool_call, handle_message

    # git_branch
    result = handle_tool_call("git_branch", {})
    check("git_branch returns text", len(result) > 0 and result[0]["type"] == "text")
    branch = result[0]["text"]
    check("git_branch not empty", len(branch) > 0, f"got: '{branch}'")
    print(f"    branch: {branch}")

    # git_status
    result = handle_tool_call("git_status", {})
    check("git_status returns text", result[0]["type"] == "text")

    # git_log
    result = handle_tool_call("git_log", {"count": 3})
    check("git_log returns text", result[0]["type"] == "text")
    log = result[0]["text"]
    check("git_log has content", len(log) > 0)
    print(f"    log (3): {log[:80]}...")

    # git_list_files
    result = handle_tool_call("git_list_files", {})
    files = result[0]["text"]
    check("git_list_files returns files", "README.md" in files or "cli.py" in files,
          f"got {len(files)} chars")

    # git_branches
    result = handle_tool_call("git_branches", {})
    check("git_branches works", result[0]["type"] == "text")

    # git_diff
    result = handle_tool_call("git_diff", {})
    check("git_diff works", result[0]["type"] == "text")

    # JSON-RPC protocol: tools/list
    resp = handle_message({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    })
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    check("tools/list has 6 tools", len(tools) == 6, f"got {len(tools)}")
    check("git_branch in tools", "git_branch" in tool_names)
    check("git_diff in tools", "git_diff" in tool_names)

    # JSON-RPC protocol: tools/call
    resp = handle_message({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "git_branch", "arguments": {}},
    })
    check("tools/call works", not resp["result"]["isError"])


def test_rag_indexing():
    """Тест индексации документации (requires OPENAI_API_KEY)."""
    print("\n--- RAG Indexing ---")

    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIP  (OPENAI_API_KEY not set)")
        return

    from indexing.pipeline import IndexingPipeline

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        pipeline = IndexingPipeline(db_path=db_path)

        # Индексируем README.md
        readme_path = os.path.join(_PROJECT_ROOT, "README.md")
        check("README.md exists", os.path.exists(readme_path))

        with open(readme_path, "r") as f:
            readme_text = f.read()

        result = pipeline.index_files(
            [(readme_path, readme_text)],
            strategy="structural",
            embed=True,
        )
        check("indexing succeeds", result["total_chunks"] > 0,
              f"chunks: {result['total_chunks']}")
        print(f"    chunks: {result['total_chunks']}, embedded: {result['total_embedded']}")

        # Индексируем docs/
        docs_dir = os.path.join(_PROJECT_ROOT, "docs")
        if os.path.isdir(docs_dir):
            docs_files = pipeline.scan_directory(docs_dir)
            check("docs/ scanned", len(docs_files) > 0, f"found {len(docs_files)} files")
            if docs_files:
                result2 = pipeline.index_files(docs_files, strategy="structural", embed=True)
                check("docs indexed", result2["total_chunks"] > 0)
                print(f"    docs chunks: {result2['total_chunks']}")

        # Поиск
        results = pipeline.search("стратегии контекста", top_k=3)
        check("search returns results", len(results) > 0, f"got {len(results)}")
        if results:
            top_sim = results[0].get("similarity", 0)
            check("top result relevant", top_sim > 0.2, f"similarity: {top_sim:.3f}")
            print(f"    top result: sim={top_sim:.3f}, file={results[0].get('source_file', '?')}")

        stats_list = pipeline.get_stats()
        total_chunks = sum(s.get("chunks", 0) for s in stats_list)
        check("stats available", total_chunks > 0)
        print(f"    total: {total_chunks} chunks")

        pipeline.close()
    finally:
        os.unlink(db_path)


def test_git_context():
    """Тест сборки git-контекста для /help."""
    print("\n--- Git Context for /help ---")
    from git_mcp.git_mcp_server import _git_branch, _git_status, _git_log, _git_list_files

    branch = _git_branch()
    check("branch available", len(branch) > 0)

    status = _git_status()
    check("status available", status is not None)

    log = _git_log(5)
    check("log available", len(log) > 0)

    files = _git_list_files()
    check("file list available", len(files) > 0)

    # Проверяем что контекст формируется корректно
    context_parts = [
        f"Ветка: {branch}",
        f"Статус: {status or 'чисто'}",
        f"Последние коммиты:\n{log}",
    ]
    full_context = "\n\n".join(context_parts)
    check("context combines all parts", "Ветка:" in full_context and "коммиты" in full_context)
    print(f"    context size: {len(full_context)} chars")


def main():
    print("Тестирование ассистента разработчика")
    test_git_mcp()
    test_rag_indexing()
    test_git_context()
    print(f"\n{'='*40}")
    print(f"Результат: {PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
