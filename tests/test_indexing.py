"""
Тесты модуля индексации документов.

Проверяет:
  1. Чанкеры: FixedSizeChunker, StructuralChunker
  2. IndexStore: CRUD, поиск, статистика
  3. Embedder: сериализация/десериализация
  4. Сравнение стратегий
"""

import os
import sys
import tempfile
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from indexing.chunkers import FixedSizeChunker, StructuralChunker, Chunk, _token_len
from indexing.index_store import IndexStore
from indexing.embedder import Embedder
from indexing.compare import compare_strategies, generate_report

W = 60


def header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def section(title):
    print(f"\n  {'-' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'-' * (W - 4)}")


def ok(msg):
    print(f"  ✓ {msg}")


def fail(msg):
    print(f"  ✗ {msg}")
    raise AssertionError(msg)


# ── Тестовые данные ──────────────────────────────────────

SAMPLE_MARKDOWN = """# Project Overview

This is a sample project for testing the indexing pipeline.
It contains multiple sections with different content.

## Architecture

The system uses a modular architecture with separate components
for chunking, embedding, and storage. Each component can be
used independently or as part of the full pipeline.

### Components

- Chunkers: split documents into smaller pieces
- Embedder: generate vector representations
- IndexStore: persist chunks and embeddings

## Installation

Run pip install -r requirements.txt to set up dependencies.
Make sure Python 3.10+ is available.

## Usage

Import the pipeline and call index_directory with the path
to your documents folder. Results are stored in SQLite.

### Configuration

You can customize chunk size, overlap, and embedding model
through constructor parameters.

### Examples

```python
from indexing import IndexingPipeline
pipeline = IndexingPipeline()
pipeline.index_directory("./docs")
results = pipeline.search("how to install")
```
"""

SAMPLE_PYTHON = '''"""Module docstring."""

import os
import sys

CONSTANT = 42


def helper_function(x, y):
    """A simple helper."""
    return x + y


class MyClass:
    """A sample class."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"


def another_function():
    """Another function at module level."""
    result = helper_function(1, 2)
    return result * CONSTANT
'''

SAMPLE_PLAIN = """First paragraph about machine learning and neural networks.
This paragraph discusses the fundamentals of deep learning.

Second paragraph about natural language processing.
NLP has evolved significantly with transformer models.

Third paragraph about computer vision applications.
Image recognition has achieved superhuman performance.

Fourth paragraph about reinforcement learning.
RL agents can learn complex strategies through trial and error."""


# ── Фаза 1: Чанкеры ─────────────────────────────────────

def test_chunkers():
    header("ФАЗА 1: ЧАНКЕРЫ")

    # -- FixedSizeChunker --
    section("FixedSizeChunker")

    chunker = FixedSizeChunker(chunk_size=64, overlap_tokens=16)
    assert chunker.strategy_name == "fixed_size"
    ok("strategy_name = 'fixed_size'")

    chunks = chunker.chunk(SAMPLE_MARKDOWN, source_file="test.md", title="Test Doc")
    assert len(chunks) > 0
    ok(f"Создано {len(chunks)} чанков из Markdown")

    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.text
        assert c.metadata["source_file"] == "test.md"
        assert c.metadata["strategy"] == "fixed_size"
        assert "token_count" in c.metadata
        assert c.metadata["token_count"] <= 64
    ok("Все чанки имеют корректные метаданные и размер <= chunk_size")

    # Проверка перекрытия: последние токены чанка N = первые токены чанка N+1
    if len(chunks) > 1:
        # overlap_tokens=16, значит start_offset чанка 1 = 64 - 16 = 48
        assert chunks[1].metadata["start_offset"] == 48
        ok("Перекрытие между чанками корректно (offset=48)")

    # -- StructuralChunker (Markdown) --
    section("StructuralChunker — Markdown")

    structural = StructuralChunker(max_chunk_tokens=512)
    assert structural.strategy_name == "structural"
    ok("strategy_name = 'structural'")

    chunks_md = structural.chunk(SAMPLE_MARKDOWN, source_file="readme.md", title="README")
    assert len(chunks_md) > 1
    ok(f"Создано {len(chunks_md)} чанков из Markdown (по заголовкам)")

    titles = [c.metadata.get("section_title", c.metadata.get("title", "")) for c in chunks_md]
    # Должны быть секции по заголовкам
    ok(f"Секции: {titles[:5]}...")

    for c in chunks_md:
        assert c.metadata["strategy"] == "structural"
    ok("Все чанки помечены стратегией 'structural'")

    # -- StructuralChunker (Python) --
    section("StructuralChunker — Python")

    chunks_py = structural.chunk(SAMPLE_PYTHON, source_file="module.py", title="Module")
    assert len(chunks_py) > 1
    ok(f"Создано {len(chunks_py)} чанков из Python (по AST)")

    section_titles = [c.metadata.get("section_title", "") for c in chunks_py]
    has_class = any("ClassDef" in t for t in section_titles)
    has_func = any("FunctionDef" in t for t in section_titles)
    assert has_class, f"Должен быть ClassDef в секциях: {section_titles}"
    ok("Найден ClassDef в секциях")
    assert has_func, f"Должен быть FunctionDef в секциях: {section_titles}"
    ok("Найден FunctionDef в секциях")

    # -- StructuralChunker (plain text) --
    section("StructuralChunker — Plain text")

    chunks_txt = structural.chunk(SAMPLE_PLAIN, source_file="notes.txt")
    assert len(chunks_txt) >= 3
    ok(f"Создано {len(chunks_txt)} чанков из plain text (по абзацам)")

    # -- Пустой текст --
    section("Граничные случаи")

    empty_chunks = chunker.chunk("", source_file="empty.txt")
    assert len(empty_chunks) == 0
    ok("Пустой текст → 0 чанков")

    tiny = chunker.chunk("Hello", source_file="tiny.txt")
    assert len(tiny) == 1
    ok("Короткий текст → 1 чанк")

    return True


# ── Фаза 2: IndexStore ──────────────────────────────────

def test_index_store():
    header("ФАЗА 2: INDEX STORE")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = IndexStore(db_path=db_path)

        # -- Добавление чанков --
        section("Добавление чанков")

        chunk1 = Chunk("Hello world", metadata={
            "source_file": "test.md",
            "title": "Test",
            "strategy": "fixed_size",
            "chunk_index": 0,
            "start_offset": 0,
            "token_count": 2,
        })

        fake_emb = [0.1] * 10
        emb_bytes = Embedder.to_bytes(fake_emb)

        cid = store.add_chunk(chunk1, emb_bytes)
        assert cid == 1
        ok(f"Чанк добавлен, id={cid}")

        # -- Чтение --
        section("Чтение чанков")

        retrieved = store.get_chunk(cid)
        assert retrieved is not None
        assert retrieved["text"] == "Hello world"
        assert retrieved["strategy"] == "fixed_size"
        assert retrieved["token_count"] == 2
        ok("Чанк прочитан корректно")

        # -- Эмбеддинг --
        emb_data = store.get_embedding(cid)
        assert emb_data is not None
        restored = Embedder.from_bytes(emb_data)
        assert len(restored) == 10
        assert abs(restored[0] - 0.1) < 1e-5
        ok("Эмбеддинг сохранён и восстановлен корректно")

        # -- Batch добавление --
        section("Batch операции")

        chunks = []
        embs = []
        for i in range(5):
            c = Chunk(f"Chunk number {i}", metadata={
                "source_file": "batch.md",
                "title": "Batch",
                "strategy": "structural",
                "chunk_index": i,
                "start_offset": i * 100,
                "token_count": 3,
            })
            chunks.append(c)
            embs.append(Embedder.to_bytes([float(i)] * 10))

        ids = store.add_chunks_batch(chunks, embs)
        assert len(ids) == 5
        ok(f"Batch: добавлено {len(ids)} чанков")

        # -- Фильтрация --
        section("Фильтрация и статистика")

        fixed = store.get_all_chunks(strategy="fixed_size")
        assert len(fixed) == 1
        ok(f"Фильтр strategy='fixed_size': {len(fixed)} чанков")

        structural = store.get_all_chunks(strategy="structural")
        assert len(structural) == 5
        ok(f"Фильтр strategy='structural': {len(structural)} чанков")

        by_file = store.get_all_chunks(source_file="batch.md")
        assert len(by_file) == 5
        ok(f"Фильтр source_file='batch.md': {len(by_file)} чанков")

        # -- Подсчёт --
        assert store.count_chunks() == 6
        assert store.count_chunks(strategy="structural") == 5
        ok("count_chunks корректен")

        # -- Статистика --
        stats = store.stats()
        assert len(stats) == 2
        ok(f"Статистика: {stats}")

        # -- Поиск --
        section("Поиск (cosine similarity)")

        query_emb = [3.0] * 10  # Ближе всего к chunk с emb=[3.0]*10
        results = store.search_similar(query_emb, top_k=2)
        assert len(results) <= 2
        assert results[0]["similarity"] > 0.9
        ok(f"Top result: id={results[0]['id']}, similarity={results[0]['similarity']}")

        # -- Очистка --
        section("Очистка")

        store.clear_all()
        assert store.count_chunks() == 0
        ok("clear_all: 0 чанков")

        store.close()

    finally:
        os.unlink(db_path)

    return True


# ── Фаза 3: Embedder сериализация ───────────────────────

def test_embedder_serialization():
    header("ФАЗА 3: EMBEDDER СЕРИАЛИЗАЦИЯ")

    section("to_bytes / from_bytes")

    original = [0.1, -0.5, 1.0, 0.0, 3.14159]
    blob = Embedder.to_bytes(original)
    assert isinstance(blob, bytes)
    assert len(blob) == len(original) * 4  # float32 = 4 bytes
    ok(f"to_bytes: {len(original)} floats → {len(blob)} bytes")

    restored = Embedder.from_bytes(blob)
    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        assert abs(a - b) < 1e-5, f"{a} != {b}"
    ok("from_bytes: восстановлены точно")

    # Пустой вектор
    empty = Embedder.to_bytes([])
    assert len(empty) == 0
    assert Embedder.from_bytes(empty) == []
    ok("Пустой вектор: корректно")

    # Большой вектор (1536 — размер text-embedding-3-small)
    big = list(range(1536))
    big_blob = Embedder.to_bytes(big)
    big_restored = Embedder.from_bytes(big_blob)
    assert len(big_restored) == 1536
    assert big_restored[0] == 0.0
    assert abs(big_restored[1535] - 1535.0) < 1e-2
    ok(f"Большой вектор (1536): {len(big_blob)} bytes, восстановлен")

    return True


# ── Фаза 4: Сравнение стратегий ─────────────────────────

def test_compare_strategies():
    header("ФАЗА 4: СРАВНЕНИЕ СТРАТЕГИЙ")

    section("compare_strategies")

    files = [
        ("readme.md", SAMPLE_MARKDOWN),
        ("module.py", SAMPLE_PYTHON),
        ("notes.txt", SAMPLE_PLAIN),
    ]

    result = compare_strategies(files)
    assert "fixed_size" in result
    assert "structural" in result
    ok("Обе стратегии в результате")

    for name in ("fixed_size", "structural"):
        data = result[name]
        assert data["chunks_count"] > 0
        assert data["total_tokens"] > 0
        assert data["files_processed"] == 3
        ok(f"{name}: {data['chunks_count']} чанков, {data['total_tokens']} токенов")

    section("generate_report")

    report = generate_report(result)
    assert "СРАВНЕНИЕ СТРАТЕГИЙ" in report
    assert "Fixed-size" in report
    assert "Structural" in report
    print()
    print(report)
    ok("Отчёт сгенерирован")

    return True


# ── Main ─────────────────────────────────────────────────

def main():
    passed = 0
    failed = 0
    tests = [
        ("Чанкеры", test_chunkers),
        ("IndexStore", test_index_store),
        ("Embedder сериализация", test_embedder_serialization),
        ("Сравнение стратегий", test_compare_strategies),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ ТЕСТ УПАЛ: {name} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * W}")
    print(f"  ИТОГО: {passed} passed, {failed} failed из {len(tests)}")
    print(f"{'=' * W}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
