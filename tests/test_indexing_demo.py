#!/usr/bin/env python3
"""
Демо-тест пайплайна индексации документов.

Показывает:
  1. Локальный индекс документов с эмбеддингами
  2. Метаданные каждого чанка
  3. Сравнение 2 стратегий chunking (fixed_size vs structural)
"""

import os
import sys
import time
import tempfile
import random
import struct

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from indexing.chunkers import FixedSizeChunker, StructuralChunker, _token_len
from indexing.embedder import Embedder
from indexing.index_store import IndexStore
from indexing.compare import compare_strategies, generate_report
from indexing.pipeline import IndexingPipeline, _read_pdf, SUPPORTED_EXTENSIONS

W = 70
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


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


def info(msg):
    print(f"    {msg}")


# ── Проверка наличия OpenAI API ──────────────────────────

def _has_openai_key():
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.environ.get("OPENAI_API_KEY"))


def _fake_embedding(dim=1536):
    """Генерация случайного вектора для демо без API."""
    return [random.gauss(0, 1) for _ in range(dim)]


# ── Сканирование документов ──────────────────────────────

def scan_docs():
    header("ЭТАП 0: СКАНИРОВАНИЕ ДОКУМЕНТОВ")

    if not os.path.isdir(DOCS_DIR):
        print(f"  ✗ Директория {DOCS_DIR} не найдена.")
        print(f"    Положите документы (PDF, MD, TXT, PY) в ./docs и перезапустите.")
        return []

    files = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if ext == ".pdf":
                text = _read_pdf(fpath)
            else:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if text.strip():
                files.append((fpath, text))
                tokens = _token_len(text)
                info(f"📄 {fname}  ({len(text):,} символов, ~{tokens:,} токенов)")
        except Exception as e:
            info(f"⚠ {fname}: ошибка — {e}")

    total_chars = sum(len(t) for _, t in files)
    total_tokens = sum(_token_len(t) for _, t in files)
    print()
    ok(f"Найдено файлов: {len(files)}")
    ok(f"Всего: {total_chars:,} символов, ~{total_tokens:,} токенов")

    return files


# ── Фаза 1: Сравнение стратегий ─────────────────────────

def phase1_compare(files):
    header("ЭТАП 1: СРАВНЕНИЕ СТРАТЕГИЙ CHUNKING")

    section("Fixed-size (256 токенов, overlap 64)")
    fixed = FixedSizeChunker(chunk_size=256, overlap_tokens=64)
    t0 = time.time()
    fixed_chunks = []
    for fpath, text in files:
        fixed_chunks.extend(fixed.chunk(text, source_file=fpath, title=os.path.basename(fpath)))
    fixed_time = time.time() - t0

    ok(f"Чанков: {len(fixed_chunks)}")
    token_counts = [c.metadata["token_count"] for c in fixed_chunks]
    ok(f"Токенов: {sum(token_counts):,} (среднее {sum(token_counts)//len(token_counts)}, "
       f"мин {min(token_counts)}, макс {max(token_counts)})")
    ok(f"Время: {fixed_time:.2f}s")

    section("Structural (Markdown/AST/абзацы, макс 512 токенов)")
    structural = StructuralChunker(max_chunk_tokens=512)
    t0 = time.time()
    struct_chunks = []
    for fpath, text in files:
        struct_chunks.extend(structural.chunk(text, source_file=fpath, title=os.path.basename(fpath)))
    struct_time = time.time() - t0

    ok(f"Чанков: {len(struct_chunks)}")
    token_counts_s = [c.metadata["token_count"] for c in struct_chunks]
    ok(f"Токенов: {sum(token_counts_s):,} (среднее {sum(token_counts_s)//len(token_counts_s)}, "
       f"мин {min(token_counts_s)}, макс {max(token_counts_s)})")
    ok(f"Время: {struct_time:.2f}s")

    section("Сравнительный отчёт")
    comparison = compare_strategies(files)
    report = generate_report(comparison)
    print()
    print(report)

    return fixed_chunks, struct_chunks


# ── Фаза 2: Метаданные чанков ───────────────────────────

def phase2_metadata(fixed_chunks, struct_chunks):
    header("ЭТАП 2: МЕТАДАННЫЕ ЧАНКОВ")

    section("Пример чанка — Fixed-size")
    c = fixed_chunks[0]
    info(f"Текст (первые 120 символов):")
    info(f"  \"{c.text[:120].strip()}...\"")
    info(f"Метаданные:")
    for k, v in c.metadata.items():
        val = v if not isinstance(v, str) or len(str(v)) < 60 else str(v)[:57] + "..."
        info(f"  {k}: {val}")

    section("Пример чанка — Structural")
    # Берём чанк с section_title, если есть
    sc = next((c for c in struct_chunks if c.metadata.get("section_title")), struct_chunks[0])
    info(f"Текст (первые 120 символов):")
    info(f"  \"{sc.text[:120].strip()}...\"")
    info(f"Метаданные:")
    for k, v in sc.metadata.items():
        val = v if not isinstance(v, str) or len(str(v)) < 60 else str(v)[:57] + "..."
        info(f"  {k}: {val}")

    section("Все поля метаданных")
    all_keys = set()
    for c in fixed_chunks[:20] + struct_chunks[:20]:
        all_keys.update(c.metadata.keys())
    for k in sorted(all_keys):
        info(f"• {k}")
    ok(f"Полей метаданных: {len(all_keys)}")


# ── Фаза 3: Локальный индекс с эмбеддингами ─────────────

def phase3_index(fixed_chunks, struct_chunks):
    header("ЭТАП 3: ЛОКАЛЬНЫЙ ИНДЕКС С ЭМБЕДДИНГАМИ")

    use_real = _has_openai_key()
    if use_real:
        info("OpenAI API key найден → реальные эмбеддинги")
    else:
        info("OpenAI API key не найден → случайные эмбеддинги (демо)")

    # Ограничиваем кол-во чанков для демо (API лимиты и скорость)
    max_chunks = 50
    demo_fixed = fixed_chunks[:max_chunks]
    demo_struct = struct_chunks[:max_chunks]

    section(f"Индексация (до {max_chunks} чанков каждой стратегии)")

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    try:
        store = IndexStore(db_path=db_path)

        embedder = None
        if use_real:
            embedder = Embedder()

        # Fixed-size чанки
        t0 = time.time()
        for c in demo_fixed:
            if use_real:
                emb = embedder.embed_one(c.text)
                emb_bytes = Embedder.to_bytes(emb)
            else:
                emb_bytes = Embedder.to_bytes(_fake_embedding())
            store.add_chunk(c, emb_bytes)
        fixed_time = time.time() - t0
        ok(f"Fixed-size: {len(demo_fixed)} чанков за {fixed_time:.2f}s")

        # Structural чанки
        t0 = time.time()
        for c in demo_struct:
            if use_real:
                emb = embedder.embed_one(c.text)
                emb_bytes = Embedder.to_bytes(emb)
            else:
                emb_bytes = Embedder.to_bytes(_fake_embedding())
            store.add_chunk(c, emb_bytes)
        struct_time = time.time() - t0
        ok(f"Structural: {len(demo_struct)} чанков за {struct_time:.2f}s")

        # Статистика
        section("Статистика индекса")
        stats = store.stats()
        for s in stats:
            info(f"Стратегия: {s['strategy']}")
            info(f"  Чанков:       {s['chunks']}")
            info(f"  Всего токенов: {s['total_tokens']:,}")
            info(f"  Среднее:      {s['avg_tokens']} токенов/чанк")
            info(f"  Файлов:       {s['files']}")

        # Размер БД
        db_size = os.path.getsize(db_path)
        ok(f"Размер SQLite: {db_size:,} bytes ({db_size/1024:.1f} KB)")

        # Проверка: чанк + эмбеддинг из БД
        section("Проверка: чтение из индекса")
        chunk_data = store.get_chunk(1)
        emb_data = store.get_embedding(1)
        emb_vec = Embedder.from_bytes(emb_data)
        info(f"Чанк #1: \"{chunk_data['text'][:80].strip()}...\"")
        info(f"  source: {chunk_data['source_file']}")
        info(f"  strategy: {chunk_data['strategy']}")
        info(f"  tokens: {chunk_data['token_count']}")
        info(f"  embedding dim: {len(emb_vec)}")
        info(f"  embedding[:5]: {[round(v, 4) for v in emb_vec[:5]]}")
        ok("Чанк и эмбеддинг успешно прочитаны из SQLite")

        # Поиск
        section("Семантический поиск (cosine similarity)")
        if use_real:
            query = "машинное обучение"
            info(f"Запрос: \"{query}\"")
            query_emb = embedder.embed_one(query)
            results = store.search_similar(query_emb, top_k=3)
            for i, r in enumerate(results, 1):
                info(f"  #{i} [sim={r['similarity']}] ({r['strategy']}) "
                     f"\"{r['text'][:70].strip()}...\"")
            ok(f"Найдено {len(results)} результатов")
        else:
            # С фейковыми эмбеддингами поиск не имеет смысла, но проверим механику
            query_emb = _fake_embedding()
            results = store.search_similar(query_emb, top_k=3)
            info(f"(Случайные эмбеддинги — результаты не семантические)")
            for i, r in enumerate(results, 1):
                info(f"  #{i} [sim={r['similarity']}] ({r['strategy']})")
            ok(f"Механика поиска работает ({len(results)} результатов)")

        store.close()

    finally:
        os.unlink(db_path)


# ── Main ─────────────────────────────────────────────────

def main():
    print(f"\n{'#' * W}")
    print(f"{'#' * W}")
    print(f"##  ДЕМО: ЛОКАЛЬНЫЙ ИНДЕКС ДОКУМЕНТОВ")
    print(f"##  Эмбеддинги + метаданные + сравнение стратегий chunking")
    print(f"{'#' * W}")
    print(f"{'#' * W}")

    # 0. Сканирование
    files = scan_docs()
    if not files:
        print("\n  Нет файлов для демо. Положите документы в ./docs/")
        return False

    # 1. Сравнение стратегий
    fixed_chunks, struct_chunks = phase1_compare(files)

    # 2. Метаданные
    phase2_metadata(fixed_chunks, struct_chunks)

    # 3. Индекс с эмбеддингами
    phase3_index(fixed_chunks, struct_chunks)

    # Итог
    header("ИТОГО")
    ok("Локальный индекс документов — SQLite (чанки + эмбеддинги)")
    ok("Метаданные: source_file, title, strategy, chunk_index, start_offset, token_count, section_title")
    ok("2 стратегии chunking: fixed_size (токены) vs structural (по структуре)")
    ok("Семантический поиск по cosine similarity")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
