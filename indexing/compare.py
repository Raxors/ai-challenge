"""
Сравнение двух стратегий чанкинга: fixed_size vs structural.
Генерирует отчёт с метриками.
"""

import time
from indexing.chunkers import FixedSizeChunker, StructuralChunker, _token_len


def compare_strategies(files, fixed_kwargs=None, structural_kwargs=None):
    """
    Сравнить две стратегии на одном наборе файлов.

    files: список (filepath, text).
    Возвращает dict с метриками обеих стратегий.
    """
    fixed = FixedSizeChunker(**(fixed_kwargs or {}))
    structural = StructuralChunker(**(structural_kwargs or {}))

    results = {}

    for name, chunker in [("fixed_size", fixed), ("structural", structural)]:
        t0 = time.time()
        all_chunks = []
        for filepath, text in files:
            chunks = chunker.chunk(text, source_file=filepath)
            all_chunks.extend(chunks)
        elapsed = time.time() - t0

        token_counts = [c.metadata.get("token_count", 0) for c in all_chunks]
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0

        results[name] = {
            "chunks_count": len(all_chunks),
            "total_tokens": total_tokens,
            "avg_tokens": round(avg_tokens, 1),
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "time_sec": round(elapsed, 3),
            "files_processed": len(files),
        }

    return results


def generate_report(comparison, format="text"):
    """Генерирует текстовый отчёт сравнения стратегий."""
    lines = [
        "=" * 60,
        "  СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА",
        "=" * 60,
        "",
    ]

    for name, data in comparison.items():
        label = "Fixed-size (токены)" if name == "fixed_size" else "Structural (по структуре)"
        lines.append(f"▸ {label}")
        lines.append(f"  Чанков:         {data['chunks_count']}")
        lines.append(f"  Всего токенов:  {data['total_tokens']}")
        lines.append(f"  Среднее:        {data['avg_tokens']} токенов/чанк")
        lines.append(f"  Мин/Макс:       {data['min_tokens']} / {data['max_tokens']}")
        lines.append(f"  Время:          {data['time_sec']}s")
        lines.append(f"  Файлов:         {data['files_processed']}")
        lines.append("")

    # Вывод
    fixed = comparison.get("fixed_size", {})
    structural = comparison.get("structural", {})

    lines.append("-" * 60)
    lines.append("  ВЫВОДЫ")
    lines.append("-" * 60)

    if fixed and structural:
        fc = fixed["chunks_count"]
        sc = structural["chunks_count"]
        lines.append(
            f"  Fixed-size создаёт {'больше' if fc > sc else 'меньше'} чанков "
            f"({fc} vs {sc})"
        )
        lines.append(
            f"  Structural даёт {'более' if structural['max_tokens'] - structural['min_tokens'] > fixed['max_tokens'] - fixed['min_tokens'] else 'менее'} "
            f"разнородные по размеру чанки"
        )
        lines.append(
            "  Fixed-size лучше для равномерного покрытия (RAG)")
        lines.append(
            "  Structural лучше для семантически цельных фрагментов")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
