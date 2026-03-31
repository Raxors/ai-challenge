"""
AI Code Review — анализ diff через LLM + RAG.

Используется:
  - из CLI: /review [base_branch]
  - из CI:  python -m dev_assistant.review --base main --output review.md
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dev_assistant.git_tools import (
    git_branch,
    git_diff_branch,
    git_changed_files,
    git_log,
    git_show_file,
)

REVIEW_SYSTEM_PROMPT = """\
Ты — опытный Senior разработчик, проводящий code review.

Проанализируй предоставленный diff и контекст проекта. Верни структурированное ревью.

Формат ответа:

## Сводка
Краткое описание что изменено (2-3 предложения).

## Потенциальные баги
Список потенциальных багов, ошибок, крайних случаев. Для каждого:
- Файл и описание проблемы
- Почему это может сломаться
Если багов нет — напиши "Потенциальных багов не обнаружено."

## Архитектурные проблемы
Проблемы с архитектурой, дизайном, связанностью, нарушением принципов проекта. Для каждой:
- Описание проблемы
- Чем это плохо
Если проблем нет — напиши "Архитектурных проблем не обнаружено."

## Безопасность
Проблемы безопасности (инъекции, утечка данных, хардкод секретов).
Если проблем нет — напиши "Проблем безопасности не обнаружено."

## Рекомендации
Конкретные предложения по улучшению кода (3-5 пунктов).

Правила:
- Будь конкретным — указывай файлы и строки
- Не придирайся к стилю, если код работает
- Фокусируйся на реальных проблемах, а не косметических
- Учитывай контекст проекта из документации
"""


def collect_review_context(base="main", pipeline=None):
    """Собрать весь контекст для ревью: diff, файлы, RAG."""
    context_parts = []

    # 1. Текущая ветка
    try:
        branch = git_branch()
        context_parts.append(f"Ветка: {branch} (base: {base})")
    except Exception:
        context_parts.append(f"Base: {base}")

    # 2. Коммиты
    try:
        log = git_log(10)
        context_parts.append(f"Последние коммиты:\n{log}")
    except Exception:
        pass

    # 3. Список изменённых файлов
    try:
        changed = git_changed_files(base)
        context_parts.append(f"Изменённые файлы:\n{changed}")
    except Exception as e:
        context_parts.append(f"Не удалось получить список файлов: {e}")
        changed = ""

    # 4. Diff
    try:
        diff = git_diff_branch(base)
        if len(diff) > 15000:
            diff = diff[:15000] + "\n\n... (diff обрезан, слишком длинный) ..."
        context_parts.append(f"Diff:\n```\n{diff}\n```")
    except Exception as e:
        context_parts.append(f"Не удалось получить diff: {e}")

    # 5. Полное содержимое изменённых файлов (для контекста)
    if changed:
        file_contents = []
        for fname in changed.split("\n"):
            fname = fname.strip()
            if not fname:
                continue
            try:
                content = git_show_file(fname)
                if len(content) > 3000:
                    content = content[:3000] + "\n... (файл обрезан) ..."
                file_contents.append(f"### {fname}\n```\n{content}\n```")
            except Exception:
                pass
        if file_contents:
            context_parts.append("Полные файлы:\n" + "\n\n".join(file_contents[:10]))

    # 6. RAG — документация проекта
    if pipeline:
        try:
            query = "архитектура проект структура конвенции правила"
            results = pipeline.search(query, top_k=3)
            if results and max(c.get("similarity", 0) for c in results) > 0.2:
                doc_parts = []
                for chunk in results:
                    source = os.path.basename(chunk.get("source_file", ""))
                    doc_parts.append(f"[{source}] {chunk['text']}")
                context_parts.append(
                    "Документация проекта:\n" + "\n---\n".join(doc_parts)
                )
        except Exception:
            pass

    return "\n\n".join(context_parts)


def run_review(base="main", pipeline=None, model="gpt-4o"):
    """Выполнить ревью и вернуть текст."""
    from openai import OpenAI
    client = OpenAI()

    context = collect_review_context(base, pipeline)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"Проведи code review:\n\n{context}"},
        ],
        max_tokens=4096,
    )

    review_text = resp.choices[0].message.content
    tokens_in = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens

    return {
        "review": review_text,
        "tokens": {"input": tokens_in, "output": tokens_out},
    }


def main():
    """CLI-точка входа для CI/CD."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI Code Review")
    parser.add_argument("--base", default="main", help="Базовая ветка (default: main)")
    parser.add_argument("--output", default=None, help="Файл для записи ревью (default: stdout)")
    parser.add_argument("--model", default="gpt-4o", help="Модель (default: gpt-4o)")
    parser.add_argument("--no-rag", action="store_true", help="Без RAG")
    args = parser.parse_args()

    pipeline = None
    if not args.no_rag:
        try:
            from dev_assistant.indexing.pipeline import IndexingPipeline
            db_path = os.path.join(PROJECT_ROOT, "dev_assistant_index.db")
            pipeline = IndexingPipeline(db_path=db_path)
        except Exception:
            print("WARNING: RAG недоступен, ревью без документации", file=sys.stderr)

    print(f"Анализирую изменения ({git_branch()} vs {args.base})...", file=sys.stderr)
    result = run_review(base=args.base, pipeline=pipeline, model=args.model)

    review = result["review"]
    tokens = result["tokens"]

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(review)
        print(f"Ревью сохранено в {args.output}", file=sys.stderr)
    else:
        print(review)

    print(f"Токены: {tokens['input']} вход / {tokens['output']} выход", file=sys.stderr)

    if pipeline:
        pipeline.close()


if __name__ == "__main__":
    main()
