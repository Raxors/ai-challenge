# Dev Assistant — AI-ассистент разработчика

Приватный ассистент для любого проекта: отвечает на вопросы по коду, проводит code review через RAG + LLM, генерирует документацию.

## Возможности

- `/help <вопрос>` — отвечает на вопросы о проекте (RAG по документации + git-контекст)
- `/review` — AI code review изменённых файлов (баги, архитектура, безопасность, рекомендации)
- `/docs` — генерация документации по коду проекта
- `/git` — информация о репозитории (ветка, статус, коммиты)
- GitHub Action — автоматическое ревью на каждый PR

---

## Установка в свой проект

### 1. Скопировать папку

Скопируй папку `dev_assistant/` в корень своего проекта:

```
your-project/
├── dev_assistant/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py
│   ├── review.py
│   ├── docs_gen.py
│   ├── git_tools.py
│   ├── requirements.txt
│   └── indexing/
│       ├── __init__.py
│       ├── base_store.py
│       ├── chunkers.py
│       ├── embedder.py
│       ├── index_store.py
│       └── pipeline.py
├── .github/
│   └── workflows/
│       └── ai-review.yml
├── README.md
├── docs/
│   └── ...
└── ...
```

### 2. Установить зависимости

```bash
pip install openai tiktoken python-dotenv
```

Или из файла:

```bash
pip install -r dev_assistant/requirements.txt
```

### 3. Настроить API-ключ OpenAI

Создай файл `.env` в корне проекта:

```
OPENAI_API_KEY=sk-...
```

Или задай переменную окружения:

```bash
export OPENAI_API_KEY=sk-...
```

### 4. Запустить

```bash
python -m dev_assistant
```

При первом запуске ассистент автоматически проиндексирует `README.md`, `CLAUDE.md` и папку `docs/` для RAG.

---

## Команды

| Команда | Описание |
|---------|----------|
| `/help <вопрос>` | Вопрос о проекте (использует RAG + git) |
| `/review [base]` | Code review текущей ветки vs base (default: `main`) |
| `/docs [файл]` | Генерация документации (в консоль или файл) |
| `/index` | Переиндексировать документацию |
| `/stats` | Статистика RAG-индекса |
| `/git` | Ветка, статус, коммиты |
| `exit` | Выход |

Любой текст без `/` обрабатывается как `/help`.

История команд сохраняется между сессиями (стрелки вверх/вниз).

---

## GitHub Action — автоматическое ревью PR

### 1. Скопировать workflow

Создай файл `.github/workflows/ai-review.yml`:

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install openai tiktoken python-dotenv

      - name: Run AI Review
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m dev_assistant.review \
            --base ${{ github.event.pull_request.base.ref }} \
            --output review.md

      - name: Post review comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('review.md', 'utf8');
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `## AI Code Review\n\n${review}`
            });
```

### 2. Добавить секрет

В GitHub репозитории: **Settings → Secrets and variables → Actions → New repository secret**

- Name: `OPENAI_API_KEY`
- Value: твой OpenAI API ключ

### 3. Готово

Теперь при каждом открытии или обновлении PR бот автоматически:
1. Получает diff и изменённые файлы
2. Анализирует через LLM
3. Постит комментарий с ревью

---

## CLI без интерактивного режима

### Code review

```bash
# Ревью текущей ветки vs main
python -m dev_assistant.review --base main

# Сохранить в файл
python -m dev_assistant.review --base main --output review.md

# Без RAG (без индексации документации)
python -m dev_assistant.review --base main --no-rag
```

### Генерация документации

```bash
# В консоль
python -m dev_assistant.docs_gen

# В файл
python -m dev_assistant.docs_gen --output docs/PROJECT.md
```

---

## Что индексируется для RAG

По умолчанию при первом запуске индексируются:

- `README.md` — описание проекта
- `CLAUDE.md` — инструкции и конвенции (если есть)
- `docs/` — вся папка с документацией

Переиндексация: команда `/index` в интерактивном режиме.

Индекс хранится в файле `dev_assistant_index.db` в корне проекта.

---

## Что анализирует code review

| Категория | Что проверяет |
|-----------|--------------|
| Потенциальные баги | Крайние случаи, ошибки логики, необработанные исключения |
| Архитектурные проблемы | Нарушения принципов проекта, связанность, дублирование |
| Безопасность | Инъекции, утечка данных, хардкод секретов, OWASP Top 10 |
| Рекомендации | Конкретные предложения по улучшению |

Ревью учитывает контекст проекта из документации (RAG).

---

## Поддерживаемые языки

Генерация документации и анализ кода работают с любым языком:

`.py` `.js` `.ts` `.tsx` `.jsx` `.go` `.rs` `.java` `.kt` `.swift` `.c` `.cpp` `.h` `.cs` `.rb` `.php` `.lua` `.sh` `.sql` `.scala` `.ex` и другие.

---

## Структура

```
dev_assistant/
├── __init__.py          # Пакет
├── __main__.py          # Точка входа: python -m dev_assistant
├── main.py              # Интерактивный CLI
├── review.py            # AI code review (CLI + модуль)
├── docs_gen.py          # Генерация документации
├── git_tools.py         # Git-операции (standalone)
├── requirements.txt     # Зависимости
└── indexing/            # RAG-пайплайн (самодостаточный)
    ├── base_store.py    # SQLite base
    ├── chunkers.py      # Разбиение на чанки
    ├── embedder.py      # OpenAI embeddings
    ├── index_store.py   # Хранилище чанков
    └── pipeline.py      # Scan → chunk → embed → search
```

## Требования

- Python 3.9+
- Git
- OpenAI API ключ
