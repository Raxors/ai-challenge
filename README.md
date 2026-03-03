# LLM Context Manager

CLI-агент с 4 стратегиями управления контекстом для OpenAI моделей. Подсчёт токенов, персистентная история, метрики стоимости.

## Быстрый старт

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Создайте `.env`:

```
OPENAI_API_KEY=sk-...
```

Запуск:

```bash
python cli.py
```

## Архитектура

```
core/               Интерфейсы, базовые классы, утилиты
  interfaces.py       LLMModel (ABC), LLMError
  base_store.py       BaseStore (ABC) — общий SQLite-бойлерплейт
  utils.py            parse_llm_json(), split_messages()

llm/                LLM-провайдеры
  openai_model.py     OpenAIModel — обёртка над OpenAI API

storage/            Персистентное хранение (SQLite)
  history_store.py    HistoryStore — история диалога (chat_history.db)
  memory_store.py     MemoryStore — рабочая + долговременная память (memory.db)

strategies/         Стратегии управления контекстом (pluggable)
  base.py             ContextStrategy ABC — 3 абстрактных + 9 опциональных методов
  sliding_window.py   Скользящее окно
  sticky_facts.py     Извлечение ключевых фактов через LLM
  branching.py        Чекпоинты и ветки диалога
  memory_layers.py    3-слойная память (short/working/long-term)
  memory_classifier.py  LLM-классификатор для memory_layers
  __init__.py         Реестр стратегий: create_strategy(), get_strategy_names()

agent.py            Agent — оркестратор (история, токены, стратегия)
cli.py              CLI — интерактивный REPL с метриками
token_counter.py    Подсчёт токенов и стоимости (tiktoken)
tests/              Автотесты
```

## 4 стратегии

### 1. `sliding_window` (по умолчанию)

Хранит последние N сообщений, старые отбрасывает. Нулевой overhead.

```
[system] + [последние N user/assistant сообщений]
```

### 2. `sticky_facts`

Извлекает ключевые факты из диалога через LLM и вставляет их как system-блок перед окном. Факты сохраняются даже когда сообщения уходят из окна.

```
[system] + [факты: goal, constraints, tech_stack...] + [последние N сообщений]
```

Категории фактов: goal, constraints, preferences, decisions, tech_stack, requirements, agreements, names, numbers, context.

### 3. `branching`

Чекпоинты и ветки диалога. Позволяет сохранить состояние, создать ветку, переключаться между ветками.

```
checkpoint save_point    # снимок текущего состояния
branch experiment        # ветка от текущего
branch alt from save_point  # ветка от чекпоинта
switch experiment        # переключиться
switch main              # вернуться
```

### 4. `memory_layers`

3-слойная система памяти с LLM-классификатором и персистентным хранением в SQLite.

| Слой | Что хранит | Где | Время жизни |
|------|-----------|-----|-------------|
| **Short-term** | Последние N сообщений | RAM (list) | Текущая сессия |
| **Working** | Контекст текущей задачи | SQLite `working_memory` | Пока задача активна |
| **Long-term** | Профиль, знания, решения | SQLite `long_term_memory` | Перманентно |

После каждого ответа `MemoryClassifier` анализирует диалог и **явно решает**, что сохранить в working memory, а что в long-term.

Категории long-term: `profile`, `preferences`, `decisions`, `knowledge`, `projects`, `contacts`.

```
[system] + [long-term блок] + [working memory блок] + [последние N сообщений]
```

## Команды CLI

### Общие

| Команда | Описание |
|---------|----------|
| `help` | Список команд |
| `exit` | Выход |
| `reset` | Сброс диалога и счётчиков |
| `stats` | Статистика сессии (токены, стоимость, стратегия) |
| `strategy` | Показать текущую стратегию |
| `strategy <name>` | Сменить стратегию |

Доступные стратегии: `sliding_window`, `sticky_facts`, `branching`, `memory_layers`.

### Sticky Facts

| Команда | Описание |
|---------|----------|
| `facts` | Показать извлечённые факты |

### Branching

| Команда | Описание |
|---------|----------|
| `checkpoint <name>` | Сохранить чекпоинт |
| `branch <name>` | Создать ветку от текущего состояния |
| `branch <name> from <cp>` | Создать ветку от чекпоинта |
| `switch <name>` | Переключиться на ветку |
| `switch main` | Вернуться в основной диалог |
| `branches` | Список веток и чекпоинтов |

### Memory Layers

| Команда | Описание |
|---------|----------|
| `memory` | Показать все 3 слоя памяти |
| `memory short` | Только short-term (последние сообщения) |
| `memory working` | Только рабочая память текущей задачи |
| `memory long` | Только долговременная память |
| `task <name>` | Переключиться на задачу (создаёт новую) |
| `tasks` | Список всех задач |
| `forget` | Полный сброс памяти (working + long-term) |

## Метрики

После каждого запроса CLI показывает:

```
──────────────────────────────────────────
  Запрос #3  |  Стратегия: MemoryLayers(N=6)
    Input tokens:    1250
    Output tokens:   180
    Стоимость:       $0.004925

  Отправлено в LLM: 1250 токенов  |  Полная история: 3400 токенов
    [########----------------------] 0.98% OK
    Задача: api_design  |  Working keys: 5  |  Long-term: 8 фактов
    Токены на классификацию: 2100

  Итого за сессию:
    Input:           3200 токенов
    Output:          520 токенов
    Стоимость:       $0.013200
──────────────────────────────────────────
```

## Поддерживаемые модели

| Модель | Контекст | Input $/1M | Output $/1M |
|--------|----------|------------|-------------|
| gpt-4o | 128K | $2.50 | $10.00 |
| gpt-4o-mini | 128K | $0.15 | $0.60 |
| gpt-4-turbo | 128K | $10.00 | $30.00 |
| gpt-4 | 8K | $30.00 | $60.00 |
| gpt-3.5-turbo | 16K | $0.50 | $1.50 |

## Тесты

4-фазный тест memory_layers с реальными LLM-вызовами:

```bash
python tests/test_memory_layers.py
```

Фазы:
1. Установка профиля пользователя -> проверка long-term
2. Задача A (API design) -> проверка working memory
3. Переключение на задачу B (dashboard) -> изоляция + long-term recall
4. Возврат к задаче A -> восстановление working memory

Проверяет:
- Какие данные попадают в каждый слой
- Как содержимое слоёв влияет на ответы ассистента
- Изоляцию рабочей памяти между задачами
- Персистентность long-term между задачами

## Расширение

Добавить новую стратегию:

```python
# strategies/my_strategy.py
from strategies.base import ContextStrategy

class MyStrategy(ContextStrategy):
    def get_name(self) -> str:
        return "MyStrategy"

    def prepare_messages(self, history: list) -> list:
        # собрать сообщения для LLM
        ...

    def get_state_info(self) -> dict:
        return {"strategy": self.get_name()}

    # опционально: reset(), close(), on_user_message(),
    # on_assistant_message(), handle_command(), get_prompt_info(),
    # get_notifications()
```

Зарегистрировать в `strategies/__init__.py`:

```python
from strategies.my_strategy import MyStrategy
register_strategy("my_strategy", MyStrategy, requires_model=False)
```

Готово — стратегия доступна через `strategy my_strategy` в CLI.
