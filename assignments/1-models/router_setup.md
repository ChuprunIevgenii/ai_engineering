## Налаштування OpenRouter

OpenRouter — агрегатор API для сотень LLM (GPT-4o, Claude, Gemini, Llama, Gemma і т.д.) через єдиний інтерфейс.

---

### 1. Реєстрація і ключ

1. Перейди на [openrouter.ai](https://openrouter.ai) → авторизуйся через Google або GitHub
2. Зайди в розділ **Keys** → "Create Key"
3. Скопіюй ключ одразу — він показується лише один раз

---

### 2. Встановити ключ як змінну середовища

Скрипти читають ключ з `OPENROUTER_API_KEY`. Встанови його перед запуском:

**macOS / Linux (термінал):**
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```
Щоб не вводити кожного разу, додай цей рядок у `~/.zshrc` або `~/.bashrc`:
```bash
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.zshrc
source ~/.zshrc
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

**Перевірка без виведення самого ключа:**
```bash
# macOS / Linux
python3 -c 'import os; print("OPENROUTER_API_KEY is set" if os.environ.get("OPENROUTER_API_KEY") else "OPENROUTER_API_KEY is missing")'

# Windows PowerShell
python -c 'import os; print("OPENROUTER_API_KEY is set" if os.environ.get("OPENROUTER_API_KEY") else "OPENROUTER_API_KEY is missing")'
```
Не додавай ключ у код, `results.md`, `responses/` або Git.

---

### 3. Баланс

OpenRouter — prepaid модель. Для всіх завдань курсу вистачить **$5–10**:
- Великі моделі (GPT-4o): ~$0.003–0.008 за запит
- Малі моделі (Gemma 4B, LLaMA 8B): ~$0.0001 за запит

---

### ⚠️ Rate Limiting

Помилка `429 RateLimitError` — провайдер тимчасово перевантажений. Що робити:
- Почекай 1-2 хвилини і запусти ще раз
- Спробуй іншу малу модель (`meta-llama/llama-3.1-8b-instruct` тощо)
- Або запусти модель локально через Ollama (нижче)

---

## Налаштування Ollama (локальний запуск)

Альтернатива OpenRouter для малих моделей — безкоштовно, без rate limit, дані не покидають комп'ютер.

**Потрібно мінімум 8 ГБ RAM** (для 2B–4B моделей), 16 ГБ — для 8B+.

### Встановлення

Завантаж з [ollama.com](https://ollama.com) та встанови. На macOS/Windows в треї з'явиться іконка — це означає що сервер запущено.

### Завантаження моделі

```bash
# Рекомендована: невелика, добре справляється із завданнями курсу
ollama pull gemma3:2b

# Більша, краща якість (потребує ~8 ГБ RAM)
ollama pull llama3.1:8b
```

Перевір що все працює:
```bash
ollama run gemma3:2b
# Введи будь-яке питання, напиши /exit щоб вийти
```

### Використання у скриптах

У `starter.py` розкоментуй рядок з `ollama/` моделлю в `SMALL_MODELS`. Ollama API ключ не потрібен — скрипт знаходить локальний сервер автоматично.
