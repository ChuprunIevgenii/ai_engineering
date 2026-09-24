# Домашнє завдання №0: Знайомство

> **Передумова:** перед виконанням домашок налаштуйте власний Git-репозиторій за інструкцією [`README.md`](../../README.md).

## Формат здачі

Запишіть відповіді на пункти 1 і 2 у файл [`results.md`](results.md). Для цієї домашньої роботи змініть і закомітьте **лише** `assignments/0-intro/results.md`, після чого відкрийте Pull Request у `main`.

Second Brain залишається вашим особистим робочим простором: його файли не потрібно додавати до репозиторію або Pull Request.

## 1. Банк ідей

Напишіть 3 кейси де б ви хотіли застосувати АІ у вашій роботі або власному проекті. Це можуть бути додаткові фічі або повноцінні продукти. Опишіть коротко ідею і суть рішення (1 абзац) — так само як би ви пітчили ідею вашим колегам чи не технічним слухачам.

## 2. Дослідження ринку

Знайдіть на LinkedIn, DOU або іншому сайті з позиціями 2-3 вакансії за запитом `"AI Engineer"` або `"LLM Engineer"` які вас цікавлять. Випишіть, над якими з вимог вам треба попрацювати.

---

## 3. Second Brain

Протягом курсу ви будете отримувати багато інформації — моделі, патерни, інструменти, помилки та інсайти. Більшість із цього забувається без системи.

**Second Brain** — це персональна база знань, яку ви поступово наповнюватимете протягом курсу. До кінця курсу у вас має залишитися не просто набір конспектів, а knowledge base, з якою може працювати ваш AI-агент.

Перед налаштуванням ознайомтеся з концепцією:

- **Основний матеріал:** [Tiago Forte — The PARA Method](https://fortelabs.com/blog/para/)
- **Опційно, для глибшого контексту:** [Building a Second Brain — Overview](https://fortelabs.com/blog/basboverview/)

Важлива ідея:

```text
Notes → Structured Knowledge → Retrieval → Context → AI Agent
```

Ваш Second Brain має залишатися у вигляді звичайних Markdown-файлів, які можна читати без AI, зберігати у власному Git-репозиторії та відкривати через [Obsidian](https://obsidian.md) або будь-який текстовий редактор.

> **Важливо:** Second Brain — це ваш особистий робочий простір. Створіть його в окремій директорії поза репозиторієм домашніх завдань. Не додавайте файли Second Brain до цього репозиторію та не комітьте їх у Pull Request.

### Варіант A — побудувати самостійно

Виконайте [`second_brain_init.md`](second_brain_init.md) через Claude Code, Codex або Cursor в окремій директорії, де ви хочете зберігати Second Brain.

У результаті ви отримаєте:

```text
00_Inbox/
01_Projects/
02_Areas/
03_Resources/
04_Archive/
99_System/
```

а також базові правила та agent routines для роботи з базою знань.

Цього варіанта достатньо для проходження курсу.

### Варіант B — Basic Memory (рекомендовано)

Використайте [Basic Memory](https://basicmemory.com/) як retrieval/memory layer поверх тієї самої Markdown-директорії. Markdown-файли залишаються source of truth, а AI отримує semantic і full-text search, зв'язки між нотатками та доступ до бази знань через MCP.

Для локального встановлення потрібні Python 3.12+ та [`uv`](https://docs.astral.sh/uv/). Спочатку створіть Second Brain через `second_brain_init.md`, а потім у його кореневій директорії виконайте:

> Команди нижче розраховані на macOS, Linux або WSL/Git Bash у Windows.

```bash
uv tool install basic-memory
basic-memory --version

bm project add ai-engineering-course "$(pwd)"
bm project default ai-engineering-course
bm project info ai-engineering-course
```

Якщо команда `basic-memory` не знайдена, виконайте `uv tool update-shell` і перезапустіть термінал.

> Перший запуск може тривати кілька хвилин: Basic Memory встановлює залежності та завантажує локальну embedding-модель. Під час індексації він також може додати службовий YAML frontmatter (`title`, `type`, `permalink`) до Markdown-файлів — це очікувана поведінка.

Після цього підключіть Basic Memory до AI-агента, яким користуєтеся.

#### Codex CLI

```bash
codex mcp add basic-memory bash -c \
  "basic-memory mcp --project ai-engineering-course"

codex mcp list
```

#### Claude Code

```bash
claude mcp add basic-memory -- \
  basic-memory mcp --project ai-engineering-course
```

Перевірити підключення можна командою `/mcp` у Claude Code.

#### Cursor

Створіть `.cursor/mcp.json` у кореневій директорії Second Brain:

```json
{
  "mcpServers": {
    "basic-memory": {
      "command": "basic-memory",
      "args": [
        "mcp",
        "--project",
        "ai-engineering-course"
      ]
    }
  }
}
```

Після налаштування почніть нову сесію та запитайте агента:

```text
What Basic Memory tools do you have?
```

### Варіант C — інше готове рішення

Ви можете використати інший Second Brain або persistent memory solution, наприклад Supermemory, Khoj чи власне рішення.

Головна вимога: knowledge base має працювати між різними AI-сесіями та залишатися доступною у Markdown або іншому переносимому форматі.

### Перший запис

Одразу після налаштування створіть файл `01_Projects/ai-engineering-course/lecture-00.md`:

```markdown
---
tags: [course, lecture]
date: YYYY-MM-DD
---
# Лекція 0 — Знайомство

## Навіщо я тут

(Яку конкретну проблему на роботі або у власному проекті я хочу вирішити за допомогою AI?)

## Мої 3 AI-кейси

(Перенесіть сюди ідеї з пункту 1 домашки.)

## Вакансії — що треба підтягнути

(З пункту 2 домашки — конкретні skills.)
```

### Експеримент із пам'яттю

Відкрийте кореневу директорію Second Brain як workspace або поточну робочу директорію AI-агента. Закрийте поточну AI-сесію, відкрийте нову та, не вказуючи шлях до нотатки, запитайте:

```text
What do you know about why I'm taking the AI Engineering course,
what projects I want to build, and what skills I want to improve?

Search my Second Brain before answering.
```

Для варіанта A агент шукатиме у Markdown-файлах. Для варіанта B він використовуватиме Basic Memory.

Мета експерименту — побачити різницю між історією чату та persistent external memory. `lecture-00.md` стане вашою точкою відліку: на випуску ви повернетеся до нього і порівняєте, як змінилося ваше розуміння AI Engineering та які з початкових ідей ви реалізували.

### Як використовувати Second Brain протягом курсу

Second Brain приносить користь лише тоді, коли ви регулярно додаєте та повторно використовуєте знання:

1. **Під час лекції:** складайте сирі нотатки, запитання та корисні посилання в `00_Inbox/`.
2. **Після лекції:** створіть або оновіть `01_Projects/ai-engineering-course/lecture-NN.md` — залиште головні ідеї, власні висновки та приклади застосування.
3. **Опрацюйте матеріал:** перенесіть універсальні концепти в `03_Resources/`, додайте теги та `[[wiki links]]` до пов'язаних нотаток.
4. **Перед домашкою:** попросіть агента знайти у Second Brain релевантні матеріали й сформувати контекст для завдання.
5. **Раз на тиждень:** розберіть `00_Inbox/`, оновіть активні проєкти та перенесіть неактуальні матеріали в `04_Archive/`.

Промпт для опрацювання матеріалів після лекції:

```text
Process the notes and links from my latest AI Engineering lecture in 00_Inbox.

Create or update the corresponding lecture-NN.md file, extract reusable
concepts into 03_Resources, and connect related notes with [[wiki links]].
Preserve the original inbox files in 04_Archive and do not invent information.
```

Промпт перед виконанням домашки:

```text
Search my Second Brain for notes relevant to [HOMEWORK TOPIC].

Summarize what I already know, identify gaps, and suggest which notes
I should review before starting. Cite the original notes with [[wiki links]]
and clearly separate retrieved knowledge from general recommendations.
```
