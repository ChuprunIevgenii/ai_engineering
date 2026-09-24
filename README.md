## 1. Налаштування Git-репозиторію

Весь код і домашки ви будете вести у власному приватному GitHub-репозиторії. Викладач публікує нові завдання у цьому репозиторії — ви підтягуєте їх до себе через `upstream`.

**Крок 1 — Створіть приватний репозиторій на GitHub** і клонуйте його локально:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**Крок 2 — Одноразове налаштування upstream** (виконується лише один раз):

```bash
# Додаємо репозиторій викладача як джерело оновлень
git remote add upstream https://github.com/dkovalenko/ai_engineering_III.git

# Завантажуємо всі дані від викладача
git fetch upstream

# Зливаємо історію (прапорець потрібен лише для першого разу)
git merge upstream/main --allow-unrelated-histories -m "Merge upstream setup"

# Відправляємо у свій репозиторій
git push origin main
```

Після цього додайте викладача як collaborator у свій репозиторій: **Settings → Collaborators → Add people → `dkovalenko`**. Це потрібно щоб викладач міг залишати review на ваших pull request'ах.

**Крок 3 — Щотижнева рутина** (кожного разу коли викладач анонсує нову домашку):

```bash
# Переконуємось що ми в головній гілці
git checkout main

# Затягуємо свіжі зміни від викладача
git fetch upstream

# Мерджимо нове завдання (якщо виникнуть конфлікти — вирішуємо їх вручну)
git merge upstream/main -m "Fetch new hometask"

# Оновлюємо свій remote
git push origin main

# Створюємо гілку для виконання домашки
git checkout -b hometask-N-topic-name
```

Виконуєте домашку в цій гілці. Коли готово — пушите і відкриваєте Pull Request у свій `main`, додаєте `dkovalenko` як reviewer. Посилання на PR скидаєте в Slack у відповідний канал домашки.
