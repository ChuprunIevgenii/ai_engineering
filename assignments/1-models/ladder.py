# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "litellm",
# ]
# ///
"""
HW1 Capability Ladder — Кейс 5

Запускає одну задачу (Nebula-V reasoning) через 6 моделей різного розміру
(Dense + MoE архітектури) при T=0.1 і показує як змінюється якість, час та вартість.

Мета: побудувати ментальну модель "де виникає capability" і відповісти:
яка мінімальна модель достатня для цієї задачі?

Run:
  uv run ladder.py

Результати зберігаються в responses/case5_capability_ladder/.
Після запуску заповни таблицю в results.md.

Setup:
  export OPENROUTER_API_KEY="sk-or-..."
"""

import time
from pathlib import Path

import litellm
from litellm import completion

litellm.suppress_debug_info = True

# ── 6 моделей від найменшої до frontier ──────────────────────────────────────
# Поєднуємо Dense та MoE архітектури щоб показати різницю
LADDER_MODELS = [
    {"id": "openrouter/meta-llama/llama-3.2-1b-instruct",  "label": "Nano   Dense  1B          (Sep 2024)"},
    {"id": "openrouter/google/gemma-3-4b-it",              "label": "Small  Dense  4B          (Mar 2025)"},
    {"id": "openrouter/qwen/qwen3-30b-a3b",                "label": "Qwen3  MoE   3B/30B       (Apr 2025)"},  # 3B active / 30B total
    {"id": "openrouter/meta-llama/llama-4-scout",          "label": "Scout  MoE   17B/109B     (Apr 2025)"},  # 17B active / 109B total
    {"id": "openrouter/meta-llama/llama-3.3-70b-instruct", "label": "Large  Dense  70B         (Dec 2024)"},
    {"id": "openrouter/anthropic/claude-sonnet-4-6",       "label": "Claude Sonnet 4.6 Frontier (Feb 2026)"},
]

TEMPERATURE = 0.1
TOP_P = 0.9
SEED = 42

# ── Правила Nebula-V (Кейс 3) ─────────────────────────────────────────────────
NEBULA_RULES = """1. Типи акаунтів:
Користувачі 'Standard' мають ліміт 10 ГБ. Акаунти 'Academic' (верифікація через .edu) отримують 50 ГБ безкоштовно.

2. Тарифікація понад ліміт:
Кожен додатковий ГБ коштує $0.50/міс. При виборі річної підписки місячна ставка за додаткове зберігання знижується на 20% (з $0.50 до $0.40/ГБ/міс). ВАЖЛИВО: Для акаунтів 'Academic' ця знижка на місячну ставку становить лише 10% (ставка $0.45/ГБ/міс), але додатково з місячного рахунку знімається $2 базової студентської знижки.

3. Політика безпеки (Fair Use):
Денний ліміт завантаження (Ingress) становить 1 ТБ для всіх типів акаунтів. Якщо обсяг завантаження перевищує ліміт у 3 рази протягом доби, акаунт переходить у статус 'Pending'. Якщо перевищення складає понад 5 разів — акаунт блокується негайно без права апеляції.

4. Спеціальні умови:
Студенти (Academic) можуть подати запит на збільшення ліміту Ingress до 10 ТБ, але ця опція активується лише через 48 годин після верифікації карти. До цього моменту діють загальні правила безпеки (п. 3)."""

PROMPT = f"""Дай відповідь на основі тексту:

Якою буде вартість зберігання 120 ГБ даних для акаунта типу 'Academic', якщо він обрав річну підписку?

Чи буде такий акаунт заблоковано, якщо він завантажить 5 терабайт даних за один день?

Відповідь має бути короткою, з посиланням на конкретні пункти правил.

Правила:
{NEBULA_RULES}"""


def ask(model_id: str, prompt: str):
    """Returns (text, elapsed_s, cost_usd_or_none)."""
    start = time.time()
    resp = completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
        max_tokens=1024,
    )
    elapsed = time.time() - start
    cost = (getattr(resp, "_hidden_params", None) or {}).get("response_cost", None)
    msg = resp.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    return text, elapsed, cost


def save_response(out_dir: Path, label: str, text: str, elapsed: float, cost):
    case_dir = out_dir / "case5_capability_ladder"
    case_dir.mkdir(parents=True, exist_ok=True)
    slug = label.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("~", "").replace("/", "-")
    path = case_dir / f"{slug}_t{TEMPERATURE}_p{TOP_P}.txt"
    cost_str = f"${cost:.6f}" if cost is not None else "n/a"
    header = (
        f"# temperature: {TEMPERATURE} | top_p: {TOP_P} | "
        f"time: {elapsed:.1f}s | cost: {cost_str}\n"
    )
    path.write_text(header + text + "\n")
    return path


def main():
    out_dir = Path(__file__).parent / "responses"

    print("=" * 70)
    print("CAPABILITY LADDER — Кейс 5")
    print(f"Task: Nebula-V reasoning  |  T={TEMPERATURE}  |  top_p={TOP_P}  |  seed={SEED}")
    print("Правильна відповідь: $29.50/міс | 5× — гранична умова (строго >5× = блок, рівно 5× = Pending)")
    print("=" * 70)

    results = []

    for entry in LADDER_MODELS:
        model_id = entry["id"]
        label = entry["label"]
        short = model_id.removeprefix("openrouter/")

        print(f"\n{'─' * 70}")
        print(f"{label}  {short}")
        print("─" * 70)

        try:
            text, elapsed, cost = ask(model_id, PROMPT)
            cost_str = f"${cost:.6f}" if cost is not None else "n/a"
            print(f"Time: {elapsed:.1f}s | Cost: {cost_str}")
            print(text[:600] + ("…" if len(text) > 600 else ""))
            path = save_response(out_dir, label, text, elapsed, cost)
            print(f"  → {path.relative_to(out_dir.parent)}")
            results.append((label, elapsed, cost_str, "OK"))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((label, None, "n/a", "ERROR"))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("SUMMARY")
    print(f"{'═' * 70}")
    print(f"{'Tier':<42} | {'Time':>7} | {'Cost':>12} | Status")
    print(f"{'─' * 42}-+-{'─' * 7}-+-{'─' * 12}-+--------")
    for label, elapsed, cost_str, status in results:
        time_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
        print(f"{label:<42} | {time_str:>7} | {cost_str:>12} | {status}")

    print(f"\n→ Деталі відповідей: responses/case5_capability_ladder/")
    print("→ Заповни таблицю в results.md")


if __name__ == "__main__":
    main()
