# Результати: Автокатегоризація тікетів через Embeddings

> Числа скрипт друкує сам: вставляйте блоки з консолі як є, нічого не переписуючи в таблиці. Власноруч ви пишете лише відповіді на питання, по 1–3 речення з посиланням на свої числа.
>
> Запуски:
> 1. `uv run starter.py --quick` — подивіться блок `ARI by k` і оберіть k.
> 2. `uv run starter.py --clusters <ваше k>` (з `OPENROUTER_API_KEY`, щоб кластери отримали назви від LLM) — з нього всі блоки секцій 1–3.
> 3. Бонус: `uv run starter.py --quick --rerank --clusters <ваше k>`. `clusters.png` перезаписується при кожному запуску, тому передавайте те саме k.

---

## 1. Кластеризація

Блок **`ARI by k`**:

```text
  ARI by k (same embeddings, your cluster_tickets):
    k=3   ARI = 0.309
    k=4   ARI = 0.433
    k=5   ARI = 0.508
    k=6   ARI = 0.528
    k=7   ARI = 0.524
    k=8   ARI = 0.497
    k=9   ARI = 0.492
    k=10  ARI = 0.528
    k=12  ARI = 0.379
```

**Яке k ви обрали і чому?** Де пік ARI, і що відбувається з кластерами при меншому та більшому k (які категорії зливаються або розпадаються)?

Обрав k=6: ARI росте з 0.309 (k=3) до піку 0.528 на k=6 і тримається на плато до k=10 (теж 0.528), а на k=12 падає до 0.379; k=7, що дорівнює кількості категорій, дає практично те саме (0.524), тож беру менше кластерів. Навіть на піку кластери не повторюють категорії: на k=6 `account` майже вся в одному кластері (40 з 43), зате 71 тікет із `returns`, `feature_request` і `general` опинились в одному кластері з 73, а `billing` розкидана щонайменше по чотирьох кластерах; при меншому k злиття сильніші — на k=5 до кластера з `shipping` притягуються затримка SMS (`account`) і сповіщень (`technical`). При більшому k розпадаються вже самі категорії: на k=12 `account` розкидана щонайменше по 4 кластерах із власними підтемами (пароль і 2FA, зміна email і телефону, вхід через Google/Apple, налаштування профілю), `billing` — по 4 (помилкові списання, повернення коштів, підписки, рахунки), `technical` — теж по 4, тому ARI падає до 0.379.


![clusters](clusters.png)

Блок **`Cluster Summary`** (з назвами від LLM):

```text
========================================================================
  Cluster Summary
========================================================================

  [0] Account Access and Security (47 tickets)
      - I forgot my password and the reset email never arrives
      - Can't log in even though I'm sure the password is correct
      - My account was locked after too many failed login attempts
      ... and 44 more

  [1] Customer Support and Feedback (73 tickets)
      - A free trial converted to a paid plan without any warning email
      - I need a VAT invoice for my company — how do I get one?
      - Photos in the listing show a blue product but I received a grey one
      ... and 70 more

  [2] Billing and Refunds (38 tickets)
      - The refund I was promised hasn't appeared on my card after two weeks
      - I requested a refund 10 days ago but haven't received any confirmation
      - Can I get an itemized receipt for my last three purchases?
      ... and 35 more

  [3] Technical Device And App Issues (52 tickets)
      - My profile picture won't upload — it just spins forever
      - I'm getting logged out from all devices simultaneously for no reason
      - My laptop screen flickers constantly after the latest system update
      ... and 49 more

  [4] Billing and Payment Issues (35 tickets)
      - I was charged twice for the same order on the same day
      - My subscription renewed automatically even though I cancelled it last 
      - The discount code I applied didn't reduce the price at checkout
      ... and 32 more

  [5] Shipping and Delivery Issues (55 tickets)
      - The verification SMS takes over 10 minutes to arrive
      - My company's purchase order number is missing from the invoice
      - Notifications on my phone are arriving hours after they're sent
      ... and 52 more
========================================================================
```

---

## 2. Пошук: 5 демо-запитів і Ticket Helper

Блок **`Summary — avg over all demo queries`**:

```text
=================================================================
  Summary — avg over all demo queries
  Method               Avg cat. P@5  Avg Latency
  ──────────────────────────────────────────────
  BM25 keyword            3.0 / 5       0.19 ms
  Cosine numpy            4.0 / 5       0.09 ms
  Fusion RRF              3.6 / 5       0.29 ms
  Qdrant :memory:         4.0 / 5       0.33 ms
  ==============================================
```

Блок **`Ticket Helper`**:

```text
========================================================================
  Ticket Helper (similar tickets + category hint)
========================================================================

  New ticket: "want to send the item back for a refund"

  Retrieval: Fusion RRF

  Suggested category (majority of top-3): returns
  Cluster (majority of top-3): "Billing and Refunds"

  Similar past tickets:
    1. [returns] I returned the wrong item by mistake — how do I send back the correct ...
    2. [billing] I requested a partial refund for an item I returned but got nothing...
    3. [returns] I changed my mind and want to return the item — is the return window s...
========================================================================
```

---

## 3. Leave-one-out на 300 запитах

П'ять демо-запитів показують, **як** поводиться кожен метод, але їх замало, щоб сказати, **який кращий**. Leave-one-out робить кожен з 300 тікетів запитом, а релевантними вважає інші тікети тієї ж категорії. Вердикт «no measurable difference» означає, що інтервал різниці містить нуль. Це повноцінний результат, а не провал.

Блок **`Leave-one-out over all 300 tickets`**:

```text
========================================================================
  Leave-one-out over all 300 tickets (query = ticket, relevant = same category)
  95% bootstrap intervals over queries
  Method             P@5                      MRR@5                   
  ──────────────────────────────────────────────────────────────────
  BM25 keyword       0.521 [0.488, 0.553]    0.734 [0.693, 0.772]
  Cosine numpy       0.646 [0.611, 0.680]    0.807 [0.771, 0.844]
  Fusion RRF         0.633 [0.603, 0.665]    0.822 [0.788, 0.856]
  Qdrant :memory:    0.646 [0.611, 0.680]    0.807 [0.771, 0.844]
  ──────────────────────────────────────────────────────────────────
  Paired comparison of P@5 (same queries, new minus old)
  Cosine vs BM25   +0.125 [+0.081, +0.171]  better on 164, worse on 91 queries -> better
  Fusion vs Cosine -0.013 [-0.043, +0.014]  better on 90, worse on 103 queries -> no measurable difference
  Qdrant vs Cosine +0.000 [+0.000, +0.000]  better on 0, worse on 0 queries -> no measurable difference
  ======================================================================
```

Блоки секцій 2–3 і висновки секції 4 отримано з токенізацією `split()` з TODO 4; після експерименту із секції 5 `starter.py` використовує regex, і leave-one-out дає BM25 0.516 і Fusion 0.629 (cosine і Qdrant без змін).

---

## 4. Висновки

**1.** На якому з 5 демо-запитів BM25 і cosine розійшлися найбільше і чому (збіг слів чи зміст)? Чи підтверджує це leave-one-out?

За перетином результатів BM25 і cosine однаково розійшлися на трьох запитах (1/5), але за якістю найбільше — на «package not delivered to my address»: BM25 = 2/5, cosine = 5/5. BM25 зачепився за загальні слова `address`, `not`, `to`, `my` і підняв «change the email address» (`account`) та «not a chatbot» (`general`), а cosine за змістом знайшов «the parcel as delivered but I never received it» без слова `package` — тобто BM25 бачить збіг слів, а cosine зміст. Leave-one-out це підтверджує: cosine 0.646 проти BM25 0.521, різниця +0.125 [+0.081, +0.171] — вимірно краще, хоча на 91 запиті з 300 BM25 все ж виграв.

**2.** Fusion RRF: що показує summary 5 демо-запитів і що показує leave-one-out? Чому висновок з 5 і з 300 запитів може відрізнятися?

На 5 демо-запитах Fusion у середньому гірший за cosine (3.6 проти 4.0 / 5), але весь розрив дає один запит: на «package not delivered» Fusion успадкував шум BM25 і впав з 5/5 до 2/5, а на «money problems with my purchase» навпаки обіграв обидва методи (4/5). На 300 запитах різниця P@5 лише −0.013 [−0.043, +0.014] (кращий на 90, гірший на 103) — вимірної різниці немає. На 5 запитах один запит може змінити середнє, а інтервалу не порахувати, тому демо показує «переможця», якого на 300 запитах немає.

**3.** Чому Qdrant `:memory:` дає ті самі результати, що й cosine numpy? І чому в production все одно береться Qdrant, а не numpy?

`:memory:` — це повний перебір без HNSW-індексу, тобто той самий точний cosine по всіх 300 векторах, що й у numpy: тому перетин 5/5 на кожному запиті, а в leave-one-out P@5 однаковий (0.646) і різниця +0.000 [+0.000, +0.000]. Обидва варіанти шукають за O(n·d) на запит, тому на 300 тікетах Qdrant навіть повільніший (0.33 ms проти 0.09 ms у numpy, плюс 2178 ms на завантаження) — на малих обсягах перемагають менші накладні витрати, а на мільйонах векторів лінійний перебір росте разом з n, тоді як сервер Qdrant з HNSW-індексом шукає наближено приблизно за O(log n) і лишається швидким. Крім того, Qdrant зберігає дані на диску, дозволяє додавати й видаляти тікети без перерахунку всієї матриці та фільтрувати пошук за payload (наприклад, лише `category = "billing"`).

---

## 5. Бонус: hybrid + rerank на 90 розмічених запитах

`uv run starter.py --quick --rerank`. Корпус: 300 тікетів + 40 з ідентифікаторами. Запити: 40 exact (з ідентифікатором) і 50 semantic. Деталі в `task.md`.

Блок **`Bonus: 90 labeled queries`** з токенізацією `split()` (як у TODO 4):

```text
==============================================================================
  Bonus: 90 labeled queries over 340 tickets (40 exact with an identifier, 50 semantic)
  Stage 1: cosine top-20 contains 98% of the grade-2 tickets (exact 100%, semantic 97%) — the ceiling for any reranker

  Method              nDCG@5 exact  semantic     all    MRR@5 exact  semantic   rerank ms/query
  ────────────────────────────────────────────────────────────────────────────────────────────
  Cosine                     0.744     0.833   0.793          0.665     0.890                 —
  BM25                       0.616     0.435   0.515          0.570     0.409                 —
  Fusion RRF                 0.771     0.613   0.683          0.735     0.611                 —
  Rerank ms-marco            0.798     0.737   0.764          0.729     0.784                13
  Rerank bge-base            0.870     0.758   0.808          0.835     0.824                34
  Rerank bge-v2-m3           0.901     0.844   0.869          0.875     0.882                92
  ────────────────────────────────────────────────────────────────────────────────────────────
  Paired nDCG@5 vs Cosine (same queries, method minus cosine), 95% bootstrap interval
  BM25               exact: -0.128 [-0.272, +0.024] no measurable difference | semantic: -0.398 [-0.481, -0.311] worse
  Fusion RRF         exact: +0.027 [-0.072, +0.133] no measurable difference | semantic: -0.220 [-0.294, -0.150] worse
  Rerank ms-marco    exact: +0.055 [-0.033, +0.146] no measurable difference | semantic: -0.096 [-0.142, -0.052] worse
  Rerank bge-base    exact: +0.127 [+0.018, +0.240] better | semantic: -0.075 [-0.120, -0.033] worse
  Rerank bge-v2-m3   exact: +0.157 [+0.052, +0.260] better | semantic: +0.011 [-0.030, +0.052] no measurable difference
  ============================================================================
```

Той самий блок після заміни токенізації на regex (див. `task.md`):

```text
==============================================================================
  Bonus: 90 labeled queries over 340 tickets (40 exact with an identifier, 50 semantic)
  Stage 1: cosine top-20 contains 98% of the grade-2 tickets (exact 100%, semantic 97%) — the ceiling for any reranker

  Method              nDCG@5 exact  semantic     all    MRR@5 exact  semantic   rerank ms/query
  ────────────────────────────────────────────────────────────────────────────────────────────
  Cosine                     0.744     0.833   0.793          0.665     0.890                 —
  BM25                       0.820     0.423   0.599          0.775     0.402                 —
  Fusion RRF                 0.914     0.606   0.743          0.884     0.582                 —
  Rerank ms-marco            0.798     0.737   0.764          0.729     0.784                 9
  Rerank bge-base            0.870     0.758   0.808          0.835     0.824                33
  Rerank bge-v2-m3           0.901     0.844   0.869          0.875     0.882                92
  ────────────────────────────────────────────────────────────────────────────────────────────
  Paired nDCG@5 vs Cosine (same queries, method minus cosine), 95% bootstrap interval
  BM25               exact: +0.076 [-0.027, +0.191] no measurable difference | semantic: -0.410 [-0.500, -0.321] worse
  Fusion RRF         exact: +0.170 [+0.108, +0.235] better | semantic: -0.227 [-0.305, -0.153] worse
  Rerank ms-marco    exact: +0.055 [-0.033, +0.146] no measurable difference | semantic: -0.096 [-0.142, -0.052] worse
  Rerank bge-base    exact: +0.127 [+0.018, +0.240] better | semantic: -0.075 [-0.120, -0.033] worse
  Rerank bge-v2-m3   exact: +0.157 [+0.052, +0.260] better | semantic: +0.011 [-0.030, +0.052] no measurable difference
  ============================================================================
```

**1.** Де Fusion RRF виграє в cosine, а де програє, і що змінила токенізація? Підказка: порівняйте токени запиту `any update on order #48213?` і тікета `My order #48213 is two weeks late...`. Що це означає для висновку «Fusion ≈ Cosine» із секції 3?

Fusion виграє на exact-запитах (nDCG@5 0.914 проти 0.744 у cosine, +0.170 [+0.108, +0.235] — better), бо BM25 ловить рідкісний токен ідентифікатора, і програє на semantic (0.606 проти 0.833, −0.227 [−0.305, −0.153] — worse), де точних збігів слів немає і BM25 додає шум. Токенізація вирішила саме exact: `split()` дає з запиту токен `#48213?`, а з тікета `#48213` — вони не збігаються, тому BM25 мав лише 0.616 і Fusion не відрізнявся від cosine (+0.027, no measurable difference); regex відрізає `?`, і BM25 зростає до 0.820, а Fusion стає вимірно кращим. Отже, «Fusion ≈ Cosine» із секції 3 — властивість тих даних (жодного ідентифікатора, релевантність = категорія), а не методу: на трафіку з номерами замовлень і кодами помилок Fusion вимірно виграє, і вибір залежить від частки exact-запитів.

**2.** Де reranker'и допомагають, а де ні? Чому cosine плутає `#48213` і `#48231`, а cross-encoder — рідше? Чому не всі моделі покращують semantic-запити (на яких даних вони тренувались)?

Reranker'и допомагають на exact-запитах — `bge-base` (+0.127) і `bge-v2-m3` (+0.157) вимірно кращі за cosine, MRR@5 зростає з 0.665 до 0.835 і 0.875, — а на semantic `ms-marco` (−0.096) і `bge-base` (−0.075) вимірно гірші і лише `bge-v2-m3` на рівні (+0.011); кандидати тут ні до чого, бо cosine top-20 уже містить 98% потрібних тікетів. Cosine стискає тікет в один вектор окремо від запиту, і чотири тікети, що відрізняються лише номером, отримують майже однакові вектори, а cross-encoder читає запит і тікет разом і через attention напряму зіставляє `48213` з `48231` — тому плутає рідше (хоча `ms-marco` на exact так і не став вимірно кращим). На semantic важать дані навчання: `ms-marco` тренувався на MS MARCO — веб-запити проти довгих абзаців, тож на коротких тікетах і перефразуваннях він не у своєму домені, `bge-base` ближчий до коротких текстів, а лише найбільша мультимовна `bge-v2-m3`, натренована на ширших даних, не поступається cosine, який і так сильний на semantic (nDCG 0.833).

**3.** Що б ви поставили в production для цього helpdesk: Fusion, rerank чи обидва? Врахуйте latency і рядок `Stage 1`.

Я б поставив cosine top-20 + rerank `bge-v2-m3`: це єдиний варіант, що не програє ні на exact (0.901 проти 0.744 у cosine), ні на semantic (0.844 проти 0.833, no measurable difference), і має найкращий загальний nDCG@5 (0.869), тоді як Fusion найкращий на exact (0.914), але вимірно гірший на semantic (−0.227). `Stage 1` показує, що cosine top-20 уже містить 98% потрібних тікетів (exact 100%), тобто глибини кандидатів вистачає і reranker'у лишається тільки правильно їх переставити, а ціна — +92 ms на запит на CPU, що для helpdesk, де агент відкриває тікет, а не шукає на кожну літеру, прийнятно; якщо latency критична, дешевший компроміс — `bge-base` (+33 ms, але вимірно гірший на semantic). Наступним кроком я б виміряв «обидва» — Fusion як кандидати для reranker'а: це могло б поєднати точний збіг ідентифікатора з розумінням змісту, але в цьому експерименті reranker отримував лише кандидатів cosine.

---

## 6. Бонус: свій запит

`uv run starter.py --query "ваш запит"`. Вставте блок цього запиту:

```text
=================================================================
  Query 1/1: "my money has not come back yet"

  Query: "my money has not come back yet"
========================================================================

  BM25 keyword (0.24 ms):
    1. [shipping] My order shows 'label created' for a week but it hasn't be... (5.3281)
    2. [account] Account language preference keeps switching back to Englis... (4.9707)
    3. [billing] The cashback reward was not applied to my account after th... (4.4516)
    4. [account] I enabled two-step verification but now can't get back in... (4.3981)
    5. [account] Login works on mobile but not on the desktop browser... (4.1648)

  Cosine numpy (pre-built) (0.10 ms):
    1. [returns] My return was received three weeks ago but no refund has b... (0.5264)
    2. [billing] The refund I was promised hasn't appeared on my card after... (0.4962)
    3. [billing] I requested a refund 10 days ago but haven't received any ... (0.4880)
    4. [returns] The product stopped working after three days of normal use... (0.4468)
    5. [billing] The cashback reward was not applied to my account after th... (0.4319)

  Fusion RRF (BM25 + cosine + merge) (0.35 ms):
    1. [billing] The cashback reward was not applied to my account after th... (0.0313)
    2. [shipping] My order shows 'label created' for a week but it hasn't be... (0.0164)
    3. [returns] My return was received three weeks ago but no refund has b... (0.0164)
    4. [account] Account language preference keeps switching back to Englis... (0.0161)
    5. [billing] The refund I was promised hasn't appeared on my card after... (0.0161)

  Qdrant :memory: (0.52 ms):
    1. [returns] My return was received three weeks ago but no refund has b... (0.5264)
    2. [billing] The refund I was promised hasn't appeared on my card after... (0.4962)
    3. [billing] I requested a refund 10 days ago but haven't received any ... (0.4880)
    4. [returns] The product stopped working after three days of normal use... (0.4468)
    5. [billing] The cashback reward was not applied to my account after th... (0.4319)

  BM25 vs Cosine overlap: 1/5
  Qdrant vs Cosine overlap: 5/5  (should be 5/5 at this size)
======================================================================
```

Чому BM25 і cosine дали саме такі результати?

У запиті немає жодного слова, яке є в тікетах про повернення коштів (`money` і `come` не трапляються в жодному знайденому тікеті), тому BM25 зачепився за службові слова `yet`, `back`, `not`, `my` і видав нерелевантні тікети з `shipping` і `account` («hasn't been picked up yet», «switching back to English»), а cosine за змістом знайшов тікети про неповернені кошти з `returns` і `billing` («no refund has been issued», «refund I was promised hasn't appeared»), хоча слова `refund` у запиті немає — перетин лише 1/5. Fusion RRF поставив першим єдиний спільний тікет «cashback reward was not applied» (3-тє місце в BM25 і 5-те в cosine), хоча він слабкий в обох списках, і пропустив у top-5 два шумні тікети BM25 — на запитах без точних збігів слів BM25 лише заважає.
