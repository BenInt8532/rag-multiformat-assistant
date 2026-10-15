"""
Опыт 3 для статьи 2 (rag-evaluation): шум «чистого» судьи. Сколько плавает оценка Faithfulness у одного и того же
ответа при тех же кусках, без повторной генерации ответа модели.

Берётся готовый прогон (ответы и куски уже есть, модель ответа не вызывается). Фиксируются 5 ответов
(выбраны по логам и ключам из klyuchi.py; «верный/неполный/неверный» - по ключам и перечитыванию):
  1. верный, простой: срок оспаривания санкции (14 дней)                           - прогон 215644, вопрос 10
  2. верный, плавал в прошлом опыте (1.0 / 0.5 / 0.5): блокировка за повторное нарушение - прогон 215644, вопрос 7
     (выбран именно потому, что плавал, поэтому шум на нём завышает типичный)
  3. неполный: комиссия «5 процентов» (нужно 5% + 2%)                              - прогон 215644, вопрос 2
  4. неверный: остаток 2000 рублей «превышает минимальную сумму», в кусках минимума нет - прогон 215644, вопрос 4
  5. неверный и уверенный: «Школа кода» 1800 руб. вместо 1500 (число из соседней строки) -
     прогон 2026-10-14-210447-rules-real-topk8-ragas.json, вопрос 24; в нём судья ставил 1.0
     (второй прогон должен лежать в той же папке progony; если его нет, пункт 5 пропускается)
Каждый ответ оценивается REPEATS раз (по умолчанию 20). Считаются среднее, размах, стандартное отклонение
и сколько разных значений выпало. Это шум одного судьи при неизменных входных данных.

Источник: progony/2026-10-14-215644-rules-real-topk8-ragas.json.
Результат - statya-ocenka/opyty/opyt-shum.json и сводка на экран.

Запуск из папки проекта (PowerShell):
  python statya-ocenka/opyt_shum.py statya-ocenka/progony/2026-10-14-215644-rules-real-topk8-ragas.json
Необязательно: --repeats 20  --parallel 4
"""
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sudya_obshchee as common

EXTRA_RUN = "2026-10-14-210447-rules-real-topk8-ragas.json"

# (метка, вид, файл прогона или None = основной, вопрос, начало ответа для сверки)
ITEMS = [
    ("1 верный простой", "верный", None,
     "В течение какого срока можно оспорить решение о санкции?", "Решение о санкции можно оспорить в течение 14 дней"),
    ("2 верный, плавал", "верный", None,
     "На сколько блокируют аккаунт за повторное нарушение правил об источниках трафика?",
     "За повторное нарушение правил об источниках трафика аккаунт блокируется на 30 дней"),
    ("3 неполный (5% без 2%)", "неполный", None,
     "Какую комиссию заплатит физическое лицо за досрочную выплату на банковскую карту?",
     "Физическое лицо заплатит комиссию в размере 5 процентов"),
    ("4 неверный (2000 руб.)", "неверный", None,
     "Что будет с остатком 2000 рублей на неактивном аккаунте?",
     "Если остаток на неактивном аккаунте составляет 2000 рублей"),
    ("5 неверный уверенный (1800)", "неверный", EXTRA_RUN,
     "Какая ставка была у оффера «Школа кода» до изменения целевого действия?",
     "Ставка у оффера «Школа кода» до изменения целевого действия была 1800"),
]


async def main():
    path, repeats, parallel = common.parse_args(default_repeats=20)
    common.prepare()

    main_run = common.load_run(path)
    extra_path = Path(path).parent / EXTRA_RUN
    rows = []
    for label, kind, run_name, question, prefix in ITEMS:
        if run_name is None:
            run = main_run
        elif extra_path.exists():
            run = common.load_run(str(extra_path))
        else:
            print(f"[!] нет файла {extra_path} - пункт «{label}» пропущен")
            continue
        q = common.find_question(run, question)
        if not q["response"].startswith(prefix):
            sys.exit(f"Ответ в прогоне не тот, что ожидался для «{label}»: {q['response'][:80]!r}")
        rows.append({"label": label, "kind": kind, "question": question, "response": q["response"],
                     "contexts": q["contexts"], "scores": []})

    judge = common.Judge(parallel)
    try:
        for row in rows:
            tasks = [judge.score(row["question"], row["response"], row["contexts"], tag=row["label"])
                     for _ in range(repeats)]
            row["scores"] = await asyncio.gather(*tasks)
            print(f"{row['label']:30s} {common.distribution(row['scores'])}")
            if judge.quota_hit:
                break
    finally:
        await judge.close()

    out = common.save("opyt-shum.json", {
        "source_run": path, "repeats": repeats, "judge": common.JUDGE_MODEL,
        "ascore_calls": judge.calls, "errors": judge.errors, "interrupted_by_quota": judge.quota_hit,
        "rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows]})

    print(f"\nВызовов ascore: {judge.calls}, сбоев: {judge.errors}")
    print("Сводка по ответам (одни и те же ответ и куски, судья вызван заново каждый раз):")
    for row in rows:
        vals = [v for v in row["scores"] if v is not None]
        if len(vals) < 2:
            continue
        print(f"  {row['label']:30s} среднее {statistics.mean(vals):.3f}  min {min(vals):.2f}  max {max(vals):.2f}  "
              f"размах {max(vals) - min(vals):.2f}  sd {statistics.pstdev(vals):.3f}  "
              f"разных значений {len(set(vals))}  оценок {len(vals)}")
    sds = [statistics.pstdev([v for v in r["scores"] if v is not None]) for r in rows
           if len([v for v in r["scores"] if v is not None]) > 1]
    if sds:
        print(f"Среднее sd по ответам: {statistics.mean(sds):.3f}; ответов с размахом больше 0: "
              f"{sum(1 for r in rows if common.spread(r['scores']) > 0)} из {len(rows)}")
    print(f"Сохранено: {out}")
    if judge.quota_hit:
        print("Опыт прерван: кончилась квота. В JSON только посчитанное.")


if __name__ == "__main__":
    asyncio.run(main())
