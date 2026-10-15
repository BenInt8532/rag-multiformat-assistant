"""
Опыт 1 для статьи 2 (rag-evaluation): парный опыт с отказами. Что решает: формулировка отказа или его правота.

Берётся готовый прогон (куски и вопросы уже есть, модель ответа не вызывается). Куски у каждого вопроса
зафиксированы из этого прогона, меняется только текст отказа, который получает судья RAGAS (Faithfulness):
  U  - единая фраза NO_CONTEXT_ANSWER из prompts.py («В документах нет информации по этому вопросу.»);
  O1 - «своя» формулировка модели, дословно из прогона 2026-10-12-020759 (первая версия промпта, фразу не задавал);
  O2 - «своя» формулировка, дословно из прогона 2026-10-12-021540 (если совпала с O1 - вариант не дублируется).
Вопросы - те же 7, что в Ошибке 2 статьи (набор подбора из 24 вопросов):
  3 вне базы (верный отказ): бонус для новых вебмастеров, оффер «Ипотека Дом», холд вертикали «нутра»;
  4 по базе (ложный отказ, в тех прогонах модель отказала сама): лимит лидов «Курс аналитика»,
  условия отключения кампании в tracker_ru.py, MIN_VISITS и STOP_MARGIN в tracker_en.py.
(Вопрос «Какая столица Франции?» не берётся: куски не находились, судье он не уходил.)
Каждый вариант оценивается REPEATS раз (по умолчанию 10). Если правота ни при чём, то при одной формулировке
верные и ложные отказы получат одинаковые оценки; если дело в формулировке, то U отличается от O1/O2 при той же правоте.

Источник кусков: progony/2026-10-12-020759-uniform-topk3-ragas.json (в нём есть все 7 вопросов; в прогонах
нарезки rules 4 из 7 вопросов отвечены, а не отказаны).
Результат - statya-ocenka/opyty/opyt-otkazy.json и сводка на экран.

Запуск из папки проекта (PowerShell):
  python statya-ocenka/opyt_otkazy.py statya-ocenka/progony/2026-10-12-020759-uniform-topk3-ragas.json
Необязательно: --repeats 10  --parallel 4
"""
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sudya_obshchee as common

# (группа, вопрос, O1, O2). O1 и O2 скопированы из логов дословно.
ITEMS = [
    ("вне базы", "Есть ли бонус для новых вебмастеров?",
     "В предоставленном контексте нет информации о бонусах для новых вебмастеров.",
     "В предоставленном контексте нет информации о бонусах для новых вебмастеров."),
    ("вне базы", "Какая ставка у оффера «Ипотека Дом»?",
     "В контексте нет информации о оффере «Ипотека Дом».",
     "В предоставленном контексте нет информации об оффере «Ипотека Дом»."),
    ("вне базы", "Какой срок холда у офферов вертикали «нутра»?",
     "В предоставленном контексте нет информации о сроке холда у офферов вертикали «нутра».",
     "В предоставленном контексте нет информации о сроке холда для офферов вертикали «нутра»."),
    ("по базе", "Какой лимит лидов в день у оффера «Курс аналитика»?",
     "В предоставленном контексте нет информации о лимите лидов в день у оффера «Курс аналитика».",
     "В предоставленном контексте нет информации о лимите лидов в день для оффера «Курс аналитика»."),
    ("по базе", "При каких условиях скрипт трекера предлагает отключить кампанию?",
     "В предоставленном контексте нет информации о том, при каких условиях скрипт трекера предлагает отключить кампанию.",
     "В предоставленном контексте нет информации о том, при каких условиях скрипт трекера предлагает отключить кампанию."),
    ("по базе", "Сколько визитов должна набрать группа в скрипте tracker_en.py, чтобы её проверили на паузу?",
     "В контексте нет информации о том, сколько визитов должна набрать группа в скрипте tracker_en.py, чтобы её проверили на паузу.",
     "В контексте нет информации о количестве визитов, которые должна набрать группа в скрипте tracker_en.py, чтобы её проверили на паузу."),
    ("по базе", "При какой марже скрипт tracker_en.py ставит группу на паузу?",
     "В предоставленном контексте нет информации о марже, при которой скрипт tracker_en.py ставит группу на паузу.",
     "В предоставленном контексте нет информации о марже, при которой скрипт tracker_en.py ставит группу на паузу."),
]


async def main():
    path, repeats, parallel = common.parse_args(default_repeats=10)
    common.prepare()
    from prompts import NO_CONTEXT_ANSWER

    run = common.load_run(path)
    rows = []
    for group, question, own1, own2 in ITEMS:
        q = common.find_question(run, question)
        if not q["contexts"]:
            sys.exit(f"У вопроса нет кусков в прогоне: {question}")
        variants = {"U": NO_CONTEXT_ANSWER, "O1": own1}
        if own2 != own1:
            variants["O2"] = own2
        rows.append({"group": group, "question": question, "contexts": q["contexts"],
                     "variants": variants, "scores": {}})

    judge = common.Judge(parallel)
    try:
        async def one(row, name, text):
            return await judge.score(row["question"], text, row["contexts"], tag=f"{row['question'][:30]} {name}")

        for n, row in enumerate(rows, 1):
            names = list(row["variants"])
            tasks = [one(row, name, row["variants"][name]) for name in names for _ in range(repeats)]
            results = await asyncio.gather(*tasks)
            for k, name in enumerate(names):
                row["scores"][name] = results[k * repeats:(k + 1) * repeats]
            print(f"{n}/{len(rows)} [{row['group']}] " +
                  "  ".join(f"{name}={statistics.mean([v for v in row['scores'][name] if v is not None] or [float('nan')]):.2f}"
                            for name in names) + f"  | {row['question'][:50]}")
            if judge.quota_hit:
                break
    finally:
        await judge.close()

    out = common.save("opyt-otkazy.json", {
        "source_run": path, "repeats": repeats, "judge": common.JUDGE_MODEL,
        "ascore_calls": judge.calls, "errors": judge.errors, "interrupted_by_quota": judge.quota_hit,
        "rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows]})

    print(f"\nВызовов ascore: {judge.calls}, сбоев: {judge.errors}")
    print("Сводка по группам (все оценки группы вместе):")
    for group in ("вне базы", "по базе"):
        sel = [r for r in rows if r["group"] == group and r["scores"]]
        for label, names in (("единая фраза U", ["U"]), ("свои формулировки O1/O2", ["O1", "O2"])):
            values = [v for r in sel for name in names for v in r["scores"].get(name, [])]
            print(f"  {group:9s} {label:24s}: {common.describe(values)}")
    print("Устойчивость: в скольких парах «вопрос+вариант» 10 оценок не совпали между собой:")
    unstable = sum(1 for r in rows for v in r["scores"].values() if len({x for x in v if x is not None}) > 1)
    total = sum(len(r["scores"]) for r in rows)
    print(f"  {unstable} из {total}")
    print(f"Сохранено: {out}")
    if judge.quota_hit:
        print("Опыт прерван: кончилась квота. В JSON только посчитанное.")


if __name__ == "__main__":
    asyncio.run(main())
