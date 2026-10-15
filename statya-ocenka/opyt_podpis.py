"""
Опыт 2 для статьи 2 (rag-evaluation): изолировать подпись источника. Сколько Faithfulness отнимает подпись
в конце ответа и сколько - подпись с искажённым именем файла (как писала модель по первой версии промпта).

Берётся готовый прогон (ответы и куски уже есть, модель ответа не вызывается). Для каждого вопроса по базе,
на который ассистент ответил (не отказ), судье RAGAS отдаются три варианта при тех же кусках:
  A - ответ как есть, без подписи;
  B - тот же ответ с правильной подписью «(файл: имя)»; если файлов несколько - «(файл: a, файл: b)», как в логах 12.10;
  D - тот же ответ с искажённым именем файла.
Каждый вариант оценивается REPEATS раз (по умолчанию 10).

Откуда искажения. В логах прогонов 12.10 (03:01 и 03:05, первая версия промпта) найдены три реальных искажённых подписи:
  «(выplaty-i-hold.pdf)» дважды, «(глоссариy.txt)» один раз - начало имени кириллицей, конец латиницей.
Они взяты как есть и применяются к файлам vyplaty-i-hold.pdf и glossariy.txt (поле real=True в JSON).
Для остальных файлов реальных искажений в логах нет: имя искажается по образцу «выplaty» - первые две буквы
заменяются кириллическими (offery.xlsx -> офfery.xlsx). Это придумано, не взято из логов (real=False);
в сводке такие строки считаются отдельно.
Подписи в тех двух прогонах бывали и без слова «файл» («(выplaty-i-hold.pdf)», «(faq.html)»); здесь везде «файл:».

Источник ответов: progony/2026-10-14-215644-rules-real-topk8-ragas.json (тот же прогон, что в прошлом опыте, 25 ответов).
Результат - statya-ocenka/opyty/opyt-podpis.json и сводка на экран.

Запуск из папки проекта (PowerShell):
  python statya-ocenka/opyt_podpis.py statya-ocenka/progony/2026-10-14-215644-rules-real-topk8-ragas.json
Необязательно: --repeats 10  --parallel 4
"""
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sudya_obshchee as common

# Реальные искажения из логов 12.10 (прогоны 030100 и 030506)
REAL_DISTORTIONS = {
    "vyplaty-i-hold.pdf": "выplaty-i-hold.pdf",
    "glossariy.txt": "глоссариy.txt",
}
# Для остальных файлов - по образцу «vy» -> «вы»: две первые буквы кириллицей
TRANSLIT = dict(zip("abvgdeziklmnoprstufhcyj", "абвгдезиклмнопрстуфхцыи"))
TRANSLIT["j"] = "й"


def distort(name: str):
    """Искажённое имя файла и признак «взято из логов»."""
    if name in REAL_DISTORTIONS:
        return REAL_DISTORTIONS[name], True
    head = "".join(TRANSLIT.get(ch, ch) for ch in name[:2])
    return head + name[2:], False


async def main():
    path, repeats, parallel = common.parse_args(default_repeats=10)
    common.prepare()

    run = common.load_run(path)
    items = [q for q in run["questions"] if q.get("expected_source") and not q["is_refusal"] and q["contexts"]]
    rows = []
    for q in items:
        files = q["cited_sources"] or [q["expected_source"]]
        bad = [distort(f) for f in files]
        rows.append({
            "question": q["user_input"], "contexts": q["contexts"], "files": files,
            "real": all(flag for _, flag in bad),
            "variants": {
                "A": q["response"],
                "B": f"{q['response']} (" + ", ".join(f"файл: {f}" for f in files) + ")",
                "D": f"{q['response']} (" + ", ".join(f"файл: {name}" for name, _ in bad) + ")",
            },
            "scores": {}})

    judge = common.Judge(parallel)
    try:
        for n, row in enumerate(rows, 1):
            names = list(row["variants"])
            tasks = [judge.score(row["question"], row["variants"][name], row["contexts"], tag=f"{n} {name}")
                     for name in names for _ in range(repeats)]
            results = await asyncio.gather(*tasks)
            for k, name in enumerate(names):
                row["scores"][name] = results[k * repeats:(k + 1) * repeats]
            print(f"{n}/{len(rows)} " + "  ".join(
                f"{name}={statistics.mean([v for v in row['scores'][name] if v is not None] or [float('nan')]):.2f}"
                for name in names) + f"  | {row['question'][:50]}")
            if judge.quota_hit:
                break
    finally:
        await judge.close()

    out = common.save("opyt-podpis.json", {
        "source_run": path, "repeats": repeats, "judge": common.JUDGE_MODEL,
        "ascore_calls": judge.calls, "errors": judge.errors, "interrupted_by_quota": judge.quota_hit,
        "rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows]})

    done = [r for r in rows if len(r["scores"]) == 3]
    print(f"\nВызовов ascore: {judge.calls}, сбоев: {judge.errors}, вопросов с полным набором: {len(done)} из {len(rows)}")

    def mean_of(row, name):
        v = [x for x in row["scores"][name] if x is not None]
        return statistics.mean(v) if v else None

    print("Сводка (все оценки вместе):")
    for name, label in (("A", "A без подписи"), ("B", "B верная подпись"), ("D", "D искажённая")):
        print(f"  {label:18s}: {common.describe([v for r in done for v in r['scores'][name]])}")
    print("Парные разности по вопросам (среднее по 10 оценкам варианта минус среднее A):")
    for name, label in (("B", "верная подпись"), ("D", "искажённая")):
        diffs = [mean_of(r, name) - mean_of(r, "A") for r in done if mean_of(r, name) is not None and mean_of(r, "A") is not None]
        if diffs:
            print(f"  {label:15s}: средняя разность {statistics.mean(diffs):+.3f}; вопросов, где ниже A больше чем на 0.1: "
                  f"{sum(1 for d in diffs if d < -0.1)}; выше A больше чем на 0.1: {sum(1 for d in diffs if d > 0.1)}; из {len(diffs)}")
    real = [r for r in done if r["real"]]
    print(f"Строки с реальными искажениями из логов (файлы vyplaty-i-hold.pdf / glossariy.txt): {len(real)} вопросов")
    for r in real:
        cells = " ".join(f"{n}={'-' if mean_of(r, n) is None else f'{mean_of(r, n):.2f}'}" for n in "ABD")
        print(f"  {cells} | {r['question'][:50]}")
    print(f"Сохранено: {out}")
    if judge.quota_hit:
        print("Опыт прерван: кончилась квота. В JSON только посчитанное.")


if __name__ == "__main__":
    asyncio.run(main())
