"""
Подбор правила нарезки для одного файла базы.

Файл по очереди режется каждым из переданных правил, ассистент отвечает на вопросы
из eval/questions.json, ответы оценивает RAGAS. Файл chunking.txt не меняется: правила передаются в команде.

Два режима:
  --full-base   в базе все файлы из chunking.txt, меняется только правило выбранного файла.
                Его чанки соревнуются с чанками других документов, как в настоящей работе.
                Задаются все вопросы; судья оценивает вопросы к выбранному файлу и те чужие вопросы,
                в контекст которых попали его чанки. Попадание и отказы считаются по всем вопросам.
  без флага     в базе один выбранный файл, вопросы только к нему. Проверка лёгкая:
                поиску не с чем перепутать, поэтому близкие правила она не различает.

Тратит деньги на счёте OpenAI (по оценке, центы).

Запуск из папки проекта:
    python tests/tune_file.py --full-base --repeats 2 vyplaty-i-hold.pdf "paragraphs, size=800, overlap=150" "sections, size=900"

Результат печатается и сохраняется в results/<дата>-tune-<файл>.json.
После подбора обычный запуск ассистента соберёт базу по chunking.txt заново (векторы возьмутся из кеша).
"""

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import evaluate_ragas as ev  # noqa: E402
from assistant import Assistant  # noqa: E402
from chunking import chunk_document, find_rule  # noqa: E402
from config import DATA_DIR, RESULTS_DIR, _parse_rule  # noqa: E402
from loaders import load_document  # noqa: E402

METRICS = ("faithfulness", "context_precision", "answer_relevancy")


def chunk_lengths(file_name: str, rules) -> list:
    """Длины чанков файла при данном правиле (без обращения к OpenAI)."""
    document = load_document(DATA_DIR / file_name, clean=ev.settings["clean_text"])
    rule = find_rule(rules, file_name, uniform=False)
    return [len(chunk["text"]) for chunk in chunk_document(document, rule, uniform=False)]


def mean(rows, name) -> float:
    """Среднее метрики по строкам вида {'score': {...}}; пропуски не учитываются."""
    return ev.average([row["score"][name] for row in rows])


def cell(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Подбор правила нарезки для одного файла")
    parser.add_argument("file", help="имя файла из папки data")
    parser.add_argument("variants", nargs="+", help="правила в кавычках, как в chunking.txt после двоеточия")
    parser.add_argument("--repeats", type=int, default=1, help="сколько раз прогнать каждое правило")
    parser.add_argument("--full-base", action="store_true", help="оставить в базе все файлы из chunking.txt")
    args = parser.parse_args()

    if not (DATA_DIR / args.file).exists():
        sys.exit(f"Нет файла в папке data: {args.file}")

    all_questions = ev.load_questions()
    if args.full_base:
        active = set(ev.rules["files"]) | {args.file}
        questions = [item for item in all_questions if not item.get("source") or item["source"] in active]
    else:
        questions = [item for item in all_questions if item.get("source") == args.file]
    own_count = sum(1 for item in questions if item.get("source") == args.file)
    if not own_count:
        sys.exit(f"В файле вопросов нет вопросов к {args.file}")

    print("=" * 70)
    print(f"ПОДБОР НАРЕЗКИ: {args.file}")
    print("=" * 70)
    print(f"Python {sys.version.split()[0]}, база: {'все файлы' if args.full_base else 'один файл'}, "
          f"вопросов: {len(questions)} (к этому файлу: {own_count}), повторов: {args.repeats}")
    print(f"Настройки: {ev.settings}\n")

    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "file": args.file,
              "full_base": args.full_base, "settings": ev.settings, "repeats": args.repeats, "variants": []}
    summary = []

    for variant in args.variants:
        files = dict(ev.rules["files"]) if args.full_base else {}
        files[args.file] = _parse_rule(variant, f"правило '{variant}'")
        rules = {"default": ev.rules["default"], "uniform": ev.rules["uniform"], "files": files}
        lengths = chunk_lengths(args.file, rules)
        print("-" * 70)
        print(f"ПРАВИЛО: {variant}")
        print(f"Чанков файла {len(lengths)}, длина в символах: наименьшая {min(lengths)}, "
              f"средняя {sum(lengths) // len(lengths)}, наибольшая {max(lengths)}\n")

        rows = []  # все ответы всех повторов этого правила
        for repeat in range(1, args.repeats + 1):
            assistant = Assistant(ev.settings, rules, log=lambda message: None)
            samples = ev.collect_samples(assistant, questions)
            # Судье идут вопросы к файлу и чужие вопросы, куда попали его чанки
            judged = [sample for sample in samples
                      if sample["expected_source"] == args.file or args.file in sample["context_sources"]]
            scores = asyncio.run(ev.score_samples(judged))
            by_question = {sample["user_input"]: score for sample, score in zip(judged, scores)}

            print(f"\n  Повтор {repeat}:")
            for sample in samples:
                score = by_question.get(sample["user_input"], {name: float("nan") for name in METRICS})
                rows.append({"repeat": repeat, "sample": sample, "score": score,
                             "own": sample["expected_source"] == args.file,
                             "judged": sample["user_input"] in by_question})
                if sample["user_input"] not in by_question:
                    continue
                kind = "к файлу" if sample["expected_source"] == args.file else \
                    f"чужой ({sample['expected_source'] or 'вне базы'})"
                print(f"  [{kind}] {sample['user_input']}")
                print(f"     Ответ: {sample['response']}")
                print(f"     Эталон: {sample['reference']}")
                print(f"     Чанки: {sample['context_sources']}, схожесть {sample['similarities']}, "
                      f"символов в контексте {sum(len(text) for text in sample['retrieved_contexts'])}")
                print(f"     Модель сослалась на: {', '.join(sample['cited_sources']) or 'нет ссылки'}")
                print(f"     Faithfulness / Context Precision / Answer Relevancy: "
                      f"{' / '.join(cell(score[name]) for name in METRICS)}")
            print()

        own = [row for row in rows if row["own"]]
        others = [row for row in rows if row["judged"] and not row["own"]]
        in_base = [row for row in rows if row["sample"]["expected_source"]]
        row = {
            "variant": variant,
            "chunks": len(lengths),
            "average_length": sum(lengths) // len(lengths),
            "own_answers": len(own),
            "own_hits": sum(1 for r in own if r["sample"]["hit_rank"] is not None),
            "own_first": sum(1 for r in own if r["sample"]["hit_rank"] == 1),
            "own_refusals": sum(1 for r in own if r["sample"]["is_refusal"]),
            "own_context_chars": sum(sum(len(t) for t in r["sample"]["retrieved_contexts"]) for r in own) // len(own),
            "own": {name: mean(own, name) for name in METRICS},
            "other_answers": len(others),
            "other": {name: mean(others, name) for name in METRICS},
            "base_answers": len(in_base),
            "base_hits": sum(1 for r in in_base if r["sample"]["hit_rank"] is not None),
            "base_missed_refusals": sum(1 for r in in_base if r["sample"]["is_refusal"]),
        }
        summary.append(row)

        def clean(values):
            return {name: (None if math.isnan(value) else round(value, 4)) for name, value in values.items()}

        report["variants"].append({
            **row, "own": clean(row["own"]), "other": clean(row["other"]),
            "answers": [{"repeat": r["repeat"], "own": r["own"], "judged": r["judged"],
                         **{key: value for key, value in r["sample"].items() if key != "retrieved_contexts"},
                         "contexts": r["sample"]["retrieved_contexts"], "scores": clean(r["score"])}
                        for r in rows]
        })

    print("=" * 70)
    print(f"СВОДКА: {args.file}, база: {'все файлы' if args.full_base else 'один файл'}, повторов: {args.repeats}")
    print("=" * 70)
    print("\nВопросы к файлу:")
    print(f"{'правило':<36} {'чанков':>6} {'ср.дл.':>6} {'попад.':>7} {'первым':>6} {'конт.':>6} "
          f"{'Faith.':>7} {'Prec.':>7} {'Relev.':>7} {'отказов':>7}")
    for row in summary:
        print(f"{row['variant'][:36]:<36} {row['chunks']:>6} {row['average_length']:>6} "
              f"{str(row['own_hits']) + '/' + str(row['own_answers']):>7} {row['own_first']:>6} "
              f"{row['own_context_chars']:>6} {cell(row['own']['faithfulness']):>7} "
              f"{cell(row['own']['context_precision']):>7} {cell(row['own']['answer_relevancy']):>7} "
              f"{row['own_refusals']:>7}")

    if args.full_base:
        print("\nЧужие вопросы, в контекст которых попали чанки файла, и вся база:")
        print(f"{'правило':<36} {'чужих':>6} {'Faith.':>7} {'Prec.':>7} {'Relev.':>7} "
              f"{'попадание по базе':>18} {'отказов-промахов':>17}")
        for row in summary:
            print(f"{row['variant'][:36]:<36} {row['other_answers']:>6} {cell(row['other']['faithfulness']):>7} "
                  f"{cell(row['other']['context_precision']):>7} {cell(row['other']['answer_relevancy']):>7} "
                  f"{str(row['base_hits']) + '/' + str(row['base_answers']):>18} {row['base_missed_refusals']:>17}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y-%m-%d-%H%M%S}-tune-{args.file}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nГотово. Результаты сохранены: {path}")


if __name__ == "__main__":
    main()
