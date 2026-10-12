"""
Мини-трекер арбитражника.

Читает выгрузку кликов и выгрузку лидов из партнёрской сети, сводит их по метке sub1
(название рекламной кампании) и считает по каждой кампании конверсию, апрув, EPC и ROI.
В конце показывает, какие кампании пора отключать.
"""

import csv
from collections import defaultdict

# Кампания проверяется на убыточность только после этого числа кликов:
# на меньшем объёме показатели слишком случайны
MIN_CLICKS = 100

# Кампания считается убыточной, если её ROI ниже этого значения в процентах
STOP_ROI = -20.0

# Статусы лида в выгрузке сети
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_HOLD = "hold"


def load_clicks(path):
    """
    Чтение выгрузки кликов.

    В файле три столбца: идентификатор клика, метка кампании sub1 и цена клика в рублях.
    Возвращает словарь: метка кампании -> число кликов и сумма расхода.
    """
    campaigns = defaultdict(lambda: {"clicks": 0, "cost": 0.0})

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            campaign = campaigns[row["sub1"]]
            campaign["clicks"] += 1
            campaign["cost"] += float(row["cost"])

    return campaigns


def load_leads(path):
    """
    Чтение выгрузки лидов из партнёрской сети.

    Лид в статусе hold ещё проверяется рекламодателем: в доход он не идёт,
    но учитывается в общем числе лидов. Отклонённый лид дохода не приносит.
    Возвращает словарь: метка кампании -> число лидов, число подтверждённых и доход.
    """
    campaigns = defaultdict(lambda: {"leads": 0, "approved": 0, "revenue": 0.0})

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            campaign = campaigns[row["sub1"]]
            campaign["leads"] += 1
            if row["status"] == STATUS_APPROVED:
                campaign["approved"] += 1
                campaign["revenue"] += float(row["payout"])

    return campaigns


# Конверсия показывает, какая доля кликов превратилась в заявки
def calc_cr(leads, clicks):
    """Конверсия в процентах: лиды, делённые на клики. Без кликов конверсия равна нулю."""
    if clicks == 0:
        return 0.0
    return leads / clicks * 100


# Апрув показывает, какую долю заявок рекламодатель подтвердил и оплатил
def calc_approve_rate(approved, leads):
    """Апрув в процентах: подтверждённые лиды, делённые на все лиды."""
    if leads == 0:
        return 0.0
    return approved / leads * 100


# По EPC сравнивают кампании между собой: сколько рублей приносит один клик
def calc_epc(revenue, clicks):
    """Средний доход с клика в рублях: подтверждённый доход, делённый на число кликов."""
    if clicks == 0:
        return 0.0
    return revenue / clicks


# ROI отвечает на главный вопрос: окупается ли реклама
def calc_roi(revenue, cost):
    """
    Окупаемость в процентах: прибыль, делённая на расход.

    Прибыль - это доход минус расход. При нулевом расходе ROI не определён,
    функция возвращает ноль, чтобы отчёт не падал на бесплатном трафике.
    """
    if cost == 0:
        return 0.0
    return (revenue - cost) / cost * 100


def build_report(clicks, leads):
    """
    Сведение кликов и лидов в отчёт по кампаниям.

    Кампания попадает в отчёт, даже если по ней нет ни одного лида:
    именно такие кампании чаще всего и сливают бюджет.
    Строки отсортированы по ROI, от худшей кампании к лучшей.
    """
    report = []

    for name, click_stats in clicks.items():
        lead_stats = leads.get(name, {"leads": 0, "approved": 0, "revenue": 0.0})
        report.append({
            "campaign": name,
            "clicks": click_stats["clicks"],
            "cost": click_stats["cost"],
            "leads": lead_stats["leads"],
            "revenue": lead_stats["revenue"],
            "cr": calc_cr(lead_stats["leads"], click_stats["clicks"]),
            "approve": calc_approve_rate(lead_stats["approved"], lead_stats["leads"]),
            "epc": calc_epc(lead_stats["revenue"], click_stats["clicks"]),
            "roi": calc_roi(lead_stats["revenue"], click_stats["cost"])
        })

    return sorted(report, key=lambda row: row["roi"])


def find_campaigns_to_stop(report):
    """
    Поиск кампаний, которые пора отключать.

    Правило остановки: кампания набрала не меньше MIN_CLICKS кликов,
    и её ROI ниже STOP_ROI. Кампании с меньшим числом кликов не трогаем,
    даже если они в минусе: данных ещё мало.
    """
    return [
        row["campaign"] for row in report
        if row["clicks"] >= MIN_CLICKS and row["roi"] < STOP_ROI
    ]


def print_report(report):
    """Вывод отчёта в консоль: одна строка на кампанию."""
    print(f"{'Кампания':<20} {'Клики':>7} {'Расход':>10} {'CR, %':>7} {'Апрув, %':>9} {'EPC':>7} {'ROI, %':>8}")
    for row in report:
        print(
            f"{row['campaign']:<20} {row['clicks']:>7} {row['cost']:>10.2f} {row['cr']:>7.2f} "
            f"{row['approve']:>9.2f} {row['epc']:>7.2f} {row['roi']:>8.2f}"
        )


if __name__ == "__main__":
    campaign_report = build_report(load_clicks("clicks.csv"), load_leads("leads.csv"))
    print_report(campaign_report)

    to_stop = find_campaigns_to_stop(campaign_report)
    if to_stop:
        print("\nПора отключать: " + ", ".join(to_stop))
