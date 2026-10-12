import csv
from collections import defaultdict

MIN_VISITS = 150

STOP_MARGIN = -35.0

STATE_PAID = "paid"
STATE_DECLINED = "declined"
STATE_PENDING = "pending"


def read_visits(path):
    groups = defaultdict(lambda: {"visits": 0, "spend": 0.0})

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            group = groups[row["sub2"]]
            group["visits"] += 1
            group["spend"] += float(row["spend"])

    return groups


def read_orders(path):
    groups = defaultdict(lambda: {"orders": 0, "paid": 0, "income": 0.0})

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            group = groups[row["sub2"]]
            group["orders"] += 1
            if row["state"] == STATE_PAID:
                group["paid"] += 1
                group["income"] += float(row["amount"])

    return groups


def order_rate(orders, visits):
    if visits == 0:
        return 0.0
    return orders / visits * 100


def paid_share(paid, orders):
    if orders == 0:
        return 0.0
    return paid / orders * 100


def income_per_visit(income, visits):
    if visits == 0:
        return 0.0
    return income / visits


def margin(income, spend):
    if spend == 0:
        return 0.0
    return (income - spend) / spend * 100


def build_summary(visits, orders):
    summary = []

    for name, visit_stats in visits.items():
        order_stats = orders.get(name, {"orders": 0, "paid": 0, "income": 0.0})
        summary.append({
            "group": name,
            "visits": visit_stats["visits"],
            "spend": visit_stats["spend"],
            "orders": order_stats["orders"],
            "income": order_stats["income"],
            "order_rate": order_rate(order_stats["orders"], visit_stats["visits"]),
            "paid_share": paid_share(order_stats["paid"], order_stats["orders"]),
            "income_per_visit": income_per_visit(order_stats["income"], visit_stats["visits"]),
            "margin": margin(order_stats["income"], visit_stats["spend"])
        })

    return sorted(summary, key=lambda row: row["margin"])


def find_groups_to_pause(summary):
    return [
        row["group"] for row in summary
        if row["visits"] >= MIN_VISITS and row["margin"] < STOP_MARGIN
    ]


def print_summary(summary):
    print(f"{'Group':<20} {'Visits':>7} {'Spend':>10} {'Rate':>7} {'Paid':>9} {'IPV':>7} {'Margin':>8}")
    for row in summary:
        print(
            f"{row['group']:<20} {row['visits']:>7} {row['spend']:>10.2f} {row['order_rate']:>7.2f} "
            f"{row['paid_share']:>9.2f} {row['income_per_visit']:>7.2f} {row['margin']:>8.2f}"
        )


if __name__ == "__main__":
    group_summary = build_summary(read_visits("visits.csv"), read_orders("orders.csv"))
    print_summary(group_summary)

    to_pause = find_groups_to_pause(group_summary)
    if to_pause:
        print("\nPause: " + ", ".join(to_pause))
