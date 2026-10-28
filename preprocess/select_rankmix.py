import argparse
import math
from collections import Counter, defaultdict

from common import load_json, save_json


def descending_ranks(indices: list[int], rows: list[dict]) -> dict[int, int]:
    order = sorted(indices, key=lambda i: (-rows[i]["siig"], rows[i]["extra_info"]["qid"]))
    return {index: rank for rank, index in enumerate(order, start=1)}


def rankmix(rows: list[dict], percent: float) -> list[int]:
    total = len(rows)
    target = math.floor(total * percent / 100)
    global_rank = descending_ranks(list(range(total)), rows)
    categories = defaultdict(list)
    for index, row in enumerate(rows):
        categories[row["extra_info"]["category"]].append(index)
    category_rank, category_size = {}, {}
    for indices in categories.values():
        category_rank.update(descending_ranks(indices, rows))
        category_size.update({index: len(indices) for index in indices})

    core = {
        i
        for i in range(total)
        if global_rank[i] <= target and category_rank[i] <= math.floor(category_size[i] * percent / 100)
    }

    def mix_score(i):
        global_score = 1.0 - (global_rank[i] - 1) / max(total - 1, 1)
        category_score = 1.0 - (category_rank[i] - 1) / max(category_size[i] - 1, 1)
        return global_score * category_score

    rest = sorted(
        (i for i in range(total) if i not in core),
        key=lambda i: (-mix_score(i), -rows[i]["siig"], rows[i]["extra_info"]["qid"]),
    )
    return sorted(core | set(rest[: target - len(core)]))


def main():
    parser = argparse.ArgumentParser(description="Select visually dependent questions with RankMix.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--percent", type=float, default=20)
    args = parser.parse_args()

    rows = load_json(args.input)
    selected = [rows[i] for i in rankmix(rows, args.percent)]
    save_json(args.output, selected)
    print(f"Selected {len(selected)} / {len(rows)} questions:")
    for category, count in Counter(row["extra_info"]["category"] for row in selected).most_common():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
