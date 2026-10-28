"""Tách tập train thành dev (lặp hằng ngày) và holdout (chỉ dùng ở mốc quyết định).

Chạy: python -m s2b_rag.eval.split

- Phân tầng theo `type`: mỗi type góp vào holdout cùng một tỉ lệ (làm tròn), để holdout
  không bị thiếu hẳn một kiểu câu hỏi.
- Trong mỗi type, chọn cách đều theo danh sách đã sắp theo `club` (systematic sampling),
  nên tỉ lệ CLB trong holdout cũng xấp xỉ train.
- Sinh thêm bản bỏ dấu của mỗi tập (`*_noaccent.jsonl`) để mô phỏng câu hỏi kiểu public.
- Seed cố định: chạy lại luôn ra đúng cách chia cũ.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from neorag.data import DATA_DIR, read_jsonl, write_jsonl
from s2b_rag.textnorm import strip_accents

SEED = 2026
N_HOLDOUT = 40  # ≈ cỡ một bộ public/private (42 câu)
TRAIN_PATH = DATA_DIR / "train" / "questions.jsonl"
OUT_DIR = DATA_DIR.parent / "experiments" / "splits"


def stratified_split(
    rows: list[dict], n_holdout: int, seed: int = SEED
) -> tuple[list[dict], list[dict]]:
    """Trả về (dev, holdout), giữ nguyên thứ tự gốc trong mỗi tập."""
    rng = random.Random(seed)
    frac = n_holdout / len(rows)
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_type[row["type"]].append(i)

    holdout_idx: set[int] = set()
    for qtype in sorted(by_type):  # sắp để thứ tự gọi rng không phụ thuộc thứ tự file
        idx = by_type[qtype]
        k = int(len(idx) * frac + 0.5)  # làm tròn nửa lên (round() của Python làm tròn về chẵn)
        keys = {i: rng.random() for i in idx}
        ordered = sorted(idx, key=lambda i: (rows[i]["club"], keys[i]))
        holdout_idx.update(ordered[int((j + 0.5) * len(ordered) / k)] for j in range(k))

    dev = [r for i, r in enumerate(rows) if i not in holdout_idx]
    holdout = [r for i, r in enumerate(rows) if i in holdout_idx]
    return dev, holdout


def without_accents(rows: list[dict]) -> list[dict]:
    """Bản sao các câu hỏi với `query` đã bỏ dấu; ground truth giữ nguyên."""
    return [{**r, "query": strip_accents(r["query"])} for r in rows]


def _table(title: str, dev: Counter, holdout: Counter) -> str:
    keys = sorted(set(dev) | set(holdout), key=lambda k: -(dev[k] + holdout[k]))
    width = max(len(title), *(len(str(k)) for k in keys))
    lines = [f"{title:<{width}}  {'dev':>4}  {'holdout':>7}  {'%holdout':>8}"]
    for k in keys:
        total = dev[k] + holdout[k]
        lines.append(f"{k!s:<{width}}  {dev[k]:>4}  {holdout[k]:>7}  {holdout[k] / total:>8.0%}")
    lines.append(f"{'TỔNG':<{width}}  {sum(dev.values()):>4}  {sum(holdout.values()):>7}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--n-holdout", type=int, default=N_HOLDOUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    rows = read_jsonl(args.train)  # đọc thô để giữ trường `type` (neorag.data.Query bỏ trường này)
    dev, holdout = stratified_split(rows, args.n_holdout, args.seed)

    files = {
        "dev": dev,
        "holdout": holdout,
        "dev_noaccent": without_accents(dev),
        "holdout_noaccent": without_accents(holdout),
    }
    for name, data in files.items():
        write_jsonl(args.out / f"{name}.jsonl", data)
        print(f"Đã ghi {args.out / f'{name}.jsonl'} ({len(data)} câu)")

    for field in ("type", "club", "difficulty", "answerable"):
        print()
        print(_table(field, Counter(r[field] for r in dev), Counter(r[field] for r in holdout)))


if __name__ == "__main__":
    main()
