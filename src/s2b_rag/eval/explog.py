"""Nhật ký thí nghiệm `experiments/log.csv`: mỗi lần đo một dòng, gắn commit để tái lập."""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path

from neorag.data import DATA_DIR

LOG_PATH = DATA_DIR.parent / "experiments" / "log.csv"
# Đặt sẵn cột cho generation/chi phí để header không phải đổi khi thêm các bước sau.
FIELDS = [
    "time", "commit", "dirty", "system", "split", "n",
    "recall@5", "mrr@5", "ndcg@5", "retrieval",
    "string_presence", "rouge_l", "generation", "score", "cost_usd", "note",
]  # fmt: skip


def git_state() -> tuple[str, bool]:
    """(commit rút gọn, có thay đổi chưa commit hay không).

    "dirty" tính cả file mới chưa track (code mới viết chưa commit), nhưng bỏ qua `experiments/`
    vì chính file log này luôn thay đổi sau mỗi lần đo."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=DATA_DIR.parent, check=False
        ).stdout.strip()

    commit = run("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain", "--", ".", ":(exclude)experiments"))
    return commit, dirty


def append(row: dict, path: Path = LOG_PATH) -> None:
    """Ghi thêm một dòng; tự điền thời gian và commit, tạo header nếu file chưa có."""
    commit, dirty = git_state()
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        **row,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        if is_new:
            writer.writeheader()
        writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
