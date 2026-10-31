"""Tên gọi khác của các CLB: dùng để làm giàu index (R4) và nhận diện CLB trong câu hỏi (R5).

Nguồn alias (để tránh overfit public test, KHÔNG lấy cách gọi chỉ xuất hiện trong public):
- corpus: tên chính thức, tên tiếng Anh, viết tắt (PROPTIT, Programming PTIT, MPC, Multimedia PTIT Club,
  CLB Sáng tạo Đa phương tiện, Game Creators Club, GCC, S2B);
- câu hỏi train: "clb lt", "clb mm", "clb da phuong tien", "clb game", "clb lam game", "cong dong";
- viết tắt tiếng Việt phổ biến: "dpt" = đa phương tiện.
So khớp trên câu hỏi đã mở rộng viết tắt (R1) và bỏ dấu, theo ranh giới từ.
"""

from __future__ import annotations

import re

from s2b_rag.textnorm import expand_query, strip_accents

# Tên `club` đúng như trong data/corpus.jsonl
PROPTIT = "CLB Lập Trình PTIT"
MULTIMEDIA = "CLB Multimedia PTIT"
GCC = "CLB Nhà Sáng Tạo Game PTIT (GCC)"
S2B = "Cộng đồng S2B"

# Alias dạng bỏ dấu, chữ thường. Không dùng từ đơn lẻ dễ nhầm ("lap trinh", "game", "media"):
# "học lập trình ở GCC" không phải câu hỏi về ProPTIT.
CLUB_ALIASES: dict[str, list[str]] = {
    PROPTIT: ["proptit", "pro ptit", "programming ptit", "clb lap trinh", "lap trinh ptit"],
    MULTIMEDIA: [
        "multimedia", "mpc", "clb mm", "mm ptit", "da phuong tien", "dpt",
    ],
    GCC: [
        "gcc", "nha sang tao game", "game creators", "clb game", "clb lam game",
    ],
    S2B: ["s2b", "cong dong"],
}  # fmt: skip

# Tiền tố làm giàu index (R4): tên chính + alias có dấu để cả dense lẫn BM25 cùng hưởng.
CLUB_LABELS = {
    PROPTIT: "CLB Lập Trình PTIT (ProPTIT, Programming PTIT, CLB lập trình)",
    MULTIMEDIA: "CLB Multimedia PTIT (MPC, Multimedia, CLB đa phương tiện, mm)",
    GCC: "CLB Nhà Sáng Tạo Game PTIT (GCC, Game Creators Club, CLB game, CLB làm game)",
    S2B: "Cộng đồng S2B (cả ba CLB ProPTIT, Multimedia, GCC)",
}


def detect_clubs(query: str) -> list[str]:
    """Các CLB được nhắc tới trong câu hỏi, theo thứ tự trong CLUB_ALIASES."""
    text = f" {strip_accents(expand_query(query))} "
    return [
        club
        for club, aliases in CLUB_ALIASES.items()
        if any(f" {alias} " in text for alias in aliases)
    ]


# "ba CLB", "cả 3 câu lạc bộ" = hỏi về cả ba CLB (câu train 0017, 0044, 0104 đều cần đoạn của cả ba). Không gộp
# "CLB nào trong S2B": train cho thấy mơ hồ (0080 "vẽ + làm game thì vào CLB nào" chỉ cần GCC).
_ALL_THREE = re.compile(r"\b(ba|3) (clb|cau lac bo)\b")


def detect_clubs_multi(query: str) -> list[str]:
    """detect_clubs + câu nói "ba CLB" thì coi như nhắc cả ba CLB (để quota R5 chia top-5 cho từng CLB)."""
    clubs = detect_clubs(query)
    if _ALL_THREE.search(strip_accents(expand_query(query))):
        clubs = list(dict.fromkeys([*clubs, PROPTIT, MULTIMEDIA, GCC]))
    return clubs


def enrich(club: str, topic: str, text: str) -> str:
    """R4: "<CLB + alias> · <chủ đề>. <đoạn văn>" — để đoạn khớp cả khi câu hỏi gọi CLB bằng tên khác."""
    return f"{CLUB_LABELS.get(club, club)} · {topic}. {text}"
