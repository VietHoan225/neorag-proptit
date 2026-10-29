"""Chuẩn hoá văn bản tiếng Việt cho truy xuất (bổ sung cho neorag.text của BTC)."""

from __future__ import annotations

import re
import unicodedata

from neorag.text import tokenize

# "đ" không phải chữ "d" + dấu kết hợp nên NFD không tách được, phải thay riêng.
_D_MAP = str.maketrans({"đ": "d", "Đ": "D"})

# Viết tắt/teencode -> dạng đầy đủ (R1). Nguồn: khảo sát câu hỏi train (token ngắn không có trong
# corpus: ko, tv, nk, lt, cn, hv, dc, mail) + teencode tiếng Việt phổ biến. Alias tên CLB (mm, mpc,
# dpt, gcc...) KHÔNG để ở đây mà ở bước nhận diện CLB (R5).
ABBREVIATIONS = {
    "ko": "không", "k": "không", "hok": "không", "hk": "không", "kh": "không",
    "dc": "được", "đc": "được", "dk": "đăng ký", "đk": "đăng ký",
    "tv": "thành viên", "tvien": "thành viên", "nk": "nhiệm kỳ",
    "cn": "chủ nhiệm", "pcn": "phó chủ nhiệm", "lt": "lập trình",
    "hv": "học viện", "sv": "sinh viên", "mail": "email",
    "j": "gì", "gi": "gì", "vs": "với", "r": "rồi", "ntn": "như thế nào", "bn": "bao nhiêu",
    "tg": "thời gian", "mn": "mọi người", "nhg": "nhưng",
}  # fmt: skip
_MONTH = re.compile(r"^t(\d{1,2})$")  # "t10" -> "tháng 10"


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt, giữ nguyên chữ hoa/thường và ký tự khác: "Lập Trình" -> "Lap Trinh"."""
    decomposed = unicodedata.normalize("NFD", text.translate(_D_MAP))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


def has_diacritics(text: str) -> bool:
    """True nếu văn bản có ít nhất một chữ mang dấu tiếng Việt (kể cả đ)."""
    return strip_accents(text) != text


def expand_query(query: str) -> str:
    """R1: chữ thường, bỏ dấu câu, mở rộng viết tắt. "tv clb lt ko?" -> "thành viên clb lập trình không"."""
    words = []
    for tok in tokenize(query):
        month = _MONTH.match(tok)
        words.append(f"tháng {month.group(1)}" if month else ABBREVIATIONS.get(tok, tok))
    return " ".join(words)


def ngrams(tokens: list[str], n_max: int) -> list[str]:
    """Âm tiết đơn + cụm 2..n_max âm tiết liền nhau ("thành lập" là một từ 2 âm tiết)."""
    out = list(tokens)
    for n in range(2, n_max + 1):
        out += ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return out


def analyze(text: str, accents: bool = True, n_max: int = 1) -> list[str]:
    """Văn bản -> danh sách term cho BM25 (có dấu hoặc bỏ dấu, kèm n-gram)."""
    tokens = tokenize(text) if accents else tokenize(strip_accents(text))
    return ngrams(tokens, n_max)
