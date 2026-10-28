"""Chuẩn hoá văn bản tiếng Việt cho truy xuất (bổ sung cho neorag.text của BTC)."""

from __future__ import annotations

import unicodedata

# "đ" không phải chữ "d" + dấu kết hợp nên NFD không tách được, phải thay riêng.
_D_MAP = str.maketrans({"đ": "d", "Đ": "D"})


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt, giữ nguyên chữ hoa/thường và ký tự khác: "Lập Trình" -> "Lap Trinh"."""
    decomposed = unicodedata.normalize("NFD", text.translate(_D_MAP))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)
