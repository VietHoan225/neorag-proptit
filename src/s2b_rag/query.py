"""Phân tích câu hỏi trước khi truy xuất.

R8 (tách ý theo luật): câu hỏi nhiều ý ("truong ban su kien proptit bay gio la ai, nk truoc do la ai") cần
nhiều đoạn khác nhau, nhưng tìm bằng cả câu thì thường chỉ bắt được ý mạnh nhất. Tách tại `,` `;` `?` thành
các câu con để tìm riêng từng ý; câu con không nhắc CLB thì mượn tên CLB của cả câu.
"""

from __future__ import annotations

import re

from s2b_rag.aliases import detect_clubs
from s2b_rag.textnorm import has_diacritics, strip_accents

_CLAUSE_SEP = re.compile(r"[,;?]")
MIN_WORDS = 3  # câu con quá ngắn ("ạ", "thế nào") không đủ nghĩa để tìm riêng


def split_subqueries(query: str, min_words: int = MIN_WORDS) -> list[str]:
    """Các câu con của một câu hỏi nhiều ý; trả về [] nếu câu chỉ có một ý."""
    clauses = [c.strip() for c in _CLAUSE_SEP.split(query)]
    clauses = [c for c in clauses if len(c.split()) >= min_words]
    if len(clauses) < 2:
        return []
    clubs = detect_clubs(query)
    subqueries = []
    for clause in clauses:
        if clubs and not detect_clubs(clause):
            # gắn ngữ cảnh CLB cho câu con mồ côi, cùng kiểu dấu với câu con: tên có dấu sẽ làm câu không dấu
            # bị coi là "có dấu" → dense đủ trọng số + reranker chạy, đều có hại với câu không dấu (ADR-0003/0004)
            names = " ".join(clubs)
            clause = f"{clause} {names if has_diacritics(clause) else strip_accents(names)}"
        subqueries.append(clause)
    return subqueries
