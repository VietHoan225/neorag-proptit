"""Phân tích câu hỏi trước khi truy xuất.

R8 (tách ý theo luật): câu hỏi nhiều ý ("truong ban su kien proptit bay gio la ai, nk truoc do la ai") cần
nhiều đoạn khác nhau, nhưng tìm bằng cả câu thì thường chỉ bắt được ý mạnh nhất. Tách tại `,` `;` `?` thành
các câu con để tìm riêng từng ý; câu con không nhắc CLB thì mượn tên CLB của cả câu.
→ Đã thử: làm GIẢM điểm vì câu con mất nghĩa ("tên của mấy bài đó"). Không dùng trong pipeline.

R7 (viết lại bằng LLM): một lần gọi claude-haiku-5-5 ngắn trả về câu hỏi CÓ DẤU đầy đủ (để dense và reranker
dùng được, ADR-0004) và các câu con TỰ ĐỦ NGHĨA (thay "đó", "hai bên" bằng đối tượng cụ thể).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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


REWRITE_SYSTEM = """Bạn chuẩn hoá câu hỏi tiếng Việt để tra cứu tài liệu về cộng đồng S2B tại Học viện Công \
nghệ Bưu chính Viễn thông (PTIT), gồm CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC) và CLB Nhà \
Sáng Tạo Game PTIT (GCC).

Trả về đúng một object JSON, không giải thích: {"query": "...", "subqueries": ["...", "..."]}
- "query": câu hỏi viết lại bằng tiếng Việt CÓ DẤU đầy đủ, đúng chính tả; mở rộng viết tắt và teencode \
(ko → không, dc → được, tv → thành viên, nk → nhiệm kỳ, cn → chủ nhiệm, hv → học viện); gọi CLB bằng tên \
đầy đủ. Giữ nguyên ý hỏi, không thêm thông tin, không trả lời câu hỏi.
- "subqueries": nếu câu hỏi hỏi nhiều điều khác nhau, tách thành tối đa 3 câu hỏi con có dấu, mỗi câu TỰ ĐỦ \
NGHĨA: thay các từ thay thế như "đó", "bên đó", "hai bên", "vậy" bằng đối tượng cụ thể và nêu rõ tên CLB. \
Nếu câu chỉ hỏi một điều thì để danh sách rỗng."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Rewrite:
    query: str
    subqueries: list[str] = field(default_factory=list)
    ok: bool = True  # False nếu LLM trả về không đọc được → dùng câu gốc


def parse_rewrite(reply: str, original: str, max_subqueries: int = 3) -> Rewrite:
    """Đọc JSON trong câu trả lời của LLM (chấp nhận bọc ```json); hỏng thì giữ câu gốc."""
    match = _JSON_OBJECT.search(reply)
    try:
        data = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        data = {}
    query = data.get("query") if isinstance(data.get("query"), str) else ""
    if not query.strip():
        return Rewrite(original, ok=False)
    subs = data.get("subqueries") if isinstance(data.get("subqueries"), list) else []
    subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()][:max_subqueries]
    return Rewrite(query.strip(), subs if len(subs) >= 2 else [])  # 1 câu con = không tách


class LLMQueryRewriter:
    """R7: viết lại câu hỏi bằng claude-haiku-5-5 (1 lần gọi ngắn, có cache, không truyền temperature)."""

    def __init__(self, llm, model: str = "claude-haiku-5-5"):
        self.llm, self.model = llm, model

    def rewrite(self, query: str) -> Rewrite:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": query},
        ]
        return parse_rewrite(self.llm.chat(messages, model=self.model), query)


def needs_rewrite(query: str) -> bool:
    """Chỉ gọi LLM khi có lợi đo được: câu không dấu (reranker/dense yếu) hoặc câu nhiều ý."""
    return not has_diacritics(query) or bool(split_subqueries(query))
