"""Chatbot: nhận diện câu xã giao ("hello", "cảm ơn", "bạn là ai"...) và tin nhắn không phải câu hỏi ("1 + 1",
"???") bằng luật → trả lời ngay, KHÔNG tìm tài liệu,
KHÔNG gọi haiku. Trước đây "Hello" đi qua cả pipeline (không dấu → haiku thêm dấu → tìm kiếm → haiku trả lời): ~9 s.

Chỉ khớp khi CẢ tin nhắn là câu xã giao (sau khi bỏ dấu, dấu câu và vài từ đệm ở hai đầu như "ạ", "nhé", "bạn ơi"),
để câu như "chào bạn, CLB Lập Trình tuyển khi nào?" vẫn đi qua pipeline. Câu ngoài phạm vi (không phải xã giao) vẫn đi
pipeline và được từ chối dựa trên tài liệu — chatbot không tự trả lời bằng kiến thức riêng của LLM.
Không dùng trong `run()` (bài nộp).
"""

from __future__ import annotations

import re

from s2b_rag.textnorm import strip_accents

_GREET = {"hello", "hi", "hey", "helo", "alo", "xin chao", "chao", "chao xin", "good morning", "yo"}
_THANKS = {
    "cam on",
    "cam on nhieu",
    "thanks",
    "thank you",
    "thank",
    "tks",
    "thanks you",
    "ok cam on",
    "oke cam on",
}
_WHO = {
    "ban la ai", "ban la gi", "ban la bot gi", "bot la ai", "ban ten gi", "ten ban la gi", "gioi thieu ve ban",
    "ban lam duoc gi", "ban giup duoc gi", "ban co the lam gi", "ban biet gi", "ban tra loi duoc gi",
}  # fmt: skip
_BYE = {"bye", "bye bye", "goodbye", "tam biet", "hen gap lai", "pp"}
_ACK = {"ok", "oke", "okay", "uh", "vang", "da", "duoc roi", "hieu roi"}

# từ đệm chỉ bỏ ở HAI ĐẦU tin nhắn ("chào bạn ạ" → "chao"; "bạn là ai" vẫn giữ "ban" ở giữa câu)
_EDGE_FILLERS = {
    "a",
    "nhe",
    "nha",
    "oi",
    "ad",
    "admin",
    "bot",
    "ban",
    "minh",
    "em",
    "nhieu",
    "shop",
}

REPLIES = {
    "greet": (
        "Chào bạn! Mình là trợ lý hỏi đáp về cộng đồng S2B, gồm CLB Lập Trình PTIT, CLB Multimedia PTIT và "
        "CLB Nhà Sáng Tạo Game PTIT (GCC). Bạn muốn biết gì về các CLB, ví dụ lịch tuyển thành viên, hoạt động "
        "hay lộ trình đào tạo?"
    ),
    "thanks": "Không có gì! Bạn cần hỏi thêm gì về ba CLB thì cứ nhắn nhé.",
    "who": (
        "Mình là chatbot hỏi đáp về cộng đồng S2B. Với mỗi câu hỏi, mình tìm các đoạn liên quan trong tài liệu "
        "của ba CLB rồi trả lời dựa trên đó, kèm nguồn. Điều gì tài liệu không ghi thì mình nói là không có "
        "thông tin, không đoán."
    ),
    "bye": "Tạm biệt bạn! Khi cần thông tin về các CLB thì quay lại hỏi mình nhé.",
    "ack": "Bạn cần hỏi thêm gì về ba CLB thì cứ nhắn nhé.",
    "scope": (
        "Mình chỉ hỗ trợ hỏi đáp về cộng đồng S2B (CLB Lập Trình PTIT, CLB Multimedia PTIT và CLB Nhà Sáng Tạo "
        "Game PTIT), nên không trả lời câu này. Bạn thử hỏi về lịch tuyển, hoạt động hay lộ trình đào tạo nhé."
    ),
}
_KINDS = [("greet", _GREET), ("thanks", _THANKS), ("who", _WHO), ("bye", _BYE), ("ack", _ACK)]


def _forms(query: str) -> set[str]:
    """Các dạng chuẩn hoá để so khớp: nguyên câu, và câu đã bỏ từ đệm ở hai đầu."""
    words = re.sub(r"[^\w\s]", " ", strip_accents(query.lower())).split()
    forms = {" ".join(words)}
    while words and words[0] in _EDGE_FILLERS:
        words = words[1:]
    while words and words[-1] in _EDGE_FILLERS:
        words = words[:-1]
    forms.add(" ".join(words))
    return forms - {""}


_MATH_WORDS = re.compile(r"\b(bang may|bang bao nhieu|la bao nhieu|bang gi|la may|tinh|bang)\b")
_MATH = re.compile(r"[\d\s+\-*/x×÷=().,^%?:]+")


def _no_question(query: str) -> bool:
    """Tin nhắn không có chữ cái ("???", emoji, "10/11") hoặc chỉ là phép tính ("1 + 1", "2x3 bang may")."""
    if not re.search(r"[^\W\d_]", query):
        return True
    core = _MATH_WORDS.sub(" ", strip_accents(query.lower()))
    return bool(re.search(r"\d", core)) and bool(_MATH.fullmatch(core))


def smalltalk_reply(query: str) -> str | None:
    """Câu trả lời có sẵn nếu cả tin nhắn là câu xã giao, hoặc không phải câu hỏi (phép tính, ký hiệu);
    ngược lại None (đi qua pipeline RAG). Server gọi cả trên câu haiku đã viết lại ("ni hao" → "Xin chào")."""
    if _no_question(query):
        return REPLIES["scope"]
    if len(query) > 60:  # câu dài thì gần như chắc chắn là câu hỏi thật
        return None
    forms = _forms(query)
    for kind, phrases in _KINDS:
        if forms & phrases:
            return REPLIES[kind]
    return None
