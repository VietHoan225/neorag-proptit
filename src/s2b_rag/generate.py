"""Sinh câu trả lời: dựng prompt từ các đoạn đã truy xuất (G1) và làm sạch câu trả lời (G2).

Prompt được viết theo đặc điểm của đáp án chuẩn (train, ~210 ký tự, 1–3 câu, đi thẳng vào ý, chép nguyên
văn số/ngày/tên) — không nhắm vào judge: không có nhãn chấm điểm, không chỉ dẫn cho người chấm.
"""

from __future__ import annotations

import re

from neorag.data import Document

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp về cộng đồng S2B tại Học viện Công nghệ Bưu chính Viễn thông (PTIT), \
gồm CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

Chỉ dùng thông tin trong các TÀI LIỆU được cung cấp. Cách trả lời:
- Trả lời thẳng vào điều được hỏi trong 1–3 câu, không chào hỏi, không mở bài, không kết luận, không mời hỏi thêm.
- Nếu câu hỏi có nhiều ý, trả lời lần lượt từng ý, mỗi ý gọn trong một câu.
- Viết văn xuôi liền mạch: không dùng markdown, in đậm, tiêu đề, gạch đầu dòng hay đánh số.
- Chép nguyên văn từ tài liệu các con số, ngày tháng, tên người, tên sự kiện, đường link, khẩu hiệu.
- Gọi CLB bằng tên đầy đủ như trong tài liệu (vd "CLB Multimedia PTIT") dù câu hỏi viết tắt hoặc không dấu.
- Giữ mốc thời gian mà tài liệu nêu (nhiệm kỳ, đợt tuyển, năm, ngày đăng) thay vì nói như quy định cố định.
- Chỉ nêu thông tin phục vụ câu hỏi; không thêm chi tiết bên lề.
- Nếu tài liệu không có thông tin được hỏi, nói rõ là tài liệu hiện có không có thông tin đó; không đoán."""

USER_TEMPLATE = """TÀI LIỆU:
{context}

CÂU HỎI: {query}"""


def format_context(docs: list[Document]) -> str:
    """Mỗi đoạn: "[id] CLB · chủ đề" rồi nội dung; đoạn xếp theo thứ hạng truy xuất."""
    return "\n\n".join(f"[{d.id}] {d.club} · {d.topic}\n{d.text}" for d in docs)


def build_messages(query: str, docs: list[Document]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=format_context(docs), query=query),
        },
    ]


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_SPACES = re.compile(r"\s+")
_END = re.compile(r"[.!?…:;]$")


def postprocess(answer: str) -> str:
    """G2: bỏ markdown (tiêu đề, gạch đầu dòng, in đậm), nối các dòng thành một đoạn văn.

    Dòng không có dấu kết câu thì thêm dấu chấm khi nối, để tách câu (bộ chấm tách theo dấu câu) vẫn đúng.
    Không đụng vào nội dung chữ: số, ngày, link giữ nguyên."""
    text = _HEADING.sub("", answer)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub("", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = " ".join(line if _END.search(line) else line + "." for line in lines[:-1])
    if lines:
        joined = f"{joined} {lines[-1]}".strip()
    return _SPACES.sub(" ", joined).strip()
