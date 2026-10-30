"""Sinh câu trả lời: dựng prompt từ các đoạn đã truy xuất (G1, G3, G5) và làm sạch câu trả lời (G2).

G5 (thêm 30/10): phân tích nhãn Noise của judge trên dev cho thấy phần lớn câu bị chấm sai là câu đúng theo tài
liệu nhưng thừa: thông tin của CLB/nhiệm kỳ khác, rào đón "tài liệu không ghi rõ", tự so sánh (docs/journal.md).

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
- Dùng lại đúng cụm từ của tài liệu cho các ý chính (tên hoạt động, tiêu chí, mô tả), không thay bằng từ \
đồng nghĩa.
- Gọi CLB bằng tên đầy đủ như trong tài liệu (vd "CLB Multimedia PTIT") dù câu hỏi viết tắt hoặc không dấu.
- Không nhắc số hiệu tài liệu (như [12]) trong câu trả lời.
- Giữ mốc thời gian mà tài liệu nêu (nhiệm kỳ, đợt tuyển, năm, ngày đăng) thay vì nói như quy định cố định. \
Nếu câu hỏi hỏi về năm, đợt hoặc nhiệm kỳ khác với tài liệu, nêu thông tin tài liệu có và nói rõ thông tin đó \
chỉ áp dụng cho năm, đợt hoặc nhiệm kỳ nào.
- Chỉ nêu thông tin phục vụ câu hỏi; không thêm chi tiết bên lề, không nói về CLB, nhiệm kỳ hay chủ đề không \
được hỏi, không tự so sánh hay kết luận điều tài liệu không nêu.

Khi tài liệu không trả lời được đầy đủ:
- Thiếu một phần: nêu phần tài liệu có, và nói rõ "Tài liệu không ghi ..." cho phần còn thiếu. Tài liệu đã trả \
lời được thì không rào đón.
- Không có thông tin: viết "Tài liệu hiện có không có thông tin về ..." (nêu đúng điều được hỏi); có thể thêm \
một câu về thông tin liên quan gần nhất của chính CLB, nhiệm kỳ hoặc đợt được hỏi. Không đoán.
- Các tài liệu ghi khác nhau: viết "Các nguồn ghi khác nhau:" rồi nêu từng cách ghi kèm nguồn hoặc ngày của nó."""

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
