"""Chatbot: nhớ ngắn hạn cho câu hỏi nối tiếp ("thế còn GCC thì sao?").

Trình duyệt gửi kèm tối đa 2 lượt hỏi đáp gần nhất (lưu trong biến JS của trang). Khi có lịch sử, MỌI câu hỏi mới
được haiku viết lại thành MỘT câu hỏi đầy đủ, có dấu, tự đủ nghĩa (conversational query rewriting / "condense
question") rồi mới đi qua pipeline — retriever không hiểu được "thế còn GCC thì sao?". Câu đã tự đủ nghĩa thì
prompt dặn chỉ thêm dấu, giữ nguyên ý. Không dùng luật đoán "câu nối tiếp" (dễ bỏ sót, vd "GCC có không?").
Câu viết lại có dấu nên retrieval không gọi thêm bước thêm dấu (R7): câu không dấu vẫn chỉ tốn 1 lần gọi trước khi
trả lời; câu có dấu tốn thêm 1 lần gọi ngắn (~1 s).

Không lưu gì ở server; tải lại trang = cuộc trò chuyện mới. Không dùng trong `run()` (bài nộp, câu hỏi độc lập).
"""

from __future__ import annotations

import json
import re

MAX_TURNS = 2  # số lượt hỏi đáp gần nhất được dùng làm ngữ cảnh
MAX_TURN_CHARS = 400  # cắt câu trả lời cũ cho gọn prompt

CONDENSE_SYSTEM = """Bạn viết lại câu hỏi nối tiếp trong một cuộc trò chuyện về cộng đồng S2B tại Học viện Công \
nghệ Bưu chính Viễn thông (PTIT), gồm CLB Lập Trình PTIT, CLB Multimedia PTIT và CLB Nhà Sáng Tạo Game PTIT (GCC).

Dựa vào hội thoại trước, viết lại CÂU HỎI MỚI thành đúng một câu hỏi tiếng Việt có dấu, đầy đủ, tự hiểu được mà \
không cần hội thoại: thay các từ như "đó", "bên đó", "còn ... thì sao", "vậy" bằng đối tượng cụ thể; gọi CLB bằng tên \
đầy đủ; mở rộng viết tắt. Giữ nguyên điều người dùng muốn hỏi, không thêm ý mới, không trả lời câu hỏi. Nếu câu hỏi \
mới đã tự đủ nghĩa hoặc không liên quan hội thoại trước, chỉ viết lại nó có dấu đầy đủ.
Trả về đúng một object JSON, không giải thích: {"query": "câu hỏi đã viết lại"}"""


def clean_history(raw) -> list[dict]:
    """Lịch sử từ trình duyệt → tối đa MAX_TURNS lượt {"q", "a"} hợp lệ, đã cắt ngắn. Dữ liệu lạ thì bỏ qua."""
    turns = []
    for turn in raw if isinstance(raw, list) else []:
        if (
            isinstance(turn, dict)
            and isinstance(turn.get("q"), str)
            and isinstance(turn.get("a"), str)
        ):
            q, a = turn["q"].strip()[:MAX_TURN_CHARS], turn["a"].strip()[:MAX_TURN_CHARS]
            if q and a:
                turns.append({"q": q, "a": a})
    return turns[-MAX_TURNS:]


def condense_messages(query: str, history: list[dict]) -> list[dict]:
    lines = []
    for turn in history:
        lines += [f"Người dùng: {turn['q']}", f"Trợ lý: {turn['a']}"]
    user = "HỘI THOẠI TRƯỚC:\n" + "\n".join(lines) + f"\n\nCÂU HỎI MỚI: {query}"
    return [{"role": "system", "content": CONDENSE_SYSTEM}, {"role": "user", "content": user}]


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_condensed(reply: str) -> str | None:
    """Lấy trường "query" trong JSON haiku trả về. Haiku đôi khi viết kèm lời giải thích ("Câu hỏi mới: ... Đây là câu
    hỏi nối tiếp, cần ..."), nên KHÔNG lấy nguyên văn phản hồi: không có JSON hợp lệ thì trả None."""
    match = _JSON_OBJECT.search(reply)
    if not match:
        return None
    try:
        value = json.loads(match.group(0)).get("query")
    except (json.JSONDecodeError, AttributeError):
        return None
    value = value.strip() if isinstance(value, str) else ""
    return value if 0 < len(value) <= 300 else None


def condense(llm, model: str, query: str, history: list[dict]) -> str:
    """Câu hỏi đầy đủ do haiku viết lại; phản hồi không đọc được hoặc gateway lỗi thì giữ câu gốc (vẫn trả lời được)."""
    try:
        reply = llm.chat(condense_messages(query, history), model=model)
    except Exception:  # gateway lỗi: không chặn câu hỏi, chỉ mất phần ngữ cảnh
        return query
    return parse_condensed(reply) or query
