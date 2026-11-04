"""Sinh câu trả lời: dựng prompt từ các đoạn đã truy xuất (G1, G3, G5) và làm sạch câu trả lời (G2).

G5 (thêm 30/10): phân tích nhãn Noise của judge trên dev cho thấy phần lớn câu bị chấm sai là câu đúng theo tài
liệu nhưng thừa: thông tin của CLB/nhiệm kỳ khác, rào đón "tài liệu không ghi rõ", tự so sánh (docs/journal.md).

Prompt được viết theo đặc điểm của đáp án chuẩn (train, ~210 ký tự, 1–3 câu, đi thẳng vào ý, chép nguyên
văn số/ngày/tên) — không nhắm vào judge: không có nhãn chấm điểm, không chỉ dẫn cho người chấm.
"""

from __future__ import annotations

import re
import unicodedata

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
- Không nhắc số hiệu tài liệu (như [12]) trong câu trả lời (chỉ ghi ở dòng NGUỒN cuối cùng).
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
- Các tài liệu ghi khác nhau: viết "Các nguồn ghi khác nhau:" rồi nêu từng cách ghi kèm nguồn hoặc ngày của nó.

Sau câu trả lời, xuống dòng và viết thêm đúng một dòng cuối "NGUỒN: " kèm số hiệu các tài liệu chứa thông tin \
bạn đã dùng, đủ cho từng ý được hỏi, cách nhau bằng dấu phẩy (vd "NGUỒN: 12, 40"). Không dùng tài liệu nào thì \
viết "NGUỒN: không"."""

# E2 (thử 03/11): bớt câu rào đón thừa. Đo trên dev: judge Noise chấm "incorrect" 33% câu "Tài liệu không ghi..."
# so với 5% câu sự kiện; 28–35% câu trả lời có rào đón trong khi đáp án chuẩn train chỉ 12% (docs/plan.md).
# Chỉ thay 2 luật, phần còn lại giữ nguyên để so sánh được với prompt hiện tại.
_HEDGE_OLD = (
    '- Thiếu một phần: nêu phần tài liệu có, và nói rõ "Tài liệu không ghi ..." cho phần còn thiếu. '
    "Tài liệu đã trả lời được thì không rào đón."
)
_HEDGE_NEW = (
    '- Thiếu một phần: nêu phần tài liệu có, và chỉ viết "Tài liệu không ghi ..." cho ý mà câu hỏi hỏi '
    "trực tiếp nhưng tài liệu không có. Không nói tài liệu thiếu những chi tiết mà câu hỏi không hỏi."
)
_EXTRA_OLD = "không tự so sánh hay kết luận điều tài liệu không nêu."
_EXTRA_NEW = (
    "không tự so sánh hay kết luận điều tài liệu không nêu, không thêm lời khuyên (như theo dõi fanpage) "
    "khi câu hỏi không hỏi."
)
assert _HEDGE_OLD in SYSTEM_PROMPT and _EXTRA_OLD in SYSTEM_PROMPT
SYSTEM_PROMPT_E2 = SYSTEM_PROMPT.replace(_HEDGE_OLD, _HEDGE_NEW).replace(_EXTRA_OLD, _EXTRA_NEW)

# A (thử 04/11) = E2 + 3 thay đổi gộp trong 1 lần chạy để tiết kiệm credit (đổi lại: không tách được tác dụng):
# (1) chọn bằng chứng cho TỪNG Ý trước khi viết đáp án, cùng 1 lần gọi. R9 viết đáp án rồi mới khai nguồn → ý bị
#     sót thì nguồn của nó cũng mất. Cache E2 trên dev_casual: 7 đoạn gold ngoài top-5 — 5 đoạn có trong 20 ứng viên
#     nhưng không được trích, 2 ngoài 20, 0 được trích mà bị đẩy ra (đều ở câu ≥3 gold). Khác R8: không tách câu để
#     truy xuất riêng, LLM vẫn đọc nguyên câu hỏi.
# (3) bỏ ví dụ, câu nhắc lại, câu kết luận "Vậy ..." không được hỏi (gặp ở public #5, vd "Vậy CLB có cả ...").
# (4) ở `keep_used_sources` (code, 0 LLM).
_CITE_RULE_OLD = "(chỉ ghi ở dòng NGUỒN cuối cùng)."
_CITE_RULE_NEW = "(chỉ ghi ở phần Ý HỎI)."
_TRIM_OLD = "không thêm lời khuyên (như theo dõi fanpage) khi câu hỏi không hỏi."
_TRIM_NEW = (
    "không thêm lời khuyên (như theo dõi fanpage) khi câu hỏi không hỏi, không đưa ví dụ minh họa không được hỏi, "
    'không nhắc lại ý đã nói, không thêm câu tóm lại kiểu "Vậy ...".'
)
_FORMAT_OLD = SYSTEM_PROMPT_E2[SYSTEM_PROMPT_E2.index("Sau câu trả lời, xuống dòng") :]
_FORMAT_NEW = """Viết đúng hai phần, theo thứ tự:
Ý HỎI:
Ý 1: <một ý mà câu hỏi hỏi, giữ đúng CLB, đối tượng và mốc thời gian của câu hỏi> | NGUỒN: <số hiệu các tài \
liệu trả lời trực tiếp ý này, tài liệu trả lời trực tiếp nhất ghi trước, cách nhau bằng dấu phẩy; không có thì \
"không">
(mỗi ý được hỏi một dòng như trên: Ý 2, Ý 3...)
TRẢ LỜI:
<câu trả lời theo các cách trả lời ở trên, chỉ dùng thông tin từ các tài liệu đã ghi ở phần Ý HỎI>"""
assert _CITE_RULE_OLD in SYSTEM_PROMPT_E2 and _TRIM_OLD in SYSTEM_PROMPT_E2
assert _FORMAT_OLD.endswith('"NGUỒN: không".')
SYSTEM_PROMPT_A = (
    SYSTEM_PROMPT_E2.replace(_CITE_RULE_OLD, _CITE_RULE_NEW)
    .replace(_TRIM_OLD, _TRIM_NEW)
    .replace(_FORMAT_OLD, _FORMAT_NEW)
)
RETRY_NOTE_A = "\n\nHãy viết đủ cả hai phần: Ý HỎI, rồi TRẢ LỜI."

USER_TEMPLATE = """TÀI LIỆU:
{context}

CÂU HỎI: {query}"""

# Chặn câu trả lời rỗng: haiku đôi khi chỉ viết dòng "NGUỒN: ..." (1/282 phản hồi từ R9), cắt xong còn rỗng → file
# nộp không hợp lệ. Gọi lại 1 lần với câu nhắc này (prompt khác nên không trúng cache cũ); vẫn rỗng thì dùng câu
# dự phòng — luôn nộp được, đổi lại câu đó mất điểm generation.
RETRY_NOTE = "\n\nHãy viết câu trả lời trước, sau đó mới viết dòng NGUỒN ở cuối."
FALLBACK_ANSWER = "Tài liệu hiện có không có thông tin để trả lời câu hỏi này."


def with_retry_note(messages: list[dict], note: str = RETRY_NOTE) -> list[dict]:
    """Bản sao `messages` với câu nhắc nối vào cuối tin nhắn người dùng (không sửa list gốc)."""
    last = {**messages[-1], "content": messages[-1]["content"] + note}
    return [*messages[:-1], last]


def format_context(docs: list[Document]) -> str:
    """Mỗi đoạn: "[id] CLB · chủ đề" rồi nội dung; đoạn xếp theo thứ hạng truy xuất."""
    return "\n\n".join(f"[{d.id}] {d.club} · {d.topic}\n{d.text}" for d in docs)


def build_messages(query: str, docs: list[Document], system: str = SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=format_context(docs), query=query),
        },
    ]


_SOURCES = re.compile(r"^[\s*_]*NGUỒN[\s*_]*:(.*)$", re.IGNORECASE | re.MULTILINE)


def split_sources(raw: str, allowed: list[int]) -> tuple[str, list[int]]:
    """R9: tách dòng "NGUỒN: 12, 40" ra khỏi câu trả lời.

    Trả về (câu trả lời không còn dòng nguồn, id được trích theo thứ tự xuất hiện). Chỉ giữ id nằm trong
    `allowed` (các đoạn đã đưa vào prompt) — id bịa hoặc lạ bị bỏ. Không có dòng nguồn → danh sách rỗng."""
    matches = list(_SOURCES.finditer(raw))
    if not matches:
        return raw, []
    last = matches[-1]
    cited: list[int] = []
    for token in re.findall(r"\d+", last.group(1)):
        doc_id = int(token)
        if doc_id in allowed and doc_id not in cited:
            cited.append(doc_id)
    return raw[: last.start()] + raw[last.end() :], cited


def promote_cited(ranked: list[int], cited: list[int]) -> list[int]:
    """R9: đưa các đoạn được trích lên đầu (giữ thứ tự retriever giữa chúng), phần còn lại giữ nguyên thứ tự.

    Giữ thứ tự retriever thay vì thứ tự LLM ghi: đoạn đứng đầu của reranker (thường là ý chính) vẫn ở hạng 1
    nếu được trích, nên MRR ít bị xáo trộn; LLM chỉ quyết định đoạn NÀO vào top, không quyết định thứ tự."""
    return promote_tiers(ranked, [cited])


def promote_tiers(ranked: list[int], tiers: list[list[int]]) -> list[int]:
    """Đưa từng tầng đoạn lên đầu theo thứ tự tầng; trong mỗi tầng giữ thứ tự retriever; phần còn lại giữ nguyên.

    R9 = 1 tầng (mọi đoạn được trích). A = 2 tầng: đoạn chính của từng ý trước, đoạn phụ sau — để mỗi ý có ít
    nhất một đoạn trong top-5 khi câu hỏi có ≤ 5 ý."""
    out: list[int] = []
    for tier in tiers:
        members = set(tier)
        out += [d for d in ranked if d in members and d not in out]
    return out + [d for d in ranked if d not in out]


_ANSWER_HEAD = re.compile(r"^[\s*_#]*TRẢ LỜI[\s*_]*:[\s*_]*", re.IGNORECASE | re.MULTILINE)
_ASPECT_SOURCES = re.compile(r"NGUỒN[\s*_]*:(.*)$", re.IGNORECASE)


def split_aspects(raw: str, allowed: list[int]) -> tuple[str, list[list[int]]]:
    """A: tách phần "Ý HỎI" (mỗi ý một dòng "Ý n: ... | NGUỒN: a, b") khỏi phần "TRẢ LỜI".

    Trả về (câu trả lời, [đoạn chính, đoạn phụ]): đoạn chính = đoạn ghi đầu tiên của mỗi ý, đoạn phụ = các đoạn
    còn lại được ghi. Chỉ giữ id trong `allowed`. Không có dòng "TRẢ LỜI:" (haiku không theo định dạng) → xử lý
    như R9 (`split_sources`), để vẫn nộp được."""
    heads = list(_ANSWER_HEAD.finditer(raw))
    if not heads:
        answer, cited = split_sources(raw, allowed)
        return answer, [cited]
    head = heads[-1]
    answer, trailing = split_sources(raw[head.end() :], allowed)  # dòng NGUỒN thừa ở cuối (nếu có)
    primary: list[int] = []
    secondary: list[int] = []
    for line in raw[: head.start()].splitlines():
        match = _ASPECT_SOURCES.search(line)
        if not match:
            continue
        ids = [int(t) for t in re.findall(r"\d+", match.group(1)) if int(t) in allowed]
        if ids and ids[0] not in primary:
            primary.append(ids[0])
        secondary += ids[1:]
    secondary = [d for d in dict.fromkeys(secondary + trailing) if d not in primary]
    return answer, [primary, secondary]


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", text.lower())).split()


def _trigrams(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def keep_used_sources(
    answer: str,
    kept: list[int],
    candidates: list[int],
    texts: dict[int, str],
    protected: set[int],
    top_k: int = 5,
    min_new: int = 1,
) -> list[int]:
    """A (4): đoạn haiku đã dùng nhưng quên ghi nguồn → đổi vào hạng top_k+1..10, để judge Groundedness thấy nó.

    Lỗi thật ở public #5 (public-0016): "tài khoản Instagram clbmultimedia.ptit" chỉ có ở đoạn hạng 11–20 bị cắt.
    Cụm "đặc trưng" = cụm 3 từ của câu trả lời chứa một từ mà không đoạn nộp nào có (vd "instagram") — loại báo động
    giả do tên CLB ("nhà sáng tạo game" có sẵn ở đoạn nộp). Một ứng viên (đã vào prompt, không được nộp) mang ≥
    `min_new` cụm đặc trưng chưa được phủ thì thay đoạn ở hạng top_k+1..10 đóng góp ít cụm riêng nhất (không phải
    đoạn được trích), nếu đoạn đó đóng góp ít hơn. Không đụng top_k (Recall/MRR/NDCG chỉ tính @5).

    Mô phỏng trên câu trả lời E2 (cache, 0 USD): ngưỡng 2 bỏ sót public-0016 (haiku diễn đạt lại → chỉ 1 cụm khớp);
    ngưỡng 1 sửa được nó, đổi 5/86 câu dev_casual — đoạn bị thay luôn đóng góp 0 cụm → xấu nhất là trung tính."""
    answer_grams = _trigrams(answer)
    kept_words = {w for i in kept for w in _words(texts[i])}
    distinctive = {g for g in answer_grams if any(w not in kept_words for w in g)}
    grams = {i: _trigrams(texts[i]) & answer_grams for i in [*kept, *candidates]}
    out = list(kept)
    for cand in candidates:
        gain = (grams[cand] & distinctive) - set().union(*(grams[i] for i in out))
        if len(gain) < min_new:
            continue
        slots = [p for p in range(top_k, len(out)) if out[p] not in protected]
        if not slots:
            break

        def unique(p: int) -> int:
            return len(grams[out[p]] - set().union(*(grams[i] for i in out if i != out[p])))

        pos = min(reversed(slots), key=unique)  # ít đóng góp nhất; hòa → hạng thấp nhất
        if unique(pos) < len(gain):
            out[pos] = cand
    return out


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
