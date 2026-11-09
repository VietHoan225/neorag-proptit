"""R12: lượt kiểm tra bằng chứng trên TOÀN corpus cho câu có dấu hiệu thiếu nguồn (thử 07/11).

Corpus chỉ 118 đoạn (~45k ký tự) nên haiku đọc hết được trong 1 lần gọi — cứu được đoạn gold nằm ngoài 20 ứng viên
mà hai lần đọc của A5 không bao giờ thấy. Chỉ đổi `retrieved_ids`, câu trả lời giữ nguyên.

Chống thêm đoạn "liên quan chung" (lỗi của A6/A7): (1) chỉ chạy khi trigger bật (không dùng gold); (2) mỗi đoạn bổ
sung phải kèm một câu trích NGUYÊN VĂN có thật trong đoạn đó; (3) đoạn bổ sung chỉ thay đoạn top-5 mà hai lần đọc
không đồng thuận. Cache dev_casual/holdout_casual (14 câu gắn cờ): thêm 1 đoạn, là gold; 0 gold bị đẩy ra.
"""

from __future__ import annotations

import re
import unicodedata

from neorag.data import Document
from s2b_rag.aliases import S2B, detect_clubs_multi
from s2b_rag.generate import _trigrams, format_context

CHECK_PROMPT = """Bạn kiểm tra bằng chứng cho một hệ thống hỏi đáp về cộng đồng S2B (CLB Lập Trình PTIT, CLB Multimedia PTIT, CLB Nhà Sáng Tạo Game PTIT).

Bạn nhận: câu hỏi, các ý hỏi, id các đoạn ĐÃ CHỌN, và TOÀN BỘ tài liệu (mỗi đoạn có [id], CLB, chủ đề).
Với từng ý hỏi, xét các đoạn đã chọn có chứa thông tin TRẢ LỜI TRỰC TIẾP ý đó không.
- Nếu đủ: ghi "Ý n: ĐỦ".
- Nếu thiếu và có MỘT đoạn khác (không nằm trong danh sách đã chọn) trả lời trực tiếp ý đó: ghi
  Ý n: THIẾU | BỔ SUNG: <id> | TRÍCH: "<một câu chép NGUYÊN VĂN từ đoạn đó, chứa thông tin trả lời>"
- Nếu thiếu mà không đoạn nào trả lời trực tiếp: ghi "Ý n: THIẾU | KHÔNG TÌM THẤY".

Quy tắc:
- Chỉ bổ sung đoạn nói đúng CLB, đúng đối tượng/vai trò, đúng thời gian mà ý hỏi nhắc tới.
- KHÔNG bổ sung đoạn chỉ liên quan chung (giới thiệu CLB, lịch sử, hoạt động khác) mà không chứa câu trả lời.
- Mỗi ý bổ sung tối đa 1 đoạn. Không giải thích, không viết gì ngoài các dòng trên."""

CHECK_USER = """CÂU HỎI: {query}

CÁC Ý HỎI:
{aspects}

ĐÃ CHỌN: {chosen}

TOÀN BỘ TÀI LIỆU:
{corpus}"""

_ASPECT = re.compile(
    r"^\s*[*_]*Ý\s*(\d+)\s*[*_]*:(.*?)\|\s*[*_]*NGUỒN[*_\s]*:(.*)$", re.IGNORECASE | re.MULTILINE
)
_ANSWER_HEAD = re.compile(r"TRẢ LỜI[\s*_]*:", re.IGNORECASE)
_SUPPLEMENT = re.compile(
    r"Ý\s*(\d+)\s*:\s*THIẾU\s*\|\s*BỔ SUNG\s*:\s*(\d+)\s*\|\s*TRÍCH\s*:\s*[\"“](.+?)[\"”]\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def aspect_lines(raw: str, allowed: set[int]) -> list[tuple[str, list[int]]]:
    """Các dòng "Ý n: ... | NGUỒN: ..." trước "TRẢ LỜI:" → [(nội dung ý, id hợp lệ)]; giữ cả ý KHÔNG có nguồn."""
    head = _ANSWER_HEAD.search(raw)
    part = raw[: head.start()] if head else ""
    return [
        (m.group(2).strip(), [int(t) for t in re.findall(r"\d+", m.group(3)) if int(t) in allowed])
        for m in _ASPECT.finditer(part)
    ]


def needs_check(
    query: str,
    aspects1: list[tuple[str, list[int]]],
    aspects2: list[tuple[str, list[int]]],
    primary1: set[int],
    primary2: set[int],
    top5_clubs: set[str],
) -> bool:
    """Trigger đặt trước, không dùng gold. Gắn cờ: dev_casual 10/86, holdout_casual 4/40, public 11/42.

    F1: có ý hỏi (lần đọc 1 hoặc 2) không có nguồn. F2: câu nhắc >= 2 CLB mà top-5 thiếu đoạn của một CLB được nhắc.
    F3: >= 2 ý hỏi và đoạn chính của 2 lần đọc lệch nhiều (Jaccard < 0.5)."""
    f1 = any(not ids for _, ids in aspects1 + aspects2)
    clubs = [c for c in detect_clubs_multi(query) if c != S2B]
    f2 = len(clubs) >= 2 and any(c not in top5_clubs for c in clubs)
    jaccard = len(primary1 & primary2) / max(1, len(primary1 | primary2))
    f3 = len(aspects1) >= 2 and jaccard < 0.5
    return f1 or f2 or f3


def check_messages(
    query: str, aspects: list[tuple[str, list[int]]], top5: list[int], corpus: list[Document]
) -> list[dict]:
    texts = [text for text, _ in aspects] or [query]  # haiku không ghi ý → coi cả câu là 1 ý
    return [
        {"role": "system", "content": CHECK_PROMPT},
        {
            "role": "user",
            "content": CHECK_USER.format(
                query=query,
                aspects="\n".join(f"Ý {i + 1}: {t}" for i, t in enumerate(texts)),
                chosen=", ".join(map(str, top5)),
                corpus=format_context(corpus),
            ),
        },
    ]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def parse_supplements(raw: str, texts: dict[int, str], top5: list[int]) -> list[int]:
    """Đoạn bổ sung hợp lệ: id có thật, chưa ở top-5, câu trích (>= 15 ký tự) có NGUYÊN VĂN trong đoạn đó."""
    added: list[int] = []
    for m in _SUPPLEMENT.finditer(raw):
        doc_id, quote = int(m.group(2)), _norm(m.group(3))
        if (
            doc_id in texts
            and doc_id not in top5
            and doc_id not in added
            and len(quote) >= 15
            and quote in _norm(texts[doc_id])
        ):
            added.append(doc_id)
    return added


def insert_supplements(
    ids: list[int], added: list[int], agreed: set[int], cut: bool = True
) -> list[int]:
    """Mỗi đoạn bổ sung thay đoạn thấp nhất trong top-5 KHÔNG được 2 lần đọc đồng thuận; đoạn bị thay xuống hạng 6.

    Hết chỗ (cả top-5 đều đồng thuận) thì dừng. `cut=True` (R12, bài #15/#16): cắt còn len(ids) id → đoạn hạng 10 rơi.
    `cut=False`: giữ cả đoạn dư để `drop_least_used` chọn đoạn bỏ."""
    top, rest = list(ids[:5]), list(ids[5:])
    for doc_id in added:
        slots = [
            p for p in range(len(top) - 1, -1, -1) if top[p] not in agreed and top[p] not in added
        ]
        if not slots:
            break
        pos = slots[0]
        rest = [top[pos], *[x for x in rest if x != doc_id]]
        top[pos] = doc_id
    out = top + [x for x in rest if x not in top]
    return out[: len(ids)] if cut else out


def drop_least_used(
    answer: str, ids: list[int], texts: dict[int, str], keep: int = 10, top_k: int = 5
) -> list[int]:
    """Cắt còn `keep` id bằng cách bỏ đoạn hạng top_k+1.. góp ít cụm 3 từ RIÊNG nhất cho câu trả lời (hòa → hạng thấp
    nhất), thay vì luôn bỏ đoạn cuối. Không đụng top-k (Recall/MRR/NDCG giữ nguyên).

    Lỗi thật ở public #16 (R12): chèn đoạn 78 vào public-0016 đẩy đoạn 91 khỏi 10 id, mà link "https://mpc.proptit.com"
    trong câu trả lời chỉ có ở đoạn 91 → Groundedness 99.36 → 97.65. Cùng nguyên tắc với `keep_used_sources` (A)."""
    answer_grams = _trigrams(answer)
    grams = {i: _trigrams(texts[i]) & answer_grams for i in ids}
    out = list(ids)
    while len(out) > keep:

        def unique(p: int) -> int:
            others = set().union(*(grams[out[q]] for q in range(len(out)) if q != p))
            return len(grams[out[p]] - others)

        del out[min(reversed(range(top_k, len(out))), key=unique)]
    return out


# ---- R13 (thử 07/11): kiểm tra MỌI câu, tách theo từng đối tượng, được bổ sung nhiều đoạn ----
# Lý do: trigger R12 bỏ sót holdout 0099 (gold ngoài 20 ứng viên, không dính cờ) và dev 0044 bị gắn cờ nhưng haiku
# trả "ĐỦ" dù thiếu đoạn 103 (câu nhiều CLB). Ví dụ trong prompt cố ý KHÔNG lấy từ dev/holdout/public.

CHECK_PROMPT_R13 = """Bạn kiểm tra bằng chứng cho một hệ thống hỏi đáp về cộng đồng S2B (CLB Lập Trình PTIT, CLB Multimedia PTIT, CLB Nhà Sáng Tạo Game PTIT).

Bạn nhận: câu hỏi, các ý hỏi (có thể chưa đủ), id các đoạn ĐÃ CHỌN, và TOÀN BỘ tài liệu (mỗi đoạn có [id], CLB, chủ đề).
Bước 1: tách câu hỏi thành các ý cần bằng chứng riêng. Câu hỏi về nhiều đối tượng (nhiều CLB, nhiều người, nhiều sự kiện) hoặc câu so sánh thì MỖI ĐỐI TƯỢNG là một ý. Ví dụ "ProPTIT và GCC CLB nào đông thành viên hơn" → "số thành viên ProPTIT", "số thành viên GCC". "Ba CLB", "S2B" = cả ba CLB.
Bước 2: với từng ý, xét các đoạn ĐÃ CHỌN có chứa thông tin TRẢ LỜI TRỰC TIẾP ý đó không. Mỗi ý ghi đúng một dòng:
Ý n: <ý> | ĐỦ
Ý n: <ý> | THIẾU | BỔ SUNG: <id> | TRÍCH: "<một câu chép NGUYÊN VĂN từ đoạn đó, chứa thông tin trả lời>"
Ý n: <ý> | THIẾU | KHÔNG TÌM THẤY

Quy tắc:
- Đoạn bổ sung phải KHÔNG nằm trong danh sách đã chọn, và nói đúng đối tượng/vai trò, đúng thời gian mà ý hỏi nhắc tới.
- KHÔNG bổ sung đoạn chỉ liên quan chung (giới thiệu CLB, lịch sử, hoạt động khác) mà không chứa câu trả lời.
- Mỗi ý bổ sung tối đa 1 đoạn. Không giải thích, không viết gì ngoài các dòng trên."""

_SUPPLEMENT_R13 = re.compile(
    r"BỔ SUNG\s*:\s*(\d+)\s*\|\s*TRÍCH\s*:\s*[\"“](.+?)[\"”]\s*$", re.IGNORECASE | re.MULTILINE
)


def check_messages_r13(
    query: str, aspects: list[tuple[str, list[int]]], top5: list[int], corpus: list[Document]
) -> list[dict]:
    messages = check_messages(query, aspects, top5, corpus)
    messages[0] = {"role": "system", "content": CHECK_PROMPT_R13}
    return messages


def parse_supplements_r13(raw: str, texts: dict[int, str], top5: list[int]) -> list[int]:
    """Như `parse_supplements` nhưng nhận dòng có mô tả ý ("Ý n: <ý> | THIẾU | BỔ SUNG: ...")."""
    added: list[int] = []
    for m in _SUPPLEMENT_R13.finditer(raw):
        doc_id, quote = int(m.group(1)), _norm(m.group(2))
        if (
            doc_id in texts
            and doc_id not in top5
            and doc_id not in added
            and len(quote) >= 15
            and quote in _norm(texts[doc_id])
        ):
            added.append(doc_id)
    return added


def insert_supplements_r13(
    ids: list[int], added: list[int], agreed: set[int], cited: set[int]
) -> list[int]:
    """Như `insert_supplements` nhưng thay ĐOẠN LẤP (không lần đọc nào trích) trước, rồi mới tới đoạn được trích mà
    không đồng thuận; đoạn đồng thuận không bao giờ bị thay. Đoạn bị thay xuống hạng 6."""
    top, rest = list(ids[:5]), list(ids[5:])
    for doc_id in added:
        free = [
            p for p in range(len(top) - 1, -1, -1) if top[p] not in agreed and top[p] not in added
        ]
        slots = [p for p in free if top[p] not in cited] + [p for p in free if top[p] in cited]
        if not slots:
            break
        pos = slots[0]
        rest = [top[pos], *[x for x in rest if x != doc_id]]
        top[pos] = doc_id
    return (top + [x for x in rest if x not in top])[: len(ids)]
