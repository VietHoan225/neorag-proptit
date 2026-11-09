"""R12: lượt kiểm tra bằng chứng toàn corpus (`s2b_rag.evidence`) và `S2BPipelineR12`."""

from conftest import FakeLLM
from neorag.data import Document
from s2b_rag.evidence import (
    CHECK_PROMPT,
    aspect_lines,
    insert_supplements,
    needs_check,
    parse_supplements,
)
from test_s2b_pipeline import _FixedRanker


def test_aspect_lines_keeps_aspects_without_valid_source():
    raw = "Ý HỎI:\nÝ 1: ngày thành lập | NGUỒN: 4, 99\nÝ 2: phương châm | NGUỒN: không có\nTRẢ LỜI:\nX."
    assert aspect_lines(raw, {4, 5}) == [("ngày thành lập", [4]), ("phương châm", [])]


def test_needs_check_flags():
    full = [("a", [1])]
    assert not needs_check("câu hỏi", full, full, {1}, {1}, set())
    assert needs_check("câu hỏi", [("a", [])], full, {1}, {1}, set())  # F1: ý không có nguồn
    two = [("a", [1]), ("b", [2])]
    assert needs_check("câu hỏi", two, two, {1, 2}, {3, 4}, set())  # F3: 2 lần đọc lệch
    assert not needs_check("câu hỏi", two, two, {1, 2}, {1, 2}, set())


def test_parse_supplements_requires_verbatim_quote_outside_top5():
    texts = {7: "CLB được thành lập ngày 09/10/2011 tại PTIT.", 8: "Đoạn khác hoàn toàn."}
    raw = (
        'Ý 1: THIẾU | BỔ SUNG: 7 | TRÍCH: "được thành lập ngày 09/10/2011"\n'
        'Ý 2: THIẾU | BỔ SUNG: 8 | TRÍCH: "câu bịa không có trong đoạn"\n'
        'Ý 3: THIẾU | BỔ SUNG: 3 | TRÍCH: "đoạn đã nằm trong top-5 rồi"'
    )
    assert parse_supplements(raw, {**texts, 3: "đoạn đã nằm trong top-5 rồi"}, [3, 4]) == [7]


def test_insert_supplements_replaces_lowest_non_agreed_and_keeps_10_ids():
    ids = list(range(1, 11))
    out = insert_supplements(ids, [42], agreed={5})
    assert out[:5] == [1, 2, 3, 42, 5] and out[5] == 4 and len(out) == 10 and 10 not in out
    assert (
        insert_supplements(ids, [42], agreed={1, 2, 3, 4, 5}) == ids
    )  # top-5 đồng thuận hết → giữ


def test_r12_adds_quoted_doc_from_whole_corpus_and_keeps_r11_answer():
    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4
    from s2b_rag.pipeline import S2BPipelineR11, S2BPipelineR12

    replies = {
        SYSTEM_PROMPT_A: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nÝ 2: y | NGUỒN: không có\nTRẢ LỜI:\nCâu trả lời.",
        SYSTEM_PROMPT_A4: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nLần hai.",
        CHECK_PROMPT: 'Ý 1: ĐỦ\nÝ 2: THIẾU | BỔ SUNG: 30 | TRÍCH: "Đoạn số 30 nói điều cần tìm."',
    }
    docs = [
        Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn số {i} nói điều cần tìm.")
        for i in range(1, 31)
    ]
    outs = {}
    for cls in (S2BPipelineR11, S2BPipelineR12):
        llm = FakeLLM(lambda m: replies[m[0]["content"]])
        pipeline = cls(retriever="bm25_noaccent_bi", llm=llm)
        pipeline.retriever = _FixedRanker(list(range(1, 31)))  # đoạn 30 ngoài 20 ứng viên
        pipeline.index(docs)
        outs[cls] = (pipeline.run("câu hỏi"), llm.calls)
    (r11, _), (r12, calls) = outs[S2BPipelineR11], outs[S2BPipelineR12]
    assert r11.retrieved_ids[:5] == [3, 1, 2, 4, 5]
    assert r12.retrieved_ids[:5] == [3, 1, 2, 4, 30] and r12.retrieved_ids[5] == 5
    assert r12.answer == r11.answer and len(r12.retrieved_ids) == 10
    assert sum(c[0]["content"] == CHECK_PROMPT for c in calls) == 1


def test_parse_supplements_r13_accepts_aspect_text_and_several_docs():
    from s2b_rag.evidence import parse_supplements_r13

    texts = {7: "ProPTIT thành lập ngày 09/10/2011.", 9: "GCC thành lập năm 2021 tại PTIT."}
    raw = (
        'Ý 1: ngày thành lập ProPTIT | THIẾU | BỔ SUNG: 7 | TRÍCH: "ProPTIT thành lập ngày 09/10/2011"\n'
        'Ý 2: ngày thành lập GCC | THIẾU | BỔ SUNG: 9 | TRÍCH: "GCC thành lập năm 2021 tại PTIT"\n'
        "Ý 3: ngày thành lập MPC | ĐỦ"
    )
    assert parse_supplements_r13(raw, texts, [1, 2]) == [7, 9]


def test_insert_supplements_r13_replaces_uncited_filler_first():
    from s2b_rag.evidence import insert_supplements_r13

    ids = list(range(1, 11))
    # 5 được trích (không đồng thuận), 4 là đoạn lấp → 4 bị thay trước
    out = insert_supplements_r13(ids, [42], agreed={1}, cited={1, 2, 3, 5})
    assert out[:5] == [1, 2, 3, 42, 5] and out[5] == 4 and len(out) == 10


def test_r13_checks_every_question_with_r13_prompt():
    from s2b_rag.evidence import CHECK_PROMPT_R13
    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4
    from s2b_rag.pipeline import S2BPipelineR13

    replies = {
        SYSTEM_PROMPT_A: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nCâu trả lời.",
        SYSTEM_PROMPT_A4: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nLần hai.",
        CHECK_PROMPT_R13: 'Ý 1: x | ĐỦ\nÝ 2: y | THIẾU | BỔ SUNG: 30 | TRÍCH: "Đoạn số 30 nói điều cần tìm."',
    }
    llm = FakeLLM(lambda m: replies[m[0]["content"]])
    docs = [
        Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn số {i} nói điều cần tìm.")
        for i in range(1, 31)
    ]
    pipeline = S2BPipelineR13(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 31)))
    pipeline.index(docs)
    out = pipeline.run("câu hỏi")  # không dính cờ R12 nào nhưng R13 vẫn kiểm tra
    assert [c[0]["content"] for c in llm.calls].count(CHECK_PROMPT_R13) == 1
    assert out.retrieved_ids[:5] == [3, 1, 2, 4, 30] and out.answer == "Câu trả lời."


def test_r12mbr_takes_answer_from_mbr_and_ids_from_r12():
    from s2b_rag.generate import (
        SYSTEM_PROMPT_A,
        SYSTEM_PROMPT_A4,
        SYSTEM_PROMPT_E2,
        SYSTEM_PROMPT_F,
    )
    from s2b_rag.pipeline import S2BPipelineMBR, S2BPipelineR12, S2BPipelineR12MBR

    replies = {
        SYSTEM_PROMPT_A: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nÝ 2: y | NGUỒN: không có\nTRẢ LỜI:\nCLB thành lập năm 2011.",
        SYSTEM_PROMPT_A4: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nCLB được thành lập năm 2011 tại PTIT.",
        SYSTEM_PROMPT_E2: "CLB được thành lập năm 2011 tại PTIT.\nNGUỒN: 3",
        SYSTEM_PROMPT_F: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nCLB được thành lập năm 2011 tại PTIT.",
        CHECK_PROMPT: 'Ý 1: ĐỦ\nÝ 2: THIẾU | BỔ SUNG: 30 | TRÍCH: "Đoạn số 30 nói điều cần tìm."',
    }
    docs = [
        Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn số {i} nói điều cần tìm.")
        for i in range(1, 31)
    ]
    outs = {}
    for cls in (S2BPipelineMBR, S2BPipelineR12, S2BPipelineR12MBR):
        pipeline = cls(
            retriever="bm25_noaccent_bi", llm=FakeLLM(lambda m: replies[m[0]["content"]])
        )
        pipeline.retriever = _FixedRanker(list(range(1, 31)))
        pipeline.index(docs)
        outs[cls] = pipeline.run("câu hỏi")
    both = outs[S2BPipelineR12MBR]
    assert both.answer == outs[S2BPipelineMBR].answer == "CLB được thành lập năm 2011 tại PTIT."
    assert both.retrieved_ids[:5] == outs[S2BPipelineR12].retrieved_ids[:5]
    assert 30 in both.retrieved_ids[:5]


def test_drop_least_used_keeps_doc_that_alone_supports_answer():
    from s2b_rag.evidence import drop_least_used

    texts = {i: f"đoạn số {i} chung chung" for i in range(1, 12)}
    texts[11] = "link đăng ký là https mpc proptit com"
    answer = "Link đăng ký là https mpc proptit com."
    ids = [
        1,
        2,
        3,
        4,
        42,
        5,
        6,
        7,
        8,
        9,
        11,
    ]  # 11 id sau khi chèn 42; đoạn 11 (cuối) là nguồn duy nhất của link
    texts[42] = "đoạn bổ sung"
    out = drop_least_used(answer, ids, texts, keep=10)
    assert len(out) == 10 and 11 in out and out[:5] == ids[:5] and 9 not in out
    # không đoạn nào được dùng → bỏ đoạn cuối, y như cắt thường
    assert drop_least_used("không liên quan", ids, texts, keep=10) == ids[:10]


def _refresh_replies():
    """LLM giả: lần đọc đầu thiếu ý 2 → R12 bổ sung đoạn 30 → sinh lại (ngữ cảnh có "[30] ") dùng đoạn 18 (hạng 18)."""
    from s2b_rag.generate import (
        SYSTEM_PROMPT_A,
        SYSTEM_PROMPT_A4,
        SYSTEM_PROMPT_E2,
        SYSTEM_PROMPT_F,
    )

    new = "Tài khoản instagram clbmultimedia ptit là nơi đăng bài."
    first = {
        SYSTEM_PROMPT_A: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nÝ 2: y | NGUỒN: không có\nTRẢ LỜI:\nCâu trả lời cũ.",
        SYSTEM_PROMPT_A4: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nCâu trả lời cũ.",
        SYSTEM_PROMPT_E2: "Câu trả lời cũ.\nNGUỒN: 3",
        SYSTEM_PROMPT_F: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nCâu trả lời cũ.",
        CHECK_PROMPT: 'Ý 1: ĐỦ\nÝ 2: THIẾU | BỔ SUNG: 30 | TRÍCH: "Đoạn số 30 nói điều cần tìm."',
    }
    again = {k: f"Ý HỎI:\nÝ 1: x | NGUỒN: 30, 18\nTRẢ LỜI:\n{new}" for k in first}
    again[SYSTEM_PROMPT_E2] = f"{new}\nNGUỒN: 30, 18"

    def reply(m):
        refreshed = m[0]["content"] != CHECK_PROMPT and "[30] " in m[1]["content"]
        return (again if refreshed else first)[m[0]["content"]]

    return reply, new


def _refresh_docs():
    docs = [
        Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn số {i} nói điều cần tìm.")
        for i in range(1, 31)
    ]
    docs[17] = Document(
        18,
        "CLB Multimedia PTIT",
        "Mạng xã hội",
        "Tài khoản instagram clbmultimedia ptit là nơi đăng bài.",
    )
    return docs


def test_refresh_answer_after_supplement_uses_new_doc_keeps_its_source_and_hides_format():
    from s2b_rag.pipeline import S2BPipeline, S2BPipelineR12

    class R12Keep(S2BPipelineR12):
        keep_answer_sources = True

    reply, new = _refresh_replies()
    for cls in (R12Keep, S2BPipeline):  # R12 (2 lần đọc) và cấu hình nộp bài (R12MBR)
        pipeline = cls(retriever="bm25_noaccent_bi", llm=FakeLLM(reply))
        pipeline.retriever = _FixedRanker(
            list(range(1, 31))
        )  # đoạn 30 ngoài 20 ứng viên, đoạn 18 hạng 18
        pipeline.index(_refresh_docs())
        out = pipeline.run("câu hỏi")
        assert out.answer == new, cls.__name__  # câu trả lời được sinh lại
        assert (
            "Ý HỎI" not in out.answer and "NGUỒN" not in out.answer
        )  # không rò định dạng (lỗi tách bản A)
        assert 30 in out.retrieved_ids[:5] and len(out.retrieved_ids) == 10
        assert 18 in out.retrieved_ids[5:], (
            cls.__name__
        )  # đoạn câu trả lời MỚI dùng có mặt ở hạng 6–10


def test_no_refresh_when_nothing_added():
    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4
    from s2b_rag.pipeline import S2BPipelineR12

    replies = {
        SYSTEM_PROMPT_A: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nCâu trả lời.",
        SYSTEM_PROMPT_A4: "Ý HỎI:\nÝ 1: x | NGUỒN: 3\nTRẢ LỜI:\nLần hai.",
    }
    llm = FakeLLM(lambda m: replies[m[0]["content"]])
    pipeline = S2BPipelineR12(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 31)))
    pipeline.index(_refresh_docs())
    assert pipeline.run("câu hỏi").answer == "Câu trả lời."
    assert (
        len({c[0]["content"] for c in llm.calls}) == 2
    )  # không gắn cờ → không kiểm tra, không sinh lại
