import re

from conftest import FakeLLM
from neorag.data import Document, Query
from neorag.predict import predict
from s2b_rag.generate import build_messages, format_context, postprocess
from s2b_rag.pipeline import S2BPipelineCtx20, S2BPipelineR9


def test_postprocess_removes_markdown_and_joins_lines():
    raw = "## Trả lời\n**CLB Lập Trình PTIT** thành lập ngày 09/10/2011\n- Phương châm: Chia sẻ"
    assert postprocess(raw) == (
        "Trả lời. CLB Lập Trình PTIT thành lập ngày 09/10/2011. Phương châm: Chia sẻ"
    )


def test_postprocess_keeps_numbers_links_and_plain_text():
    text = "Đăng ký tại https://mpc.proptit.com trước 10/09/2026, phí 0 đồng."
    assert postprocess(text) == text
    assert postprocess("") == ""


def test_context_lists_id_club_topic_in_rank_order(corpus):
    ctx = format_context([corpus[1], corpus[0]])
    assert ctx.startswith("[2] CLB Multimedia PTIT · Giới thiệu\n")
    assert ctx.index("[2]") < ctx.index("[1]")
    messages = build_messages("Multimedia thành lập khi nào?", corpus)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "CÂU HỎI: Multimedia thành lập khi nào?" in messages[1]["content"]


def test_pipeline_end_to_end_with_fake_llm_and_bm25(corpus):
    llm = FakeLLM(lambda messages: "**CLB Multimedia PTIT** thành lập ngày 10/11/2012.")
    pipeline = S2BPipelineR9(retriever="bm25_noaccent_bi", n_context=2, llm=llm)
    pipeline.prepare()
    pipeline.index(corpus)
    (pred,) = predict(pipeline, [Query("q", "clb multimedia thanh lap ngay nao")], max_retrieved=10)
    assert pred.answer == "CLB Multimedia PTIT thành lập ngày 10/11/2012."
    assert pred.retrieved_ids[0] == 2
    assert len(pred.retrieved_ids) == 3  # corpus giả chỉ có 3 đoạn
    prompt = llm.calls[0][1]["content"]
    assert re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE) == ["2", "1"]  # đúng 2 đoạn, theo hạng
    assert "temperature" not in str(llm.calls)


def test_prepare_finds_reranker_in_nested_layers(monkeypatch):
    import s2b_rag.retrieval.rerank as rerank

    loaded = []
    monkeypatch.setattr(rerank, "load_cross_encoder", lambda name=None: loaded.append(name))
    S2BPipelineR9(retriever="hybrid_r7", llm=FakeLLM()).prepare()  # reranker nằm sâu bên trong
    S2BPipelineR9(retriever="bm25_noaccent_bi", llm=FakeLLM()).prepare()  # không có reranker
    assert loaded == ["BAAI/bge-reranker-v2-m3"]


def test_split_sources_strips_line_and_keeps_only_allowed_ids():
    from s2b_rag.generate import split_sources

    raw = "GCC thành lập năm 2019.\nNGUỒN: 97, 12, 97, 555"
    assert split_sources(raw, allowed=[12, 40, 97]) == ("GCC thành lập năm 2019.\n", [97, 12])
    # markdown quanh nhãn, chữ thường
    assert split_sources("Trả lời.\n**Nguồn:** 40", allowed=[40])[1] == [40]
    # không có dòng nguồn / "không" → không trích gì, câu trả lời giữ nguyên
    assert split_sources("Trả lời.", allowed=[1]) == ("Trả lời.", [])
    assert split_sources("Tài liệu không có.\nNGUỒN: không", allowed=[1])[1] == []


def test_promote_cited_moves_cited_to_front_in_retriever_order():
    from s2b_rag.generate import promote_cited

    ranked = [5, 3, 9, 1, 7, 8, 2, 4]
    assert promote_cited(ranked, [2, 3]) == [3, 2, 5, 9, 1, 7, 8, 4]
    assert promote_cited(ranked, []) == ranked


def test_pipeline_promotes_cited_doc_and_hides_sources_line(corpus):
    llm = FakeLLM(lambda messages: "CLB Multimedia PTIT thành lập ngày 10/11/2012.\nNGUỒN: 1")
    pipeline = S2BPipelineR9(retriever="bm25_noaccent_bi", n_context=2, llm=llm)
    pipeline.index(corpus)
    (pred,) = predict(pipeline, [Query("q", "clb multimedia thanh lap ngay nao")], max_retrieved=10)
    assert "NGUỒN" not in pred.answer
    assert pred.answer == "CLB Multimedia PTIT thành lập ngày 10/11/2012."
    assert pred.retrieved_ids[:2] == [1, 2]  # đoạn 1 được trích nên lên trước đoạn 2


class _FixedRanker:
    """Retriever giả: luôn xếp các đoạn theo đúng thứ tự `order`."""

    def __init__(self, order):
        self.order = order

    def index(self, corpus):
        pass

    def search(self, query):
        return [(doc_id, 1.0 / rank) for rank, doc_id in enumerate(self.order, 1)]


def _pipeline_with(n_context, cite, cls=S2BPipelineR9):
    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    llm = FakeLLM(lambda messages: f"Trả lời.\nNGUỒN: {cite}")
    pipeline = cls(retriever="bm25_noaccent_bi", n_context=n_context, llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 26)))  # hạng i = đoạn i
    pipeline.index(docs)
    return pipeline, llm


def test_e1_cited_doc_beyond_rank_10_enters_top5_and_still_10_ids():
    pipeline, llm = _pipeline_with(n_context=20, cite="15, 2")
    out = pipeline.run("câu hỏi")
    # đoạn được trích lên đầu (giữ thứ tự retriever: 2 trước 15), rồi lấp đủ 10 id
    assert out.retrieved_ids == [2, 15, 1, 3, 4, 5, 6, 7, 8, 9]
    assert len(re.findall(r"^\[\d+\]", llm.calls[0][1]["content"], re.MULTILINE)) == 20


def test_e1_ctx10_ignores_unseen_doc_and_matches_cut_before():
    # haiku chỉ thấy 10 đoạn → id 15 bị bỏ (không có trong prompt); kết quả như bài nộp #4
    pipeline, _ = _pipeline_with(n_context=10, cite="15, 2")
    assert pipeline.run("câu hỏi").retrieved_ids == [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]


def test_e1_cut_after_equals_cut_before_when_cited_within_top10():
    import random

    from s2b_rag.generate import promote_cited

    rng = random.Random(0)
    for _ in range(200):
        ranked = rng.sample(range(1, 119), 30)
        cited = rng.sample(ranked[:10], rng.randint(0, 6))
        assert promote_cited(ranked, cited)[:10] == promote_cited(ranked[:10], cited)


def test_ctx20_class_defaults():
    pipeline = S2BPipelineCtx20(retriever="bm25_noaccent_bi", llm=FakeLLM())
    assert (pipeline.n_context, pipeline.n_retrieved) == (20, 10)


def _sources_only_then(second):
    """LLM giả: lần đầu chỉ viết dòng NGUỒN (lỗi thật ở train-0063), lần gọi lại (có câu nhắc) trả `second`."""
    from s2b_rag.generate import RETRY_NOTE

    return FakeLLM(lambda m: second if RETRY_NOTE in m[-1]["content"] else "NGUỒN: 3")


def test_empty_answer_retries_once_with_note(corpus):
    llm = _sources_only_then("CLB Multimedia PTIT thành lập ngày 10/11/2012.\nNGUỒN: 2")
    pipeline = S2BPipelineR9(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.index(corpus)
    out = pipeline.run("clb multimedia thanh lap ngay nao")
    assert out.answer == "CLB Multimedia PTIT thành lập ngày 10/11/2012."
    assert out.retrieved_ids[0] == 2  # nguồn lấy từ lần gọi lại
    assert len(llm.calls) == 2


def test_empty_twice_uses_fallback_but_keeps_first_citation(corpus):
    from s2b_rag.generate import FALLBACK_ANSWER

    llm = _sources_only_then("NGUỒN: 3")
    pipeline = S2BPipelineR9(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.index(corpus)
    out = pipeline.run("clb multimedia thanh lap ngay nao")
    assert out.answer == FALLBACK_ANSWER  # không bao giờ rỗng → file nộp luôn hợp lệ
    assert out.retrieved_ids[0] == 3
    assert len(llm.calls) == 2  # chỉ gọi lại đúng 1 lần


def test_normal_answer_calls_llm_once(corpus):
    llm = FakeLLM(lambda m: "GCC ra đời năm 2023.\nNGUỒN: 3")
    pipeline = S2BPipelineR9(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.index(corpus)
    assert pipeline.run("gcc ra doi nam nao").answer == "GCC ra đời năm 2023."
    assert len(llm.calls) == 1


def test_e2_prompt_changes_only_two_rules():
    import difflib

    from s2b_rag.generate import SYSTEM_PROMPT, SYSTEM_PROMPT_E2

    changed = [
        d
        for d in difflib.ndiff(SYSTEM_PROMPT.splitlines(), SYSTEM_PROMPT_E2.splitlines())
        if d[0] in "+-"
    ]
    assert len(changed) == 4  # 2 dòng bị thay, mỗi dòng một "-" và một "+"
    assert "câu hỏi hỏi trực tiếp" in SYSTEM_PROMPT_E2 and "lời khuyên" in SYSTEM_PROMPT_E2


def test_e2_class_uses_new_prompt_and_default_stays_old():
    from s2b_rag.generate import SYSTEM_PROMPT, SYSTEM_PROMPT_E2
    from s2b_rag.pipeline import S2BPipelineE2

    e2 = S2BPipelineE2(retriever="bm25_noaccent_bi", llm=FakeLLM())
    assert (e2.n_context, e2.system_prompt) == (20, SYSTEM_PROMPT_E2)
    assert S2BPipelineR9(retriever="bm25_noaccent_bi", llm=FakeLLM()).system_prompt == SYSTEM_PROMPT
    pipeline, llm = _pipeline_with(n_context=20, cite="1", cls=S2BPipelineE2)
    pipeline.run("câu hỏi")
    assert llm.calls[0][0]["content"] == SYSTEM_PROMPT_E2


# --- Thí nghiệm A: chọn nguồn theo từng ý (1) + trả lời gọn (3) + giữ nguồn đã dùng (4) ---

_A_REPLY = (
    "Ý HỎI:\n"
    "Ý 1: GCC thành lập năm nào | NGUỒN: 15, 2\n"
    "Ý 2: đơn vị hỗ trợ chuyên môn | NGUỒN: 3\n"
    "**TRẢ LỜI:**\n"
    "GCC thành lập năm 2023 và được Khoa Đa phương tiện hỗ trợ chuyên môn."
)


def test_split_aspects_primary_per_aspect_then_secondary():
    from s2b_rag.generate import split_aspects

    answer, tiers = split_aspects(_A_REPLY, allowed=[2, 3, 15])
    assert answer.strip() == "GCC thành lập năm 2023 và được Khoa Đa phương tiện hỗ trợ chuyên môn."
    assert tiers == [[15, 3], [2]]  # đoạn ghi đầu của mỗi ý = chính; còn lại = phụ
    assert split_aspects(_A_REPLY, allowed=[2, 3])[1] == [
        [2, 3],
        [],
    ]  # 15 ngoài prompt → 2 thành đoạn chính


def test_split_aspects_falls_back_to_r9_when_format_ignored():
    from s2b_rag.generate import split_aspects

    assert split_aspects("GCC ra đời năm 2023.\nNGUỒN: 3", allowed=[3]) == (
        "GCC ra đời năm 2023.\n",
        [[3]],
    )


def test_promote_tiers_keeps_retriever_order_within_tier():
    from s2b_rag.generate import promote_cited, promote_tiers

    ranked = [5, 9, 2, 1, 7, 3]
    assert promote_tiers(ranked, [[3, 2], [9]]) == [2, 3, 9, 5, 1, 7]
    assert promote_tiers(ranked, [[2, 3]]) == promote_cited(ranked, [2, 3])


def test_prompt_a_changes_only_format_citation_and_trim_rules():
    import difflib

    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_E2

    removed = [
        d
        for d in difflib.ndiff(SYSTEM_PROMPT_E2.splitlines(), SYSTEM_PROMPT_A.splitlines())
        if d[0] == "-"
    ]
    assert len(removed) == 3  # luật số hiệu, luật bỏ lời khuyên, đoạn định dạng NGUỒN cũ (1 dòng)
    assert "TRẢ LỜI:" in SYSTEM_PROMPT_A and "Sau câu trả lời, xuống dòng" not in SYSTEM_PROMPT_A


def test_pipeline_a_puts_each_aspect_primary_in_top5_and_hides_aspects():
    from s2b_rag.pipeline import S2BPipelineA

    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    llm = FakeLLM(lambda m: _A_REPLY)
    pipeline = S2BPipelineA(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 26)))
    pipeline.index(docs)
    out = pipeline.run("câu hỏi")
    assert out.retrieved_ids[:3] == [3, 15, 2] and len(out.retrieved_ids) == 10
    assert (
        "Ý" not in out.answer.split()[0]
        and "NGUỒN" not in out.answer
        and "TRẢ LỜI" not in out.answer
    )
    assert llm.calls[0][0]["content"].endswith(
        "chỉ dùng thông tin từ các tài liệu đã ghi ở phần Ý HỎI>"
    )


def test_keep_used_sources_swaps_uncited_used_doc_into_ranks_6_to_10():
    from s2b_rag.generate import keep_used_sources

    texts = {i: f"đoạn số {i} nói chuyện khác hẳn" for i in range(1, 13)}
    texts[11] = (
        "CLB có tài khoản Instagram clbmultimedia ptit để đăng ảnh"  # bị cắt nhưng đã được dùng
    )
    texts[10] = "lịch sinh hoạt hằng tuần của câu lạc bộ"  # không đóng góp gì → bị thay
    answer = "CLB Multimedia PTIT có tài khoản Instagram clbmultimedia.ptit."
    kept = list(range(1, 11))
    out = keep_used_sources(answer, kept, candidates=[11, 12], texts=texts, protected={1})
    assert out[:5] == kept[:5]  # top-5 không đổi → Recall/MRR/NDCG@5 không đổi
    assert 11 in out[5:] and len(out) == 10
    # không có cụm từ riêng của đoạn bị cắt → giữ nguyên
    assert keep_used_sources("Không liên quan.", kept, [11], texts, set()) == kept


def test_keep_used_sources_never_drops_doc_that_supports_answer_or_is_cited():
    from s2b_rag.generate import keep_used_sources

    texts = {i: f"văn bản riêng số {i} khác" for i in range(1, 12)}
    texts[10] = "hạn đăng ký là ngày 10/09/2026 qua link"
    texts[11] = "tài khoản Instagram clbmultimedia ptit"
    answer = "Hạn đăng ký là ngày 10/09/2026, có tài khoản Instagram clbmultimedia ptit."
    kept = list(range(1, 11))
    out = keep_used_sources(answer, kept, [11], texts, protected={9})
    assert 10 in out and 9 in out  # 10 có cụm riêng, 9 được trích → không bị thay
    assert out[7] == 11 and out[8:] == [9, 10]  # thay đoạn hợp lệ cuối cùng: hạng 8


def test_keep_used_sources_ignores_club_names_already_in_kept_docs():
    from s2b_rag.generate import keep_used_sources

    # báo động giả thật ở dev (train-0011): cụm "gcc tức clb" mới, nhưng mọi TỪ đều có ở đoạn nộp
    texts = {i: f"GCC là CLB Nhà Sáng Tạo Game PTIT, tức là đoạn {i}" for i in range(1, 11)}
    texts[11] = "GCC tức CLB Nhà Sáng Tạo Game PTIT"
    answer = "GCC tức CLB Nhà Sáng Tạo Game PTIT."
    kept = list(range(1, 11))
    assert keep_used_sources(answer, kept, [11], texts, protected=set()) == kept


# --- A2: bỏ câu rào đón thuần ở cuối câu trả lời ---


def test_drop_tail_hedges_removes_pure_trailing_hedge_only():
    from s2b_rag.generate import drop_tail_hedges

    ans = "Hạn đăng ký là 10/09/2026. Tài liệu không ghi số tiền cụ thể."
    assert drop_tail_hedges(ans) == "Hạn đăng ký là 10/09/2026."
    # vế "chỉ ..." mang thông tin → giữ
    keep = "Có học vẽ khối cầu. Tài liệu không ghi rõ vẽ dáng người, chỉ có nội dung gesture."
    assert drop_tail_hedges(keep) == keep
    # rào đón nằm giữa câu có thông tin (không mở đầu bằng "Tài liệu") → giữ
    mixed = "CLB có phòng riêng. Nếu vi phạm cả nhóm bị phạt, và tài liệu không ghi hình thức phạt."
    assert drop_tail_hedges(mixed) == mixed


def test_drop_tail_hedges_keeps_refusal_when_first_sentence_hedges():
    from s2b_rag.generate import drop_tail_hedges

    refusal = "Tài liệu hiện có không có thông tin về học phí. Tài liệu không ghi số tiền cụ thể."
    assert drop_tail_hedges(refusal) == refusal  # Abstention: câu không có đáp án giữ nguyên
    assert drop_tail_hedges("Một câu.") == "Một câu."


def test_a2_applies_hedge_rule_and_keeps_a_prompt():
    from s2b_rag.generate import SYSTEM_PROMPT_A
    from s2b_rag.pipeline import S2BPipelineA2

    reply = "Ý HỎI:\nÝ 1: hạn | NGUỒN: 2\nTRẢ LỜI:\nHạn là 10/09/2026. Tài liệu không ghi số tiền cụ thể."
    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    llm = FakeLLM(lambda m: reply)
    pipeline = S2BPipelineA2(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 26)))
    pipeline.index(docs)
    assert pipeline.run("câu hỏi").answer == "Hạn là 10/09/2026."
    assert llm.calls[0][0]["content"] == SYSTEM_PROMPT_A  # cùng prompt với A → trúng cache


# --- A3: bỏ rào đón theo câu đầu + vế giữa câu + "Theo tài liệu" ---


def test_trim_hedges_cleans_informative_answer_with_hedge_in_first_sentence():
    from s2b_rag.generate import trim_hedges

    ans = "CLB thành lập ngày 09/10/2011, dù tài liệu không ghi ngày sinh nhật. Tài liệu không ghi số tiền."
    assert trim_hedges(ans) == "CLB thành lập ngày 09/10/2011."
    assert trim_hedges("Có phòng riêng. Theo tài liệu, trưởng ban là Trần Ngọc Mai.") == (
        "Có phòng riêng. Trưởng ban là Trần Ngọc Mai."
    )


def test_trim_hedges_keeps_refusal_and_contrast():
    from s2b_rag.generate import trim_hedges

    refusal = (
        "Tài liệu hiện có không có thông tin về học phí. Tài liệu chỉ ghi CLB tổ chức training."
    )
    assert trim_hedges(refusal) == refusal
    keep = "Có học vẽ khối cầu. Tài liệu không ghi rõ vẽ dáng người, chỉ có nội dung gesture."
    assert trim_hedges(keep) == keep


def test_a3_class_uses_trim_hedges():
    from s2b_rag.pipeline import S2BPipelineA3

    reply = "Ý HỎI:\nÝ 1: x | NGUỒN: 2\nTRẢ LỜI:\nCó phòng riêng cho thành viên, dù tài liệu không ghi rõ giờ mở. Tài liệu không ghi số tiền."
    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    pipeline = S2BPipelineA3(retriever="bm25_noaccent_bi", llm=FakeLLM(lambda m: reply))
    pipeline.retriever = _FixedRanker(list(range(1, 26)))
    pipeline.index(docs)
    assert pipeline.run("câu hỏi").answer == "Có phòng riêng cho thành viên."


def test_default_pipeline_is_a5_submission_config():
    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4
    from s2b_rag.pipeline import S2BPipeline, S2BPipelineA5

    pipeline = S2BPipeline(retriever="bm25_noaccent_bi", llm=FakeLLM())
    assert isinstance(pipeline, S2BPipelineA5) and pipeline.second_prompt == SYSTEM_PROMPT_A4
    assert (pipeline.n_context, pipeline.n_retrieved, pipeline.system_prompt) == (
        20,
        10,
        SYSTEM_PROMPT_A,
    )


def test_prompt_a4_only_adds_rules_and_keeps_refusal_rules():
    import difflib

    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4

    diff = list(difflib.ndiff(SYSTEM_PROMPT_A.splitlines(), SYSTEM_PROMPT_A4.splitlines()))
    removed = [d[2:] for d in diff if d[0] == "-"]
    added = [d[2:] for d in diff if d[0] == "+"]
    assert (
        len(removed) == 1 and len(added) == 4
    )  # 1 dòng định dạng Ý (vai trò + mọi nguồn) + 3 luật mới
    assert "Ý 1:" in removed[0]
    # luật từ chối và định dạng Ý HỎI/TRẢ LỜI giữ nguyên
    for keep in ("Tài liệu hiện có không có thông tin về", "Không đoán.", "Ý HỎI:", "TRẢ LỜI:"):
        assert keep in SYSTEM_PROMPT_A4
    assert "không suy ra quy định, lịch hay con số mà tài liệu không ghi" in SYSTEM_PROMPT_A4


def test_a4_class_uses_a4_prompt_and_default_answers_with_prompt_a():
    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4
    from s2b_rag.pipeline import S2BPipeline, S2BPipelineA2, S2BPipelineA4

    a4 = S2BPipelineA4(retriever="bm25_noaccent_bi", llm=FakeLLM())
    assert isinstance(a4, S2BPipelineA2)
    assert (a4.n_context, a4.n_retrieved, a4.system_prompt) == (20, 10, SYSTEM_PROMPT_A4)
    assert S2BPipeline(retriever="bm25_noaccent_bi", llm=FakeLLM()).system_prompt == SYSTEM_PROMPT_A


def test_merge_tiers_unions_primary_and_secondary_and_treats_single_tier_as_primary():
    from s2b_rag.generate import merge_tiers, promote_tiers

    assert merge_tiers([[3], [5]], [[7, 3], [9]]) == [[3, 7, 3], [5, 9]]
    assert merge_tiers([[4, 6]], [[], []]) == [[4, 6], []]  # 1 tầng (lỗi định dạng) → tầng chính
    ranked = list(range(1, 12))
    assert promote_tiers(ranked, merge_tiers([[3], [5]], [[7, 3], [9]]))[:5] == [3, 7, 5, 9, 1]


def test_a5_keeps_first_answer_and_merges_sources_of_second_read():
    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4
    from s2b_rag.pipeline import S2BPipelineA5

    first = "Ý HỎI:\nÝ 1: x | NGUỒN: 4\nTRẢ LỜI:\nCâu trả lời lần một."
    second = "Ý HỎI:\nÝ 1: x | NGUỒN: 4\nÝ 2: y | NGUỒN: 12, 8\nTRẢ LỜI:\nCâu trả lời lần hai."
    replies = {SYSTEM_PROMPT_A: first, SYSTEM_PROMPT_A4: second}
    llm = FakeLLM(lambda m: replies[m[0]["content"]])
    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    pipeline = S2BPipelineA5(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 26)))
    pipeline.index(docs)
    out = pipeline.run("câu hỏi")
    assert len(llm.calls) == 2  # lần đọc thứ 2 dùng prompt A4
    assert out.answer == "Câu trả lời lần một."  # câu trả lời của lần 1 (= A2)
    assert out.retrieved_ids[:5] == [4, 12, 8, 1, 2] and len(out.retrieved_ids) == 10


def test_a2_still_calls_llm_once():
    from s2b_rag.pipeline import S2BPipelineA2

    llm = FakeLLM(lambda m: "Ý HỎI:\nÝ 1: x | NGUỒN: 4\nTRẢ LỜI:\nCâu trả lời.")
    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    pipeline = S2BPipelineA2(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 26)))
    pipeline.index(docs)
    pipeline.run("câu hỏi")
    assert len(llm.calls) == 1


def test_a6_third_read_only_fills_free_top5_slots():
    from s2b_rag.generate import SYSTEM_PROMPT_A, SYSTEM_PROMPT_A4, SYSTEM_PROMPT_E2
    from s2b_rag.pipeline import S2BPipelineA6

    replies = {
        SYSTEM_PROMPT_A: "Ý HỎI:\nÝ 1: x | NGUỒN: 4\nTRẢ LỜI:\nCâu trả lời lần một.",
        SYSTEM_PROMPT_A4: "Ý HỎI:\nÝ 1: x | NGUỒN: 4, 9\nTRẢ LỜI:\nLần hai.",
        SYSTEM_PROMPT_E2: "Lần ba.\nNGUỒN: 17, 4",
    }
    llm = FakeLLM(lambda m: replies[m[0]["content"]])
    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    pipeline = S2BPipelineA6(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 26)))
    pipeline.index(docs)
    out = pipeline.run("câu hỏi")
    assert len(llm.calls) == 3
    assert out.answer == "Câu trả lời lần một."
    # chính [4] → phụ [9] → tầng 3 chỉ thêm 17 (4 đã có) → rồi retriever
    assert out.retrieved_ids[:5] == [4, 9, 17, 1, 2] and len(out.retrieved_ids) == 10


# ---- Chatbot: retrieve / stream_answer / finish_answer (không đụng run()) ----


class FakeStreamClient:
    """Client giả kiểu openai: `chat.completions.create(..., stream=True)` nhả từng mẩu chữ."""

    def __init__(self, pieces):
        self.pieces, self.calls = pieces, []
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)

        def chunk(text):
            delta = type("Delta", (), {"content": text})()
            return type("Chunk", (), {"choices": [type("Choice", (), {"delta": delta})()]})()

        return [chunk(p) for p in self.pieces] + [type("Chunk", (), {"choices": []})()]


def _stream(pieces, wait_head=True):
    from s2b_rag.generate import AnswerStreamFilter

    f = AnswerStreamFilter(wait_head)
    return "".join(f.feed(p) for p in pieces) + f.finish()


def test_stream_filter_hides_aspects_and_sources_even_when_split_mid_word():
    pieces = ["Ý HỎI:\nÝ 1: x | NGU", "ỒN: 4\nTRẢ L", "ỜI:\nCó. CLB ", "đạt giải.\nNGU", "ỒN: 4"]
    assert _stream(pieces) == "Có. CLB đạt giải.\n"
    assert (
        _stream(["Trả lời ", "một dòng.\n", "NGUỒN: 1, 2"], wait_head=False)
        == "Trả lời một dòng.\n"
    )
    assert (
        _stream(["Ý HỎI:\nÝ 1: x | NGUỒN: 4\n", "quên TRẢ LỜI"]) == ""
    )  # lỗi định dạng: không lộ gì
    # câu bắt đầu bằng "Nguồn" nhưng không phải dòng nguồn vẫn được hiện
    assert _stream(
        ["Ý HỎI:\nÝ 1: x | NGUỒN: 4\n**TRẢ LỜI:**\n", "Nguồn tin cho biết CLB tuyển."]
    ) == ("Nguồn tin cho biết CLB tuyển.")


def test_chatbot_stream_matches_submission_answer_and_prompt():
    from s2b_rag.pipeline import S2BPipeline

    raw = (
        "Ý HỎI:\nÝ 1: x | NGUỒN: 7\nÝ 2: y | NGUỒN: 3, 12\nTRẢ LỜI:\n"
        "**Có**, CLB có phòng riêng. Tài liệu không ghi giờ mở cửa."
    )
    llm = FakeLLM(lambda m: raw)
    docs = [Document(i, "CLB Lập Trình PTIT", "Chủ đề", f"Đoạn {i}.") for i in range(1, 26)]
    pipeline = S2BPipeline(retriever="bm25_noaccent_bi", llm=llm)
    pipeline.retriever = _FixedRanker(list(range(1, 26)))
    pipeline.index(docs)
    pipeline.stream_client = client = FakeStreamClient([raw[:9], raw[9:40], raw[40:70], raw[70:]])

    expected = pipeline.run("câu hỏi").answer  # bài nộp
    r = pipeline.retrieve("câu hỏi")
    stream = pipeline.stream_answer("câu hỏi", r.context)
    visible = "".join(stream)
    answer, cited = pipeline.finish_answer(stream.raw, r.context)

    assert len(r.context) == 20 and len(r.docs) == 10  # haiku đọc 20 đoạn như bài nộp
    assert (
        "Ý HỎI" not in visible and "NGUỒN" not in visible and visible.lstrip("*").startswith("Có")
    )
    assert answer == expected == "Có, CLB có phòng riêng."  # cùng hậu xử lý G2 + A2 với bài nộp
    assert cited == [7, 3, 12]
    (call,) = client.calls
    assert call["messages"] == llm.calls[0]  # cùng prompt với lần gọi 1 của bài nộp
    assert (
        call["stream"] is True and call["model"] == "claude-haiku-5-5" and "temperature" not in call
    )
