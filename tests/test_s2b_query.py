from s2b_rag.query import split_subqueries
from s2b_rag.retrieval.multiquery import MultiQueryRetriever
from s2b_rag.textnorm import has_diacritics


def test_single_question_is_not_split():
    assert split_subqueries("CLB Lập Trình PTIT thành lập ngày nào?") == []
    assert split_subqueries("gcc la gi, a?") == []  # câu con quá ngắn bị bỏ


def test_orphan_clause_borrows_club_without_adding_diacritics():
    subs = split_subqueries("truong ban su kien proptit bay gio la ai, nk truoc do la ai")
    assert subs[0] == "truong ban su kien proptit bay gio la ai"
    assert subs[1].startswith("nk truoc do la ai") and "Lap Trinh" in subs[1]
    assert not any(has_diacritics(s) for s in subs)


def test_accented_orphan_clause_gets_accented_club_name():
    subs = split_subqueries("ProPTIT tuyển thành viên khi nào, vòng phỏng vấn kéo dài bao lâu?")
    assert subs[1].endswith("CLB Lập Trình PTIT")


class Fixed:
    def __init__(self, table):
        self.table = table

    def index(self, corpus):
        pass

    def search(self, query):
        return [(i, 0.0) for i in self.table.get(query, [1, 2, 3])]


def test_multiquery_fuses_full_query_and_subqueries():
    table = {"a b c, d e f": [1, 2, 3], "a b c": [1, 2, 3], "d e f": [3, 1, 2]}
    r = MultiQueryRetriever(Fixed(table), sub_weight=1.0)
    ranked = [i for i, _ in r.search("a b c, d e f")]
    assert ranked[:2] == [1, 3]  # đoạn 3 (đứng đầu câu con thứ hai) được kéo lên
    assert sorted(ranked) == [1, 2, 3]


def test_multiquery_returns_base_result_for_single_question():
    r = MultiQueryRetriever(Fixed({}))
    assert [i for i, _ in r.search("chỉ một ý thôi")] == [1, 2, 3]


from conftest import FakeLLM  # noqa: E402
from s2b_rag.query import LLMQueryRewriter, needs_rewrite, parse_rewrite  # noqa: E402
from s2b_rag.retrieval.rewrite import RewriteRetriever  # noqa: E402


def test_parse_rewrite_accepts_fenced_json_and_falls_back():
    reply = '```json\n{"query": "Câu lạc bộ có phòng riêng không?", "subqueries": []}\n```'
    assert parse_rewrite(reply, "x").query == "Câu lạc bộ có phòng riêng không?"
    bad = parse_rewrite("xin lỗi, tôi không hiểu", "clb mm co phong ko")
    assert bad.query == "clb mm co phong ko" and not bad.ok
    one = parse_rewrite('{"query": "A?", "subqueries": ["chỉ một câu"]}', "a")
    assert one.subqueries == []  # một câu con thì không tách


def test_needs_rewrite_only_for_no_diacritics_or_multi_clause():
    assert needs_rewrite("clb mm co phong rieng ko")
    assert needs_rewrite("ProPTIT có 3 bài hát truyền thống, tên của mấy bài đó là gì vậy?")
    assert not needs_rewrite("CLB Lập Trình PTIT thành lập ngày nào?")


def test_rewrite_retriever_uses_rewritten_query_and_subqueries():
    reply = '{"query": "q có dấu", "subqueries": ["con một", "con hai"]}'
    llm = FakeLLM(lambda m: reply)
    table = {"q có dấu": [1, 2, 3], "con một": [1, 2, 3], "con hai": [3, 2, 1]}
    r = RewriteRetriever(Fixed(table), LLMQueryRewriter(llm), use_subqueries=True, sub_weight=1.0)
    ranked = [i for i, _ in r.search("q khong dau")]
    assert ranked[0] in (1, 3) and sorted(ranked) == [1, 2, 3]
    assert r.last_rewrite.subqueries == ["con một", "con hai"]
    assert len(llm.calls) == 1 and "temperature" not in str(llm.calls)


def test_rewrite_retriever_skips_llm_for_simple_accented_query():
    llm = FakeLLM(lambda m: "{}")
    r = RewriteRetriever(Fixed({}), LLMQueryRewriter(llm))
    assert [i for i, _ in r.search("CLB thành lập ngày nào")] == [1, 2, 3]
    assert llm.calls == []  # không tốn lần gọi nào


def test_rewrite_retriever_can_fuse_original_query():
    llm = FakeLLM(lambda m: '{"query": "câu có dấu", "subqueries": []}')
    table = {"câu có dấu": [1, 2, 3], "cau goc": [3, 2, 1]}
    r = RewriteRetriever(
        Fixed(table), LLMQueryRewriter(llm), should_rewrite=lambda q: True, original_weight=1.0
    )
    ranked = [i for i, _ in r.search("cau goc")]
    assert sorted(ranked) == [1, 2, 3] and ranked[0] in (1, 3)  # cả hai thứ hạng cùng góp phiếu
