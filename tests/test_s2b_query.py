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
