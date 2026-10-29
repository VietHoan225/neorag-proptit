from neorag.data import Document
from s2b_rag.retrieval.rerank import CrossEncoderReranker


class Fixed:
    def __init__(self, ranking):
        self.ranking = ranking

    def index(self, corpus):
        pass

    def search(self, query):
        return [(doc_id, 0.0) for doc_id in self.ranking]


class FakeReranker(CrossEncoderReranker):
    """Không nạp model thật: điểm = độ dài văn bản đoạn (đoạn dài hơn được xếp trước)."""

    calls = 0

    def score(self, query, doc_ids):
        FakeReranker.calls += 1
        return [float(len(self.texts[i])) for i in doc_ids]


CORPUS = [Document(i, "c", "t", "x" * length) for i, length in [(1, 1), (2, 3), (3, 2), (4, 9)]]


def test_reranks_head_and_keeps_tail_after_it():
    r = FakeReranker(Fixed([1, 2, 3, 4]), top_n=3)
    r.index(CORPUS)
    results = r.search("câu hỏi có dấu")
    assert [i for i, _ in results] == [2, 3, 1, 4]  # top-3 xếp lại theo điểm, đoạn 4 vẫn ở đuôi
    assert results[2][1] > results[3][1]  # điểm đuôi thấp hơn mọi điểm đã rerank


def test_skips_queries_without_diacritics():
    FakeReranker.calls = 0
    r = FakeReranker(Fixed([1, 2, 3, 4]), top_n=3)
    r.index(CORPUS)
    assert [i for i, _ in r.search("cau hoi khong dau")] == [1, 2, 3, 4]
    assert FakeReranker.calls == 0  # không tốn thời gian chấm


def test_can_rerank_all_queries_when_asked():
    r = FakeReranker(Fixed([1, 2, 3, 4]), top_n=4, only_with_diacritics=False)
    r.index(CORPUS)
    assert [i for i, _ in r.search("cau hoi khong dau")][0] == 4
