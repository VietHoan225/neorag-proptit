import pytest

from neorag.data import Document
from s2b_rag.retrieval.bm25 import BM25, BM25Retriever
from s2b_rag.retrieval.fusion import HybridRetriever, rrf
from s2b_rag.textnorm import analyze, expand_query


def test_bm25_rare_term_beats_common_term():
    bm25 = BM25()
    bm25.fit([["clb", "game"], ["clb", "multimedia"], ["clb", "lap", "trinh"]])
    scores = bm25.scores(["clb", "game"])
    assert scores.argmax() == 0
    assert bm25.idf["game"] > bm25.idf["clb"]  # "clb" có ở mọi đoạn nên ít giá trị phân biệt
    assert bm25.scores(["unknown"]).tolist() == [0.0, 0.0, 0.0]


def test_noaccent_channel_matches_query_without_diacritics(corpus):
    retriever = BM25Retriever(lambda t: analyze(t, accents=False), expand_query)
    retriever.index(corpus)
    assert retriever.search("clb multimedia thanh lap ngay nao")[0][0] == 2


def test_rrf_combines_ranks():
    fused = rrf([[1, 2, 3], [2, 3, 1]], k=60)
    assert [doc_id for doc_id, _ in fused] == [2, 1, 3]  # 2 đứng cao ở cả hai kênh
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_indexes_all_channels():
    class Fixed:
        def __init__(self, ranking):
            self.ranking, self.indexed = ranking, False

        def index(self, corpus):
            self.indexed = True

        def search(self, query):
            return [(i, 0.0) for i in self.ranking]

    channels = [Fixed([1, 2]), Fixed([2, 1])]
    hybrid = HybridRetriever(channels)
    hybrid.index([Document(1, "", "", ""), Document(2, "", "", "")])
    assert all(c.indexed for c in channels)
    assert {doc_id for doc_id, _ in hybrid.search("q")} == {1, 2}


def test_hybrid_switches_weights_for_queries_without_diacritics():
    class Fixed:
        def __init__(self, ranking):
            self.ranking = ranking

        def index(self, corpus):
            pass

        def search(self, query):
            return [(i, 0.0) for i in self.ranking]

    hybrid = HybridRetriever(
        [Fixed([1, 2]), Fixed([2, 1])], weights=[1.0, 0.5], noaccent_weights=[0.2, 1.0]
    )
    assert hybrid.search("có dấu")[0][0] == 1  # kênh 1 nặng hơn
    assert hybrid.search("khong dau")[0][0] == 2  # kênh 2 nặng hơn
