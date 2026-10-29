import csv

import pytest

from conftest import FakeEmbedder
from s2b_rag.eval import explog
from s2b_rag.eval.retrieval import evaluate_retriever, summarize, summarize_by
from s2b_rag.retrieval.dense import DenseRetriever


class FixedRetriever:
    """Retriever giả: luôn trả cùng một thứ hạng."""

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query):
        return [(doc_id, 1.0) for doc_id in self.ranking]


def test_dense_retriever_ranks_whole_corpus(corpus):
    retriever = DenseRetriever(FakeEmbedder())
    retriever.index(corpus)
    results = retriever.search("CLB Lập Trình PTIT thành lập ngày 09/10/2011.")
    assert sorted(doc_id for doc_id, _ in results) == [1, 2, 3]
    assert results[0][0] == 1
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_evaluate_matches_official_rules():
    rows = [
        {"id": "a", "query": "q", "relevant_ids": [2], "type": "t1"},
        {"id": "b", "query": "q", "relevant_ids": [9], "type": "t2"},
        {"id": "c", "query": "q", "relevant_ids": [], "type": "t2", "answerable": False},
    ]
    results = evaluate_retriever(FixedRetriever([1, 2, 3]), rows)
    assert results[0]["mrr@5"] == pytest.approx(0.5)  # doc đúng ở vị trí 2
    assert results[1]["retrieval"] == 0.0  # không tìm thấy
    assert "retrieval" not in results[2]  # không có relevant_ids thì không chấm

    summary = summarize(results)
    assert summary["n"] == 2
    assert summary["recall@5"] == pytest.approx(0.5)
    assert set(summarize_by(results, "type")) == {"t1", "t2"}


def test_explog_writes_header_once(tmp_path):
    path = tmp_path / "log.csv"
    explog.append({"system": "dense", "split": "dev", "retrieval": 0.123456}, path)
    explog.append({"system": "dense", "split": "dev_noaccent"}, path)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["split"] for r in rows] == ["dev", "dev_noaccent"]
    assert rows[0]["retrieval"] == "0.1235"
    assert rows[0]["commit"]
