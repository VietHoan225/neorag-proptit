"""BM25 (Okapi): xếp hạng theo mức trùng từ khoá, có trọng số độ hiếm (IDF) và chuẩn hoá độ dài.

score(q, d) = Σ_{t∈q} IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
IDF(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np

from neorag.data import Document


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b  # k1: độ bão hoà khi từ lặp lại; b: mức phạt đoạn dài

    def fit(self, docs: list[list[str]]) -> None:
        self.tfs = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if docs else 0.0
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        out = np.zeros(len(self.tfs))
        for term in set(query):  # từ lặp trong câu hỏi không cộng nhiều lần
            if term not in self.idf:
                continue
            tf = np.array([d[term] for d in self.tfs], dtype=np.float64)
            out += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out


class BM25Retriever:
    """Một kênh BM25. `analyzer` biến văn bản thành term (dùng chung cho đoạn văn và câu hỏi),
    `query_fn` chuẩn hoá câu hỏi trước khi phân tích (vd mở rộng viết tắt)."""

    def __init__(
        self,
        analyzer: Callable[[str], list[str]],
        query_fn: Callable[[str], str] = str,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.analyzer, self.query_fn = analyzer, query_fn
        self.bm25 = BM25(k1, b)

    def index(self, corpus: list[Document]) -> None:
        self.ids = [d.id for d in corpus]
        self.bm25.fit([self.analyzer(d.text) for d in corpus])

    def search(self, query: str) -> list[tuple[int, float]]:
        scores = self.bm25.scores(self.analyzer(self.query_fn(query)))
        order = np.argsort(-scores, kind="stable")
        return [(self.ids[i], float(scores[i])) for i in order]
