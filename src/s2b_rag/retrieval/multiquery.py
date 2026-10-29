"""R8: tìm bằng cả câu hỏi và từng câu con, gộp các thứ hạng bằng RRF.

Câu gốc luôn được tìm (trọng số 1) vì dấu phẩy không phải lúc nào cũng tách hai câu hỏi ("thích chụp ảnh,
quay phim thì..."); câu con chỉ bổ sung, trọng số `sub_weight` mỗi câu.
"""

from __future__ import annotations

from collections.abc import Callable

from neorag.data import Document
from s2b_rag.query import split_subqueries
from s2b_rag.retrieval.fusion import rrf


class MultiQueryRetriever:
    def __init__(
        self,
        base,
        sub_weight: float = 0.5,
        k: int = 60,
        splitter: Callable[[str], list[str]] = split_subqueries,
    ):
        self.base, self.sub_weight, self.k, self.splitter = base, sub_weight, k, splitter

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)

    def search(self, query: str) -> list[tuple[int, float]]:
        full = self.base.search(query)
        subqueries = self.splitter(query)
        if not subqueries:
            return full
        rankings = [[i for i, _ in full]]
        rankings += [[i for i, _ in self.base.search(q)] for q in subqueries]
        weights = [1.0] + [self.sub_weight] * len(subqueries)
        return rrf(rankings, self.k, weights)
