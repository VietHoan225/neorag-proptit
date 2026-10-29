"""Gộp nhiều kênh truy xuất bằng Reciprocal Rank Fusion (RRF).

RRF(d) = Σ_kênh w / (k + rank_kênh(d)), rank bắt đầu từ 1. Chỉ dùng THỨ HẠNG, không dùng điểm gốc,
nên không cần đưa điểm BM25 (0..~30) và cosine (0..1) về cùng thang. k (thường 60) làm phẳng chênh
lệch giữa các hạng đầu: một kênh xếp hạng 1 không át hẳn kênh khác xếp hạng 3.
"""

from __future__ import annotations

from collections import defaultdict

from neorag.data import Document
from s2b_rag.textnorm import has_diacritics


def rrf(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])  # sorted ổn định: hoà điểm giữ thứ tự gặp


class HybridRetriever:
    """Gộp các kênh bằng RRF. `noaccent_weights` (tuỳ chọn) thay `weights` khi câu hỏi viết không dấu:
    đo trên dev, dense bge-m3 tụt mạnh với câu không dấu (0.708 -> 0.498) nên cần giảm tiếng nói của nó."""

    def __init__(
        self,
        retrievers: list,
        k: int = 60,
        weights: list[float] | None = None,
        noaccent_weights: list[float] | None = None,
    ):
        self.retrievers, self.k = retrievers, k
        self.weights, self.noaccent_weights = weights, noaccent_weights

    def index(self, corpus: list[Document]) -> None:
        for r in self.retrievers:
            r.index(corpus)

    def search(self, query: str) -> list[tuple[int, float]]:
        rankings = [[doc_id for doc_id, _ in r.search(query)] for r in self.retrievers]
        weights = self.weights
        if self.noaccent_weights is not None and not has_diacritics(query):
            weights = self.noaccent_weights
        return rrf(rankings, self.k, weights)
