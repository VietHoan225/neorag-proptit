"""R7: truy xuất bằng câu hỏi đã được LLM viết lại (có dấu, câu con tự đủ nghĩa).

Câu không cần viết lại (`should_rewrite` = False) đi thẳng vào `base`, không tốn lần gọi nào. Câu được viết
lại: tìm bằng câu đã có dấu (w=1); tuỳ chọn gộp RRF thêm câu gốc (`original_weight`, phòng LLM làm mất ý,
vd bỏ mất "Chuyên Tin") và các câu con (`use_subqueries`, mỗi câu `sub_weight`).
"""

from __future__ import annotations

from collections.abc import Callable

from neorag.data import Document
from s2b_rag.query import needs_rewrite
from s2b_rag.retrieval.fusion import rrf


class RewriteRetriever:
    def __init__(
        self,
        base,
        rewriter,
        use_subqueries: bool = False,
        sub_weight: float = 0.5,
        k: int = 60,
        should_rewrite: Callable[[str], bool] = needs_rewrite,
        original_weight: float = 0.0,
    ):
        self.base, self.rewriter = base, rewriter
        self.use_subqueries, self.sub_weight, self.k = use_subqueries, sub_weight, k
        self.should_rewrite, self.original_weight = should_rewrite, original_weight
        self.last_rewrite = None  # để pipeline/chatbot xem câu đã viết lại

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)

    def search(self, query: str) -> list[tuple[int, float]]:
        self.last_rewrite = None
        if not self.should_rewrite(query):
            return self.base.search(query)
        rw = self.last_rewrite = self.rewriter.rewrite(query)
        main = self.base.search(rw.query)
        queries, weights = [], []
        if self.original_weight > 0 and rw.query != query:
            queries.append(query)
            weights.append(self.original_weight)
        if self.use_subqueries:
            queries += rw.subqueries
            weights += [self.sub_weight] * len(rw.subqueries)
        if not queries:
            return main
        rankings = [[i for i, _ in main]] + [[i for i, _ in self.base.search(q)] for q in queries]
        return rrf(rankings, self.k, [1.0, *weights])
