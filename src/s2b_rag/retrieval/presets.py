"""Các cấu hình retrieval có tên, dùng CHUNG cho script đo (`s2b_rag.eval.retrieval`) và pipeline nộp bài
(`s2b_rag.pipeline`) — để con số đo được và bài nộp chắc chắn cùng một cấu hình.

Lịch sử quyết định và số liệu: docs/adr/0003, docs/adr/0004, experiments/log.csv.
"""

from __future__ import annotations

from collections.abc import Callable

from s2b_rag.aliases import enrich
from s2b_rag.retrieval.bm25 import BM25Retriever
from s2b_rag.retrieval.club_boost import ClubBoostRetriever, EnrichedRetriever
from s2b_rag.retrieval.dense import gateway_dense
from s2b_rag.retrieval.fusion import HybridRetriever
from s2b_rag.retrieval.multiquery import MultiQueryRetriever
from s2b_rag.retrieval.rerank import CrossEncoderReranker
from s2b_rag.textnorm import analyze, expand_query

DEFAULT = "hybrid_rerank"  # cấu hình đang dùng cho bài nộp


def bm25(accents: bool, n_max: int = 1, expand: bool = True) -> BM25Retriever:
    """Một kênh BM25: có dấu hoặc bỏ dấu, n-gram tới n_max, có/không mở rộng viết tắt (R1)."""
    return BM25Retriever(
        lambda text: analyze(text, accents, n_max), expand_query if expand else str
    )


def hybrid_equal() -> HybridRetriever:
    """R3 ban đầu: RRF đều 3 kênh dense + BM25 có dấu + BM25 bỏ dấu (giữ để so sánh)."""
    return HybridRetriever(
        [gateway_dense(), bm25(accents=True, n_max=2), bm25(accents=False, n_max=2)]
    )


def hybrid(enriched: bool = False) -> HybridRetriever:
    """R3 chốt: dense + BM25 bỏ dấu (cụm 2 âm tiết); câu không dấu thì trọng số dense 0.2.
    0.2 nằm giữa vùng ổn định 0.1-0.3 khi dò trên dev (ADR-0003).
    enriched=True (R4): cả hai kênh index văn bản có tiền tố CLB + alias + chủ đề."""
    channels = [gateway_dense(), bm25(accents=False, n_max=2)]
    if enriched:
        channels = [EnrichedRetriever(c) for c in channels]
    return HybridRetriever(channels, weights=[1.0, 1.0], noaccent_weights=[0.2, 1.0])


def hybrid_club() -> ClubBoostRetriever:
    """R4 + R5 chốt: hybrid làm giàu index + boost nhẹ đoạn của CLB được nhắc (0.1) + quota top-5.
    Dò boost {0, 0.1, 0.25, 0.5, 1}: 0-0.25 tương đương, >=0.5 kém hơn -> chọn 0.1."""
    return ClubBoostRetriever(hybrid(enriched=True), weight=0.1)


def hybrid_rerank(top_n: int = 20, only_with_diacritics: bool = True) -> CrossEncoderReranker:
    """R6 chốt: hybrid_club + cross-encoder bge-reranker-v2-m3 trên top-20, chỉ cho câu có dấu.
    Đầu vào reranker: câu hỏi đã mở rộng viết tắt × đoạn đã làm giàu (tiền tố CLB + chủ đề)."""
    return CrossEncoderReranker(
        hybrid_club(),
        top_n=top_n,
        query_fn=expand_query,
        doc_text=lambda d: enrich(d.club, d.topic, d.text),
        only_with_diacritics=only_with_diacritics,
    )


# Tên retriever -> hàm tạo. Mỗi kỹ thuật mới thêm một dòng ở đây.
RETRIEVERS: dict[str, Callable] = {
    "dense": gateway_dense,  # baseline BTC
    "bm25": lambda: bm25(accents=True),  # BM25 có dấu + R1
    "bm25_noaccent": lambda: bm25(accents=False),  # R2 kênh bỏ dấu
    "bm25_noaccent_bi": lambda: bm25(accents=False, n_max=2),  # + cụm 2 âm tiết
    "hybrid_equal": hybrid_equal,  # R3 đều 3 kênh
    "hybrid": hybrid,  # R3 chốt
    "hybrid_enrich": lambda: hybrid(enriched=True),  # + R4
    "hybrid_club": hybrid_club,  # + R4 + R5
    "hybrid_rerank_all": lambda: hybrid_rerank(only_with_diacritics=False),  # R6 rerank mọi câu
    "hybrid_rerank": hybrid_rerank,  # + R6 chỉ câu có dấu (chốt hiện tại)
    # R8 tách ý theo luật: làm GIẢM điểm (câu con mất nghĩa do "đó", "vậy"...) → không dùng, giữ để tái lập
    "hybrid_rerank_r8rule": lambda: MultiQueryRetriever(hybrid_rerank(), sub_weight=0.5),
}
