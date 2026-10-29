"""Các cấu hình retrieval có tên, dùng CHUNG cho script đo (`s2b_rag.eval.retrieval`) và pipeline nộp bài
(`s2b_rag.pipeline`) — để con số đo được và bài nộp chắc chắn cùng một cấu hình.

Lịch sử quyết định và số liệu: docs/adr/0003, docs/adr/0004, experiments/log.csv.
"""

from __future__ import annotations

from collections.abc import Callable

from neorag.llm import LLMClient
from s2b_rag.aliases import detect_clubs, detect_clubs_multi, enrich
from s2b_rag.query import LLMQueryRewriter
from s2b_rag.retrieval.bm25 import BM25Retriever
from s2b_rag.retrieval.club_boost import ClubBoostRetriever, EnrichedRetriever
from s2b_rag.retrieval.dense import CACHE_PATH, gateway_dense
from s2b_rag.retrieval.fusion import HybridRetriever
from s2b_rag.retrieval.multiquery import MultiQueryRetriever
from s2b_rag.retrieval.rerank import CrossEncoderReranker
from s2b_rag.retrieval.rewrite import RewriteRetriever
from s2b_rag.textnorm import analyze, expand_query, has_diacritics

DEFAULT = "hybrid_r10"  # cấu hình nộp bài (public #2: hybrid_r7; trước 29/10 chiều: hybrid_rerank)


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


def hybrid_club(detect=detect_clubs) -> ClubBoostRetriever:
    """R4 + R5 chốt: hybrid làm giàu index + boost nhẹ đoạn của CLB được nhắc (0.1) + quota top-5.
    Dò boost {0, 0.1, 0.25, 0.5, 1}: 0-0.25 tương đương, >=0.5 kém hơn -> chọn 0.1."""
    return ClubBoostRetriever(hybrid(enriched=True), weight=0.1, detect=detect)


def hybrid_rerank(
    top_n: int = 20, only_with_diacritics: bool = True, detect=detect_clubs
) -> CrossEncoderReranker:
    """R6 chốt: hybrid_club + cross-encoder bge-reranker-v2-m3 trên top-20, chỉ cho câu có dấu.
    Đầu vào reranker: câu hỏi đã mở rộng viết tắt × đoạn đã làm giàu (tiền tố CLB + chủ đề)."""
    return CrossEncoderReranker(
        hybrid_club(detect),
        top_n=top_n,
        query_fn=expand_query,
        doc_text=lambda d: enrich(d.club, d.topic, d.text),
        only_with_diacritics=only_with_diacritics,
    )


def hybrid_rewrite(use_subqueries: bool = False, sub_weight: float = 0.5) -> RewriteRetriever:
    """R7: hybrid_rerank trên câu hỏi do claude-haiku-5-5 viết lại (chỉ câu không dấu hoặc nhiều ý)."""
    rewriter = LLMQueryRewriter(LLMClient(cache_path=CACHE_PATH))
    return RewriteRetriever(hybrid_rerank(), rewriter, use_subqueries, sub_weight)


def hybrid_r7(original_weight: float = 0.5) -> RewriteRetriever:
    """R7 chốt: chỉ câu KHÔNG DẤU được haiku viết lại thành câu có dấu (câu có dấu viết lại thì giảm nhẹ);
    gộp RRF câu viết lại (w=1) + câu gốc (w=0.5, giữa vùng ổn định 0.25-1.0) để phòng LLM làm mất ý;
    không dùng câu con (không giúp thêm). Dev: public-like 0.867 -> 0.914."""
    rewriter = LLMQueryRewriter(LLMClient(cache_path=CACHE_PATH))
    return RewriteRetriever(
        hybrid_rerank(),
        rewriter,
        should_rewrite=lambda q: not has_diacritics(q),
        original_weight=original_weight,
    )


def hybrid_r10() -> RewriteRetriever:
    """R10 chốt: hybrid_r7 + quota CLB đặt SAU reranker (reranker xếp lại top-20 nên xoá quota R5 đặt trước nó)
    + câu "ba CLB" coi như nhắc cả ba CLB. Chẩn đoán: ~80% lỗi top-5 là đoạn đúng có trong top-20 nhưng xếp sai,
    nửa trong đó là câu hỏi về cả ba CLB. Public-like dev 0.9137 -> 0.9155, holdout 0.8988 -> 0.9093 (holdout đã
    bị nhìn khi thiết kế luật -> bằng chứng yếu, xem docs/journal.md)."""
    rewriter = LLMQueryRewriter(LLMClient(cache_path=CACHE_PATH))
    reranked = hybrid_rerank(detect=detect_clubs_multi)
    quota = ClubBoostRetriever(reranked, weight=0.0, detect=detect_clubs_multi)
    return RewriteRetriever(
        quota, rewriter, should_rewrite=lambda q: not has_diacritics(q), original_weight=0.5
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
    "hybrid_rewrite": hybrid_rewrite,  # R7 chỉ khôi phục dấu (GỌI HAIKU)
    "hybrid_rewrite_sub": lambda: hybrid_rewrite(use_subqueries=True),  # R7 + câu con (GỌI HAIKU)
    "hybrid_r7": hybrid_r7,  # R7 chốt: chỉ câu không dấu, + câu gốc w=0.5 (GỌI HAIKU)
    "hybrid_r10": hybrid_r10,  # + quota sau reranker + "ba CLB" (GỌI HAIKU, như R7)
}
