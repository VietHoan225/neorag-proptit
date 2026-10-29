"""R6: rerank bằng cross-encoder chạy local (BAAI/bge-reranker-v2-m3).

Bi-encoder (dense) embed câu hỏi và đoạn văn RIÊNG rồi so vector → nhanh, nhưng không "nhìn" hai bên cùng
lúc. Cross-encoder đưa cặp (câu hỏi, đoạn) vào CÙNG một lượt transformer → chấm độ liên quan chính xác hơn,
nhưng chậm (mỗi cặp một lượt) → chỉ chấm top-N ứng viên từ tầng retrieve. Corpus nhỏ (118 đoạn) nên N lớn
vẫn chấp nhận được.

Cần cài: python scripts/install.py --local. Model nạp lười (lần search đầu) và chỉ nạp một lần.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from neorag.data import Document
from s2b_rag.textnorm import has_diacritics

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str = RERANK_MODEL, max_length: int = 512):
    """Nạp model một lần cho cả tiến trình; GPU thì chạy fp16 (nhanh gấp ~2, đủ chính xác để xếp hạng)."""
    import torch
    from sentence_transformers import CrossEncoder

    from neorag.embeddings import is_downloaded

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
        local_files_only=is_downloaded(model_name),  # đã có trong cache HF thì không gọi mạng
    )
    if device == "cuda":
        model.half()
    model.eval()
    return model


class CrossEncoderReranker:
    """Lấy kết quả của `base`, chấm lại `top_n` ứng viên đầu bằng cross-encoder, phần còn lại giữ nguyên
    thứ tự phía sau (để vẫn trả đủ thứ hạng toàn corpus).

    `only_with_diacritics`: bge-reranker-v2-m3 rất yếu với câu hỏi không dấu (đo trên dev: rerank làm
    dev_noaccent tụt 0.829 -> 0.71-0.83 ở mọi cấu hình) nên mặc định chỉ rerank câu có dấu."""

    def __init__(
        self,
        base,
        top_n: int = 30,
        query_fn: Callable[[str], str] = str,
        doc_text: Callable[[Document], str] = lambda d: d.text,
        model_name: str = RERANK_MODEL,
        batch_size: int = 32,
        only_with_diacritics: bool = True,
    ):
        self.base, self.top_n, self.query_fn, self.doc_text = base, top_n, query_fn, doc_text
        self.model_name, self.batch_size = model_name, batch_size
        self.only_with_diacritics = only_with_diacritics

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)
        self.texts = {d.id: self.doc_text(d) for d in corpus}

    def score(self, query: str, doc_ids: list[int]) -> list[float]:
        model = load_cross_encoder(self.model_name)
        q = self.query_fn(query)
        pairs = [(q, self.texts[i]) for i in doc_ids]
        return [float(s) for s in model.predict(pairs, batch_size=self.batch_size)]

    def search(self, query: str) -> list[tuple[int, float]]:
        candidates = self.base.search(query)
        if self.only_with_diacritics and not has_diacritics(query):
            return candidates
        head = [i for i, _ in candidates[: self.top_n]]
        scores = self.score(query, head)
        reranked = sorted(zip(head, scores, strict=True), key=lambda r: -r[1])
        floor = min(scores, default=0.0) - 1.0  # phần đuôi luôn đứng sau phần đã rerank
        tail = [(i, floor - rank) for rank, (i, _) in enumerate(candidates[self.top_n :], 1)]
        return reranked + tail
