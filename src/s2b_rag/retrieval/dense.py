"""Dense retrieval: cosine similarity giữa embedding câu hỏi và embedding đoạn văn."""

from __future__ import annotations

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.baseline import GatewayEmbedder

# Dùng chung file cache với baseline của BTC: embedding corpus và câu hỏi train đã có sẵn trong đó,
# nên đo lại dense baseline không tốn request nào. Khóa cache khác nhau nên không lẫn dữ liệu.
CACHE_PATH = ".cache/pipeline.sqlite"
EMBEDDING_MODEL = "bge-m3"


class DenseRetriever:
    def __init__(self, embedder):
        self.embedder = embedder  # có .embed(texts) -> ma trận đã chuẩn hóa L2

    def index(self, corpus: list[Document]) -> None:
        self.ids = [d.id for d in corpus]
        self.vectors = self.embedder.embed([d.text for d in corpus])

    def search(self, query: str) -> list[tuple[int, float]]:
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q  # vector đã chuẩn hóa nên tích vô hướng = cosine
        order = np.argsort(-scores, kind="stable")  # stable: hòa điểm thì giữ thứ tự corpus
        return [(self.ids[i], float(scores[i])) for i in order]


def gateway_dense() -> DenseRetriever:
    """Dense baseline của BTC: bge-m3 qua gateway, embed nguyên văn bản đoạn và câu hỏi."""
    llm = LLMClient(cache_path=CACHE_PATH)
    return DenseRetriever(GatewayEmbedder(llm, EMBEDDING_MODEL))
