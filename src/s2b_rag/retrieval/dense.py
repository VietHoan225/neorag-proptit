"""Dense retrieval: cosine similarity giữa embedding câu hỏi và embedding đoạn văn."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from neorag.data import Document
from neorag.embeddings import LocalEmbedder, is_downloaded
from neorag.llm import LLMClient
from neorag.pipelines.baseline import GatewayEmbedder

# Dùng chung file cache với baseline của BTC: embedding corpus và câu hỏi train đã có sẵn trong đó,
# nên đo lại dense baseline không tốn request nào. Khóa cache khác nhau nên không lẫn dữ liệu.
CACHE_PATH = ".cache/pipeline.sqlite"
EMBEDDING_MODEL = "bge-m3"


class DenseRetriever:
    def __init__(self, embedder, query_fn: Callable[[str], str] = str):
        self.embedder = embedder  # có .embed(texts) -> ma trận đã chuẩn hóa L2
        self.query_fn = query_fn  # chuẩn hoá câu hỏi trước khi embed (mặc định giữ nguyên)

    def index(self, corpus: list[Document]) -> None:
        self.ids = [d.id for d in corpus]
        self.vectors = self.embedder.embed([d.text for d in corpus])

    def search(self, query: str) -> list[tuple[int, float]]:
        q = self.embedder.embed([self.query_fn(query)])[0]
        scores = self.vectors @ q  # vector đã chuẩn hóa nên tích vô hướng = cosine
        order = np.argsort(-scores, kind="stable")  # stable: hòa điểm thì giữ thứ tự corpus
        return [(self.ids[i], float(scores[i])) for i in order]


class SafeLocalEmbedder:
    """LocalEmbedder bge-m3: chạy trên GPU nếu có, max_seq_length=512 để nạp nhanh."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None, batch_size: int = 32):
        import os
        import torch

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device or os.getenv("EMBEDDING_DEVICE", default_device)
        self.batch_size = batch_size
        self._model = None

    def load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name, device=self.device, local_files_only=is_downloaded(self.model_name)
            )
            self._model.max_seq_length = 512
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self.load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def gateway_dense(query_fn: Callable[[str], str] = str) -> DenseRetriever:
    """bge-m3 qua gateway hoặc local.

    Ưu tiên LocalEmbedder nếu đã có model trên máy (hoặc EMBEDDING_PROVIDER=local / base_url
    không phải BTC), tránh lỗi khi API key gateway hết hạn hoặc khi dùng LLM bên thứ ba (Groq/Gemini).
    """
    import os

    force_gateway = os.getenv("EMBEDDING_PROVIDER") == "gateway"
    if not force_gateway and (
        os.getenv("EMBEDDING_PROVIDER") == "local"
        or not os.getenv("OPENAI_BASE_URL", "").startswith("https://proptit.dekthe.dev")
        or is_downloaded("BAAI/bge-m3")
    ):
        embedder = SafeLocalEmbedder("BAAI/bge-m3")
    else:
        llm = LLMClient(cache_path=CACHE_PATH)
        embedder = GatewayEmbedder(llm, EMBEDDING_MODEL)
    return DenseRetriever(embedder, query_fn)
