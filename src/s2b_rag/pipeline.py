"""Pipeline nộp bài: `neorag run <split> --pipeline s2b_rag.pipeline:S2BPipeline`.

query ─► retrieval preset (mặc định `hybrid_r10`: câu không dấu được haiku viết lại có dấu, rồi hybrid + rerank
         + quota CLB; xem ADR-0003/0004 và docs/journal.md) ─► 10 id
      └► `n_context` đoạn đầu vào prompt (G1) ─► claude-haiku-5-5 (1 lần gọi) ─► hậu xử lý (G2)

Runner của BTC không truyền tham số khi khởi tạo pipeline, nên giá trị mặc định ở đây CHÍNH LÀ cấu hình nộp bài.

Chatbot dùng `retrieve()` (hiện nguồn + thời gian trước) rồi `stream_answer()` (chữ hiện dần). `run()` cũng đi qua
`retrieve()` và cùng `build_messages`, nên chatbot và bài nộp luôn cùng retrieval và cùng prompt.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines import RAGOutput, RAGPipeline
from s2b_rag.generate import build_messages, postprocess
from s2b_rag.retrieval.dense import CACHE_PATH
from s2b_rag.retrieval.presets import DEFAULT, RETRIEVERS

LLM_MODEL = "claude-haiku-5-5"  # LLM duy nhất được phép; KHÔNG truyền temperature


@dataclass
class Retrieval:
    """Kết quả tầng truy xuất của một câu hỏi (để chatbot hiển thị và để sinh câu trả lời)."""

    docs: list[Document]  # các đoạn nộp (n_retrieved), theo hạng
    context: list[Document]  # n_context đoạn đầu, đưa vào prompt
    rewritten: str | None  # câu hỏi do haiku viết lại (R7), None nếu không viết lại
    seconds: float  # thời gian truy xuất (gồm cả lần gọi viết lại nếu có)

    @property
    def ids(self) -> list[int]:
        return [d.id for d in self.docs]


class S2BPipeline(RAGPipeline):
    def __init__(
        self,
        retriever: str = DEFAULT,
        n_context: int = 8,
        n_retrieved: int = 10,
        llm: LLMClient | None = None,
        stream_client=None,
    ):
        self.retriever_name = retriever
        self.retriever = RETRIEVERS[retriever]()
        # số đoạn đưa vào prompt. 8 thay vì 5: holdout có judge 86.03 -> 86.52 (SP, ROUGE-L, Relevancy tăng,
        # Noise không đổi); dev: ~7/16 câu thiếu key fact vì đoạn đúng ở hạng 6-10. Xem docs/journal.md.
        self.n_context = n_context
        self.n_retrieved = n_retrieved  # số id nộp; luôn ⊇ các đoạn trong prompt (Groundedness)
        self.llm_model = LLM_MODEL  # runner đọc thuộc tính này để in ra màn hình
        self.llm = llm or LLMClient(cache_path=CACHE_PATH)  # cache: chạy lại cùng prompt = 0 USD
        self._stream_client = (
            stream_client  # openai.OpenAI cho chatbot; tạo khi cần (test truyền client giả)
        )

    def prepare(self) -> None:
        """Nạp sẵn model reranker (~27 s) nếu cấu hình có dùng, để câu hỏi đầu tiên không bị chậm."""
        from s2b_rag.retrieval.rerank import CrossEncoderReranker, load_cross_encoder

        layer = self.retriever
        while layer is not None:  # các tầng lồng nhau qua thuộc tính `base`
            if isinstance(layer, CrossEncoderReranker):
                load_cross_encoder(layer.model_name)
                return
            layer = getattr(layer, "base", None)

    def index(self, corpus: list[Document]) -> None:
        self.docs = {d.id: d for d in corpus}
        self.retriever.index(corpus)

    def retrieve(self, query: str) -> Retrieval:
        start = time.perf_counter()
        ranked = [doc_id for doc_id, _ in self.retriever.search(query)]
        seconds = time.perf_counter() - start
        docs = [self.docs[i] for i in ranked[: self.n_retrieved]]
        return Retrieval(docs, docs[: self.n_context], self._last_rewrite(), seconds)

    def _last_rewrite(self) -> str | None:
        """Câu viết lại của lần search vừa rồi, tìm qua các tầng lồng nhau (`base`)."""
        layer = self.retriever
        while layer is not None:
            if (rw := getattr(layer, "last_rewrite", None)) is not None:
                return rw.query
            layer = getattr(layer, "base", None)
        return None

    def run(self, query: str) -> RAGOutput:
        r = self.retrieve(query)
        answer = self.llm.chat(build_messages(query, r.context), model=self.llm_model)
        return RAGOutput(answer=postprocess(answer), retrieved_ids=r.ids)

    def stream_answer(self, query: str, context: list[Document]) -> Iterator[str]:
        """Chatbot: cùng prompt như `run()`, nhưng trả từng mẩu chữ (`stream=True`) để hiện dần.

        Không qua cache của LLMClient (cache cần cả câu trả lời), nên mỗi lần hỏi là một lần gọi (~0.0005 USD).
        Ghép các mẩu rồi `postprocess` thì ra đúng định dạng của `run()`. Không truyền temperature."""
        if self._stream_client is None:
            from openai import OpenAI

            self._stream_client = (
                OpenAI()
            )  # đọc OPENAI_API_KEY / OPENAI_BASE_URL (neorag.llm đã nạp .env)
        stream = self._stream_client.chat.completions.create(
            model=self.llm_model, messages=build_messages(query, context), stream=True
        )
        for chunk in stream:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                yield delta
