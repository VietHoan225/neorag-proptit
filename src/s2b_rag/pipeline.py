"""Pipeline nộp bài: `neorag run <split> --pipeline s2b_rag.pipeline:S2BPipeline`.

query ─► retrieval preset (mặc định `hybrid_r7`: câu không dấu được haiku viết lại có dấu, rồi hybrid + rerank;
         xem ADR-0003/0004 và docs/journal.md) ─► 10 id
      └► `n_context` đoạn đầu vào prompt (G1) ─► claude-haiku-5-5 (1 lần gọi) ─► hậu xử lý (G2)

Runner của BTC không truyền tham số khi khởi tạo pipeline, nên giá trị mặc định ở đây CHÍNH LÀ cấu hình nộp bài.
"""

from __future__ import annotations

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines import RAGOutput, RAGPipeline
from s2b_rag.generate import build_messages, postprocess
from s2b_rag.retrieval.dense import CACHE_PATH
from s2b_rag.retrieval.presets import DEFAULT, RETRIEVERS

LLM_MODEL = "claude-haiku-5-5"  # LLM duy nhất được phép; KHÔNG truyền temperature


class S2BPipeline(RAGPipeline):
    def __init__(
        self,
        retriever: str = DEFAULT,
        n_context: int = 8,
        n_retrieved: int = 10,
        llm: LLMClient | None = None,
    ):
        self.retriever_name = retriever
        self.retriever = RETRIEVERS[retriever]()
        # số đoạn đưa vào prompt. 8 thay vì 5: holdout có judge 86.03 -> 86.52 (SP, ROUGE-L, Relevancy tăng,
        # Noise không đổi); dev: ~7/16 câu thiếu key fact vì đoạn đúng ở hạng 6-10. Xem docs/journal.md.
        self.n_context = n_context
        self.n_retrieved = n_retrieved  # số id nộp; luôn ⊇ các đoạn trong prompt (Groundedness)
        self.llm_model = LLM_MODEL  # runner đọc thuộc tính này để in ra màn hình
        self.llm = llm or LLMClient(cache_path=CACHE_PATH)  # cache: chạy lại cùng prompt = 0 USD

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

    def run(self, query: str) -> RAGOutput:
        ranked = [doc_id for doc_id, _ in self.retriever.search(query)]
        retrieved_ids = ranked[: self.n_retrieved]
        context = [self.docs[i] for i in retrieved_ids[: self.n_context]]
        answer = self.llm.chat(build_messages(query, context), model=self.llm_model)
        return RAGOutput(answer=postprocess(answer), retrieved_ids=retrieved_ids)
