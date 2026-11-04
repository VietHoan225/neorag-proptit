"""Pipeline nộp bài: `neorag run <split> --pipeline s2b_rag.pipeline:S2BPipeline`.

query ─► retrieval preset (mặc định `hybrid_r10`: câu không dấu được haiku viết lại có dấu, rồi hybrid + rerank
         + quota CLB; xem ADR-0003/0004 và docs/journal.md) ─► 10 id
      └► `n_context` đoạn đầu vào prompt (G1) ─► claude-haiku-5-5 (1 lần gọi) ─► hậu xử lý (G2)
         └► dòng "NGUỒN:" → đoạn được trích lên đầu (R9) → cắt còn 10 id (E1: cắt sau, không cắt trước)

Runner của BTC không truyền tham số khi khởi tạo pipeline, nên giá trị mặc định ở đây CHÍNH LÀ cấu hình nộp bài.
"""

from __future__ import annotations

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines import RAGOutput, RAGPipeline
from s2b_rag.generate import (
    FALLBACK_ANSWER,
    RETRY_NOTE,
    RETRY_NOTE_A,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_A,
    SYSTEM_PROMPT_E2,
    build_messages,
    keep_used_sources,
    postprocess,
    promote_tiers,
    split_aspects,
    split_sources,
    with_retry_note,
)
from s2b_rag.retrieval.dense import CACHE_PATH
from s2b_rag.retrieval.presets import DEFAULT, RETRIEVERS

LLM_MODEL = "claude-haiku-5-5"  # LLM duy nhất được phép; KHÔNG truyền temperature


class S2BPipeline(RAGPipeline):
    def __init__(
        self,
        retriever: str = DEFAULT,
        n_context: int = 10,
        n_retrieved: int = 10,
        llm: LLMClient | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.retriever_name = retriever
        self.retriever = RETRIEVERS[retriever]()
        # số đoạn đưa vào prompt. 8 thay vì 5: holdout có judge 86.03 -> 86.52 (SP, ROUGE-L, Relevancy tăng,
        # Noise không đổi); dev: ~7/16 câu thiếu key fact vì đoạn đúng ở hạng 6-10. Xem docs/journal.md.
        # R9 bước 1 (thử 03/11): 10 = toàn bộ id nộp, để haiku thấy cả đoạn hạng 9-10 của câu nhiều ý.
        self.n_context = n_context
        self.system_prompt = (
            system_prompt  # E2 thử prompt khác qua class con, mặc định = prompt đang nộp
        )
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
        context = [self.docs[i] for i in ranked[: self.n_context]]
        allowed = [d.id for d in context]
        messages = build_messages(query, context, self.system_prompt)
        # R9: haiku ghi các đoạn đã dùng ở dòng "NGUỒN:" → đưa chúng lên đầu, để câu nhiều ý có đủ đoạn
        # cho mọi ý trong top-5. Dòng nguồn bị cắt khỏi câu trả lời; thiếu/hỏng thì giữ thứ tự retriever.
        answer, tiers = self.parse_reply(self.llm.chat(messages, model=self.llm_model), allowed)
        if not postprocess(answer):  # chỉ còn dòng NGUỒN (hoặc rỗng) → gọi lại 1 lần
            retry_answer, retry_tiers = self.parse_reply(
                self.llm.chat(with_retry_note(messages, self.retry_note), model=self.llm_model),
                allowed,
            )
            if postprocess(retry_answer):
                answer, tiers = retry_answer, retry_tiers
        # E1: cắt còn `n_retrieved` id SAU khi đưa đoạn được trích lên, để đoạn hạng 11–20 mà haiku trích
        # vẫn vào được top-5. Với n_context <= n_retrieved, kết quả y hệt cắt trước (bài nộp #4).
        retrieved_ids = promote_tiers(ranked, tiers)[: self.n_retrieved]
        answer = postprocess(answer) or FALLBACK_ANSWER
        return RAGOutput(
            answer=answer, retrieved_ids=self.finalize_ids(answer, retrieved_ids, allowed, tiers)
        )

    retry_note = RETRY_NOTE

    def parse_reply(self, raw: str, allowed: list[int]) -> tuple[str, list[list[int]]]:
        """R9: (câu trả lời, [các đoạn được trích]) — một tầng. Class con đổi định dạng đầu ra (A)."""
        answer, cited = split_sources(raw, allowed)
        return answer, [cited]

    def finalize_ids(
        self, answer: str, ids: list[int], allowed: list[int], tiers: list[list[int]]
    ) -> list[int]:
        """Chỉnh lần cuối danh sách id nộp; mặc định giữ nguyên (A dùng để giữ nguồn đã dùng)."""
        return ids


class S2BPipelineCtx20(S2BPipeline):
    """Thí nghiệm E1: haiku đọc 20 đoạn (nộp vẫn 10 id). Trần đo trên dev_casual: 0.985 (10) → 0.997 (20).

    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineCtx20` (CLI không truyền tham số nên cần class riêng)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("n_context", 20)
        super().__init__(**kwargs)


class S2BPipelineE2(S2BPipelineCtx20):
    """Thí nghiệm E1+E2: 20 đoạn + prompt bớt rào đón thừa (`SYSTEM_PROMPT_E2`).

    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineE2`. So với `S2BPipelineCtx20` để tách tác dụng của E2."""

    def __init__(self, **kwargs):
        kwargs.setdefault("system_prompt", SYSTEM_PROMPT_E2)
        super().__init__(**kwargs)


class S2BPipelineA(S2BPipelineE2):
    """Thí nghiệm A (gộp 3 thay đổi, 1 lần chạy): (1) haiku ghi từng ý + nguồn TRƯỚC khi trả lời, đoạn chính của mỗi
    ý lên top; (3) prompt bỏ ví dụ/nhắc lại/"Vậy ..."; (4) đoạn đã dùng mà quên trích → đổi vào hạng 6–10.

    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineA`. So với `S2BPipelineE2` (bài nộp #5, 87.02)."""

    retry_note = RETRY_NOTE_A

    def __init__(self, **kwargs):
        kwargs.setdefault("system_prompt", SYSTEM_PROMPT_A)
        super().__init__(**kwargs)

    def parse_reply(self, raw: str, allowed: list[int]) -> tuple[str, list[list[int]]]:
        return split_aspects(raw, allowed)

    def finalize_ids(
        self, answer: str, ids: list[int], allowed: list[int], tiers: list[list[int]]
    ) -> list[int]:
        candidates = [i for i in allowed if i not in ids]
        texts = {i: self.docs[i].text for i in allowed}
        protected = {i for tier in tiers for i in tier}
        return keep_used_sources(answer, ids, candidates, texts, protected)
