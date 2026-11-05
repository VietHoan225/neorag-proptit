"""Pipeline nộp bài: `neorag run <split> --pipeline s2b_rag.pipeline:S2BPipeline` (= A5, bài nộp #9, 89.15).

query ─► retrieval preset (mặc định `hybrid_r10`: câu không dấu được haiku viết lại có dấu, rồi hybrid + rerank
         + quota CLB; xem ADR-0003/0004 và docs/journal.md) ─► 20 đoạn đầu vào prompt
      └► claude-haiku-5-5 lần 1 (prompt A): "Ý HỎI: Ý n ... | NGUỒN: a, b" rồi "TRẢ LỜI:" (A)
      └► claude-haiku-5-5 lần 2 (prompt A4): chỉ lấy NGUỒN, gộp với lần 1 (A5, ensemble nguồn)
         └► đoạn chính của mỗi ý lên đầu, rồi đoạn phụ → cắt 10 id (R9/E1, ADR-0005)
         └► hậu xử lý: bỏ markdown (G2) + bỏ câu rào đón thuần ở cuối (A2)
         └► đoạn đã dùng mà quên trích → đổi vào hạng 6–10 (A)

Runner của BTC không truyền tham số khi khởi tạo pipeline, nên giá trị mặc định của class CHÍNH LÀ cấu hình nộp
bài. Lịch sử: S2BPipelineR9 (#4) → S2BPipelineCtx20 → S2BPipelineE2 (#5) → S2BPipelineA (#6) → S2BPipelineA2 (#7)
→ S2BPipelineA5 (#9, = S2BPipeline) ; S2BPipelineA3 (#8, bỏ) ; S2BPipelineA4 (bỏ) ; S2BPipelineA6 (#10, bỏ).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines import RAGOutput, RAGPipeline
from s2b_rag.generate import (
    FALLBACK_ANSWER,
    RETRY_NOTE,
    RETRY_NOTE_A,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_A,
    SYSTEM_PROMPT_A4,
    SYSTEM_PROMPT_E2,
    AnswerStreamFilter,
    build_messages,
    drop_tail_hedges,
    keep_used_sources,
    merge_tiers,
    postprocess,
    promote_tiers,
    split_aspects,
    split_sources,
    trim_hedges,
    with_retry_note,
)
from s2b_rag.retrieval.dense import CACHE_PATH
from s2b_rag.retrieval.presets import DEFAULT, RETRIEVERS

LLM_MODEL = "claude-haiku-5-5"  # LLM duy nhất được phép; KHÔNG truyền temperature


@dataclass
class Retrieval:
    """Kết quả tầng truy xuất của một câu hỏi (chatbot hiển thị nguồn trước khi có câu trả lời)."""

    docs: list[Document]  # n_retrieved đoạn đầu, theo hạng retriever
    context: list[Document]  # n_context đoạn đầu, đưa vào prompt
    rewritten: str | None  # câu haiku viết lại (R7), None nếu không viết lại
    seconds: float  # thời gian truy xuất (gồm cả lần gọi viết lại nếu có)

    @property
    def ids(self) -> list[int]:
        return [d.id for d in self.docs]


class AnswerStream:
    """Duyệt qua → các mẩu chữ được phép hiện; duyệt xong → `.raw` là phản hồi đầy đủ (cả phần bị giấu)."""

    def __init__(self, chunks, stream_filter: AnswerStreamFilter):
        self._chunks, self._filter, self._parts = chunks, stream_filter, []

    @property
    def raw(self) -> str:
        return "".join(self._parts)

    def __iter__(self) -> Iterator[str]:
        for chunk in self._chunks:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                self._parts.append(delta)
                if visible := self._filter.feed(delta):
                    yield visible
        if tail := self._filter.finish():
            yield tail


class S2BPipelineR9(RAGPipeline):
    """Lớp gốc = cấu hình R9 (bài nộp #4, 85.09): retrieval → 10 đoạn → haiku ghi NGUỒN → đoạn được trích lên đầu.

    Các thí nghiệm sau kế thừa từ đây (Ctx20 → E2 → A → A2 → A3) để mỗi bài nộp cũ vẫn tái hiện được.
    Cấu hình nộp bài hiện tại là `S2BPipeline` ở cuối file."""

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
        tiers = self.extra_tiers(query, context, allowed, tiers)
        retrieved_ids = promote_tiers(ranked, tiers)[: self.n_retrieved]
        answer = self.finalize_answer(postprocess(answer) or FALLBACK_ANSWER)
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

    def finalize_answer(self, answer: str) -> str:
        """Chỉnh lần cuối câu trả lời (sau G2); mặc định giữ nguyên (A2 bỏ câu rào đón thuần ở cuối)."""
        return answer

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        """Bổ sung nguồn trước khi xếp id; mặc định giữ nguyên (A5 gộp nguồn của lần haiku đọc thứ 2)."""
        return tiers

    # ---- Chatbot (`python -m s2b_rag.web`). KHÔNG dùng trong `run()`: bài nộp không đổi. ----
    # Chatbot chỉ gọi haiku lần 1 (câu trả lời); lần đọc thứ 2 của A5 chỉ phục vụ id nộp bài nên bỏ để nhanh.
    streams_aspects = False  # định dạng A: phải giấu phần "Ý HỎI" trước "TRẢ LỜI:" khi stream
    stream_client = None  # openai.OpenAI, tạo khi cần; test gán client giả

    def retrieve(self, query: str) -> Retrieval:
        """Cùng retrieval với `run()`: n_retrieved đoạn để hiển thị, n_context đoạn đưa vào prompt."""
        start = time.perf_counter()
        ranked = [doc_id for doc_id, _ in self.retriever.search(query)]
        seconds = time.perf_counter() - start
        return Retrieval(
            docs=[self.docs[i] for i in ranked[: self.n_retrieved]],
            context=[self.docs[i] for i in ranked[: self.n_context]],
            rewritten=self._last_rewrite(),
            seconds=seconds,
        )

    def _last_rewrite(self) -> str | None:
        """Câu haiku viết lại (thêm dấu) ở lần search vừa rồi, tìm qua các tầng lồng nhau (`base`)."""
        layer = self.retriever
        while layer is not None:
            if (rw := getattr(layer, "last_rewrite", None)) is not None:
                return rw.query
            layer = getattr(layer, "base", None)
        return None

    def stream_answer(self, query: str, context: list[Document]) -> AnswerStream:
        """Gọi haiku lần 1 với CÙNG prompt như `run()` (`stream=True`), chỉ cho hiện phần câu trả lời.

        Không qua cache của LLMClient (cache cần cả câu trả lời) → mỗi câu hỏi là một lần gọi. Không truyền temperature.
        """
        if self.stream_client is None:
            from openai import OpenAI

            self.stream_client = (
                OpenAI()
            )  # đọc OPENAI_API_KEY / OPENAI_BASE_URL (neorag.llm đã nạp .env)
        chunks = self.stream_client.chat.completions.create(
            model=self.llm_model,
            messages=build_messages(query, context, self.system_prompt),
            stream=True,
        )
        return AnswerStream(chunks, AnswerStreamFilter(wait_head=self.streams_aspects))

    def finish_answer(self, raw: str, context: list[Document]) -> tuple[str, list[int]]:
        """Từ phản hồi đầy đủ: (câu trả lời đã hậu xử lý y như `run()`, id các đoạn haiku đã trích)."""
        answer, tiers = self.parse_reply(raw, [d.id for d in context])
        answer = self.finalize_answer(postprocess(answer) or FALLBACK_ANSWER)
        return answer, list(dict.fromkeys(d for tier in tiers for d in tier))


class S2BPipelineCtx20(S2BPipelineR9):
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
    streams_aspects = True  # chatbot: giấu phần "Ý HỎI" khi stream

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


class S2BPipelineA2(S2BPipelineA):
    """Thí nghiệm A2 = A + bỏ câu rào đón thuần ở cuối câu trả lời (`drop_tail_hedges`, code, 0 lần gọi LLM).

    Cùng prompt với A nên chạy lại trên dev_casual/public đều trúng cache (0 USD).
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineA2`. So với `S2BPipelineA` (bài nộp #6, 87.36)."""

    def finalize_answer(self, answer: str) -> str:
        return drop_tail_hedges(answer)


class S2BPipelineA3(S2BPipelineA2):
    """Thí nghiệm A3 = A + `trim_hedges` (A2 mở rộng: bảo vệ theo câu đầu mở đầu bằng rào đón, bỏ vế rào đón giữa câu,
    bỏ "Theo tài liệu, ..."). Code, 0 lần gọi LLM; cùng prompt với A → trúng cache.
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineA3`. So với `S2BPipelineA2` (bài nộp #7, 88.17)."""

    def finalize_answer(self, answer: str) -> str:
        return trim_hedges(answer)


class S2BPipelineA4(S2BPipelineA2):
    """Thí nghiệm A4 = A2 + prompt `SYSTEM_PROMPT_A4`: Ý HỎI giữ đúng vai trò được hỏi, câu liệt kê nêu đủ mục và
    trích mọi nguồn của từng ý, được tính từ số/ngày có sẵn, mỗi ý chỉ nêu thông tin trong phạm vi được hỏi.
    Prompt mới → không trúng cache (dev_casual ~0.14 USD, public ~0.07 USD).
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineA4`. So với `S2BPipelineA2` (bài nộp #7, 88.17)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("system_prompt", SYSTEM_PROMPT_A4)
        super().__init__(**kwargs)


class S2BPipelineA5(S2BPipelineA2):
    """Thí nghiệm A5 = câu trả lời của A2 + nguồn gộp từ 2 lần haiku đọc cùng 20 đoạn (ensemble nguồn).

    Lần 1: prompt A (= A2) → câu trả lời + nguồn. Lần 2: prompt A4 ("mọi nguồn cho từng ý") → CHỈ lấy nguồn, bỏ câu
    trả lời. id nộp = [đoạn chính của cả 2 lần] rồi [đoạn phụ của cả 2 lần] → cắt 10. Cùng một luật cho mọi câu, không
    dùng gold. Đo từ cache dev_casual (0 USD): retrieval 98.40 → 98.76. Chi phí: 2 lần gọi/câu (public: lần 1 trúng
    cache #7). Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineA5`. So với `S2BPipelineA2` (bài nộp #7, 88.17)."""

    second_prompt = SYSTEM_PROMPT_A4

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        raw = self.llm.chat(
            build_messages(query, context, self.second_prompt), model=self.llm_model
        )
        _, second = split_aspects(raw, allowed)
        return merge_tiers(tiers, second)


class S2BPipelineA6(S2BPipelineA5):
    """Thí nghiệm A6 = A5 + lần đọc thứ 3 (prompt E2 của bài #5: trả lời trước, 1 dòng NGUỒN ở cuối) → CHỈ lấy nguồn,
    xếp làm tầng thứ 3: chỉ lấp các ô còn trống sau đoạn chính/phụ của A5, không đẩy đoạn nào của A5 ra.

    Lý do (cache dev_casual): đoạn được trích là gold 81.9%, ô lấp theo retriever chỉ 1.2%; public A5 có 119/210 ô top-5
    là ô lấp. Chọn E2 vì đã có trong cache (public 0 USD), không phải vì nó là prompt chọn nguồn tốt nhất.
    Chi phí ngày thi: 3 lần gọi/câu. Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineA6`. So với `S2BPipelineA5` (#9)."""

    third_prompt = SYSTEM_PROMPT_E2

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        merged = super().extra_tiers(query, context, allowed, tiers)
        raw = self.llm.chat(build_messages(query, context, self.third_prompt), model=self.llm_model)
        _, third = split_sources(raw, allowed)
        return [*merged, third]


class S2BPipeline(S2BPipelineA5):
    """CẤU HÌNH NỘP BÀI (chốt 04/11) = A5: public #9 89.15 (retrieval 93.93). Trước đó A2: #7 88.17 (chấm lại 87.99).
    Mỗi câu gọi haiku 2 lần (câu trả lời của lần 1 + nguồn gộp của 2 lần), +1 lần thêm dấu nếu câu không dấu (R10);
    chatbot chỉ cần lần 1 (+ thêm dấu).

    Ngày thi: `neorag run private_test --pipeline s2b_rag.pipeline:S2BPipeline`. Đổi cấu hình nộp bài = đổi lớp cha
    của class này, không sửa các class thí nghiệm ở trên."""
