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

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines import RAGOutput, RAGPipeline
from s2b_rag.aliases import mirror_aliases
from s2b_rag.evidence import (
    aspect_lines,
    check_messages,
    check_messages_r13,
    drop_least_used,
    insert_supplements,
    insert_supplements_r13,
    needs_check,
    parse_supplements,
    parse_supplements_r13,
)
from s2b_rag.generate import (
    FALLBACK_ANSWER,
    RETRY_NOTE,
    RETRY_NOTE_A,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_A,
    SYSTEM_PROMPT_A4,
    SYSTEM_PROMPT_E2,
    SYSTEM_PROMPT_F,
    AnswerStreamFilter,
    build_messages,
    consensus_first,
    drop_tail_hedges,
    keep_used_sources,
    mbr_select,
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

LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")


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


class S2BPipelineMBR(S2BPipelineA5):
    """Thí nghiệm MBR = A5 + chọn câu trả lời theo đồng thuận trong 4 bản haiku viết cho cùng 20 đoạn.

    Lần đọc: A (câu trả lời + nguồn, như A5), A4 (nguồn, như A5; nay giữ cả câu trả lời), E2, F (chỉ lấy câu trả
    lời). Mỗi bản qua cùng hậu xử lý A2 rồi `mbr_select` chọn 1. id nộp giữ cách gộp nguồn của A5 (top-5 như #9);
    `keep_used_sources` với câu trả lời được chọn đổi đoạn nó dùng vào hạng 6–10 (Groundedness).
    dev_casual (cache, 0 USD): SP 0.9359 → 0.9444, ROUGE-L 0.6375 → 0.6844. Ngày thi: 4 lần gọi/câu (+ thêm dấu).
    Chatbot không dùng (chỉ đọc lần A). Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineMBR`. So với A5 (#9)."""

    candidate_prompts = (SYSTEM_PROMPT_E2, SYSTEM_PROMPT_F)

    def run(self, query: str) -> RAGOutput:
        self._others: list[str] = []
        try:
            return super().run(query)
        finally:
            self._others = []  # không để bản cũ lọt sang lần gọi sau (chatbot dùng finalize_answer)

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        # lần đọc 2 y hệt A5 (cùng messages → trúng cache), nhưng giữ lại câu trả lời
        raw = self.llm.chat(
            build_messages(query, context, self.second_prompt), model=self.llm_model
        )
        answer4, second = split_aspects(raw, allowed)
        others = [answer4]
        for prompt in self.candidate_prompts:
            raw = self.llm.chat(build_messages(query, context, prompt), model=self.llm_model)
            parse = split_sources if prompt == SYSTEM_PROMPT_E2 else split_aspects
            others.append(parse(raw, allowed)[0])
        self._others = others
        return merge_tiers(tiers, second)

    def finalize_answer(self, answer: str) -> str:
        first = super().finalize_answer(answer)
        others = [
            super(S2BPipelineMBR, self).finalize_answer(postprocess(a))
            for a in getattr(self, "_others", [])
        ]
        return mbr_select([first, *others]) or first


class S2BPipelineR11(S2BPipelineA5):
    """Thí nghiệm R11 = A5 + xếp lại top-5: đoạn được CẢ HAI lần đọc chọn làm đoạn chính lên trước (`consensus_first`).

    Cùng 2 lần gọi như A5 (trúng cache) → 0 USD; câu trả lời và tập top-5 y hệt A5, chỉ đổi thứ tự (public đổi 5/42).
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineR11`. So với `S2BPipelineA5` (bài nộp #9, 89.15)."""

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        raw = self.llm.chat(
            build_messages(query, context, self.second_prompt), model=self.llm_model
        )
        _, second = split_aspects(raw, allowed)
        # predict chạy tuần tự từng câu → lưu tạm trên instance (như MBR), finalize_ids đọc ngay sau đó
        self._agreed = set(tiers[0] if tiers else []) & set(second[0] if second else [])
        return merge_tiers(tiers, second)

    def finalize_ids(
        self, answer: str, ids: list[int], allowed: list[int], tiers: list[list[int]]
    ) -> list[int]:
        ids = super().finalize_ids(answer, ids, allowed, tiers)
        return consensus_first(ids, getattr(self, "_agreed", set()))


class S2BPipelineR12(S2BPipelineR11):
    """Thí nghiệm R12 = R11 + lượt kiểm tra bằng chứng trên TOÀN corpus cho câu bị gắn cờ (`s2b_rag.evidence`).

    Câu trả lời y hệt R11 (#13); chỉ thay đoạn top-5 không đồng thuận bằng đoạn có câu trích nguyên văn trả lời ý thiếu.
    Chi phí: +1 lần gọi (~51k ký tự) cho câu bị gắn cờ (public 11/42). Lần đọc 1 và 2 được đọc lại từ cache.
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineR12`. So với `S2BPipelineR11` (bài nộp #13, 89.42)."""

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        merged = super().extra_tiers(query, context, allowed, tiers)
        self._capture_check(query, context, allowed, tiers)
        return merged

    # True = khi chèn đoạn bổ sung làm dư id, bỏ đoạn hạng 6–10 câu trả lời ít dùng nhất (`drop_least_used`) thay vì
    # đoạn cuối. False ở các class thí nghiệm để tái lập đúng bài #15/#16; bật ở `S2BPipeline`.
    keep_answer_sources = False

    def finalize_ids(
        self, answer: str, ids: list[int], allowed: list[int], tiers: list[list[int]]
    ) -> list[int]:
        return self._evidence_check(super().finalize_ids(answer, ids, allowed, tiers), answer)

    def _capture_check(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> None:
        """Lưu tín hiệu cho trigger (ý + đoạn chính của 2 lần đọc). Dùng chung cho R12, R13, R12MBR."""
        self._last_context = context
        self._last_added = []
        allowed_set = set(allowed)
        # cùng messages với lần đọc 1 (run) và lần 2 (R11) → LLMClient trả từ cache
        raw1 = self.llm.chat(
            build_messages(query, context, self.system_prompt), model=self.llm_model
        )
        raw2 = self.llm.chat(
            build_messages(query, context, self.second_prompt), model=self.llm_model
        )
        _, second = split_aspects(raw2, allowed)
        self._check = {
            "query": query,
            "aspects1": aspect_lines(raw1, allowed_set),
            "aspects2": aspect_lines(raw2, allowed_set),
            "primary1": set(tiers[0] if tiers else []),
            "primary2": set(second[0] if second else []),
        }

    def _evidence_check(self, ids: list[int], answer: str = "") -> list[int]:
        """Câu bị gắn cờ → haiku đọc toàn corpus, bổ sung đoạn có trích nguyên văn vào top-5 (sau R11)."""
        info = getattr(self, "_check", None)
        if not info:
            return ids
        top5 = ids[:5]
        if not needs_check(
            info["query"],
            info["aspects1"],
            info["aspects2"],
            info["primary1"],
            info["primary2"],
            {self.docs[i].club for i in top5},
        ):
            return ids
        corpus = list(self.docs.values())
        raw = self.llm.chat(
            check_messages(info["query"], info["aspects1"], top5, corpus), model=self.llm_model
        )
        texts = {i: d.text for i, d in self.docs.items()}
        added = parse_supplements(raw, texts, top5)
        self._last_added = added
        if not getattr(self, "keep_answer_sources", False):
            return insert_supplements(ids, added, self._agreed)
        return drop_least_used(
            answer, insert_supplements(ids, added, self._agreed, cut=False), texts, keep=len(ids)
        )

    def _refresh_answer_if_added(
        self, query: str, answer: str, retrieved_ids: list[int]
    ) -> tuple[str, list[int]]:
        """Khi _evidence_check tìm được đoạn bổ sung ngoài 20 đoạn ban đầu, cập nhật câu trả lời với context mới."""
        added = getattr(self, "_last_added", None)
        if not added:
            return answer, retrieved_ids
        old_ctx = getattr(self, "_last_context", [])
        new_ctx = [self.docs[i] for i in added] + [d for d in old_ctx if d.id not in added]
        new_ctx = new_ctx[: self.n_context]
        allowed = [d.id for d in new_ctx]
        raw = self.llm.chat(
            build_messages(query, new_ctx, self.system_prompt), model=self.llm_model
        )
        new_ans, _ = self.parse_reply(raw, allowed)
        new_ans = self.finalize_answer(postprocess(new_ans) or answer)
        return new_ans, self._resync_ids(new_ans, retrieved_ids, new_ctx)

    def _resync_ids(self, answer: str, ids: list[int], context: list[Document]) -> list[int]:
        """Sau khi sinh lại câu trả lời: chạy lại `keep_used_sources` với câu trả lời MỚI, để đoạn nó dùng (trong ngữ
        cảnh mới) có mặt ở hạng 6–10. Trước đây id được chọn theo câu trả lời cũ (public-0016 #17 giữ đoạn 91 chỉ vì
        câu cũ dùng link MPC). Không đụng top-5 → retrieval giữ nguyên."""
        if not getattr(self, "keep_answer_sources", False):
            return ids
        texts = {i: d.text for i, d in self.docs.items()}
        candidates = [d.id for d in context if d.id not in ids]
        return keep_used_sources(answer, ids, candidates, texts, set(ids[:5]))

    def run(self, query: str) -> RAGOutput:
        out = super().run(query)
        new_answer, new_ids = self._refresh_answer_if_added(query, out.answer, out.retrieved_ids)
        return RAGOutput(answer=new_answer, retrieved_ids=new_ids)


class S2BPipelineR13(S2BPipelineR12):
    """Thí nghiệm R13 = R11 + kiểm tra toàn corpus cho MỌI câu (không trigger), tách theo từng đối tượng, được bổ sung
    nhiều đoạn (`CHECK_PROMPT_R13`); đoạn lấp không được trích bị thay trước. Câu trả lời y hệt R11/R12.
    Chi phí: +1 lần gọi (~51k ký tự, ~0.0032 USD) cho MỌI câu. So với `S2BPipelineR12` (#15, retrieval 94.82)."""

    def finalize_ids(
        self, answer: str, ids: list[int], allowed: list[int], tiers: list[list[int]]
    ) -> list[int]:
        ids = S2BPipelineR11.finalize_ids(
            self, answer, ids, allowed, tiers
        )  # bỏ qua lượt kiểm tra R12
        info = getattr(self, "_check", None)
        if not info:
            return ids
        top5 = ids[:5]
        raw = self.llm.chat(
            check_messages_r13(info["query"], info["aspects1"], top5, list(self.docs.values())),
            model=self.llm_model,
        )
        added = parse_supplements_r13(raw, {i: d.text for i, d in self.docs.items()}, top5)
        cited = {d for tier in tiers for d in tier}
        return insert_supplements_r13(ids, added, self._agreed, cited)


class S2BPipelineR11MBR(S2BPipelineMBR):
    """Thí nghiệm R11+MBR = câu trả lời chọn theo đồng thuận 4 bản (MBR, #12) + xếp lại top-5 theo R11 (#13).

    Hai phần độc lập: MBR chỉ đổi câu trả lời (+ hạng 6–10 qua `keep_used_sources`), R11 chỉ đổi thứ tự top-5.
    Public: 4 lần đọc đều trúng cache → 0 USD. Ngày thi: 4 lần gọi/câu (lần đọc A4 được đọc lại từ cache, không tốn).
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineR11MBR`. So với `S2BPipelineR11` (bài nộp #13, 89.42)."""

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        merged = super().extra_tiers(query, context, allowed, tiers)
        # MBR đã gọi đúng messages này → LLMClient trả từ cache; cần tầng chính riêng của lần 2 cho R11
        raw = self.llm.chat(
            build_messages(query, context, self.second_prompt), model=self.llm_model
        )
        _, second = split_aspects(raw, allowed)
        self._agreed = set(tiers[0] if tiers else []) & set(second[0] if second else [])
        return merged

    def finalize_ids(
        self, answer: str, ids: list[int], allowed: list[int], tiers: list[list[int]]
    ) -> list[int]:
        ids = super().finalize_ids(answer, ids, allowed, tiers)
        return consensus_first(ids, getattr(self, "_agreed", set()))


class S2BPipelineR12MBR(S2BPipelineR11MBR):
    """Thí nghiệm R12+MBR = câu trả lời MBR (như #14) + id của R12 (như #15: R11 + kiểm tra toàn corpus câu gắn cờ).

    Hai phần độc lập: MBR chỉ đổi câu trả lời (+ hạng 6–10 qua `keep_used_sources`), R12 chỉ đổi top-5.
    Public: mọi lần gọi đã có trong cache → 0 USD. Ngày thi: 4 lần đọc/câu + 1 lần kiểm tra cho câu gắn cờ (+ thêm dấu).
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineR12MBR`. So với #14 (89.56) và #15 (retrieval 94.82)."""

    keep_answer_sources = True

    def extra_tiers(
        self, query: str, context: list[Document], allowed: list[int], tiers: list[list[int]]
    ) -> list[list[int]]:
        merged = super().extra_tiers(query, context, allowed, tiers)
        S2BPipelineR12._capture_check(self, query, context, allowed, tiers)
        return merged

    def finalize_ids(
        self, answer: str, ids: list[int], allowed: list[int], tiers: list[list[int]]
    ) -> list[int]:
        ids = super().finalize_ids(answer, ids, allowed, tiers)
        return S2BPipelineR12._evidence_check(self, ids, answer)

    def _refresh_answer_if_added(
        self, query: str, answer: str, retrieved_ids: list[int]
    ) -> tuple[str, list[int]]:
        """Với MBR, khi có đoạn bổ sung ngoài 20 đoạn ban đầu, chạy MBR trên context mới để cập nhật câu trả lời."""
        added = getattr(self, "_last_added", None)
        if not added:
            return answer, retrieved_ids
        old_ctx = getattr(self, "_last_context", [])
        new_ctx = [self.docs[i] for i in added] + [d for d in old_ctx if d.id not in added]
        new_ctx = new_ctx[: self.n_context]
        allowed = [d.id for d in new_ctx]

        candidates = []
        for prompt in [self.system_prompt, self.second_prompt, *self.candidate_prompts]:
            raw = self.llm.chat(build_messages(query, new_ctx, prompt), model=self.llm_model)
            # như S2BPipelineMBR: chỉ E2 ghi 1 dòng NGUỒN; A/A4/F có phần "Ý HỎI" phải tách bằng split_aspects
            parse = split_sources if prompt == SYSTEM_PROMPT_E2 else split_aspects
            cand_ans, _ = parse(raw, allowed)
            candidates.append(self.finalize_answer(postprocess(cand_ans)))
        new_ans = mbr_select(candidates) or answer
        return new_ans, S2BPipelineR12._resync_ids(self, new_ans, retrieved_ids, new_ctx)

    def run(self, query: str) -> RAGOutput:
        out = super().run(query)
        new_answer, new_ids = self._refresh_answer_if_added(query, out.answer, out.retrieved_ids)
        return RAGOutput(answer=new_answer, retrieved_ids=new_ids)


class S2BPipelineR12MBRParen(S2BPipelineR12MBR):
    """Thí nghiệm 08/11 = R12MBR + `mirror_aliases`: người hỏi viết "proptit"/"clb mm"/"clb game" → câu trả lời ghi
    thêm tên đó trong ngoặc, ví dụ "CLB Lập Trình PTIT (ProPTIT)". Chỉ hậu xử lý câu trả lời, id giữ nguyên;
    cùng prompt với R12MBR → chạy lại trúng cache (0 USD). Judge trên câu bị đổi: Relevancy +0.025 (dev/holdout),
    quy ra toàn bộ ≈ 0 (dưới điều kiện đặt trước) — nộp thử vì private tính lần nộp cao nhất.
    Chạy: `--pipeline s2b_rag.pipeline:S2BPipelineR12MBRParen`."""

    def run(self, query: str) -> RAGOutput:
        out = super().run(query)
        return RAGOutput(answer=mirror_aliases(query, out.answer), retrieved_ids=out.retrieved_ids)


class S2BPipeline(S2BPipelineR12MBRParen):
    """CẤU HÌNH NỘP BÀI (09/11) = R12MBR + paren — bài Private điểm cao nhất **90.05** (BTC tính lần cao nhất).
    Paren chỉ thêm tên CLB người hỏi dùng vào ngoặc (11/42 câu private); id giữ nguyên. R12MBR không paren: private
    89.95 (nộp lại 89.94). Trên dev/holdout paren không đổi Noise → phần +0.10 nằm trong nhiễu (docs/journal.md).

    R12MBR (chốt 07/11): public #17 89.99 (nộp lại 89.92); retrieval 94.82, SP 96.15, ROUGE-L 63.70.
    = R11 (đoạn được 2 lần đọc đồng thuận lên đầu top-5) + R12 (câu bị gắn cờ → kiểm tra toàn corpus, bổ sung đoạn có
    trích nguyên văn) + MBR (chọn câu trả lời đồng thuận trong 4 bản A/A4/E2/F) + khi có đoạn bổ sung thì sinh lại câu
    trả lời và giữ ở hạng 6–10 đoạn mà câu trả lời dùng (`keep_answer_sources`).
    Mỗi câu gọi haiku 4 lần, +1 lần kiểm tra toàn corpus nếu bị gắn cờ (public 11/42), +4 lần sinh lại nếu được bổ
    sung đoạn (public 1/42), +1 lần thêm dấu nếu câu không dấu (R10); chatbot chỉ cần lần 1 (+ thêm dấu).

    Ngày thi: `neorag run private_test --pipeline s2b_rag.pipeline:S2BPipeline`. Đổi cấu hình nộp bài = đổi lớp cha
    của class này, không sửa các class thí nghiệm ở trên."""

    keep_answer_sources = True
