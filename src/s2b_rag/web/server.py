"""Server cho chatbot demo (phương án B: Starlette + trang HTML tự viết). Chạy: `python -m s2b_rag.web`.

Cùng `S2BPipeline` (A5) với bài nộp: `retrieve()` (nguồn, câu viết lại, thời gian) rồi `stream_answer()` = haiku lần 1
(chữ hiện dần, đã giấu phần "Ý HỎI"/"NGUỒN"), cuối cùng `finish_answer()` (hậu xử lý y như bài nộp). Chatbot không gọi
lần đọc thứ 2 của A5 (lần đó chỉ phục vụ id nộp bài).

API: POST /api/ask {"query": "..."} trả về NDJSON (mỗi dòng một JSON) theo thứ tự:
  (câu xã giao / phép tính, xét cả câu đã viết lại) {"type": "smalltalk", "answer": "...", "rewritten": ..., "seconds": ...}
  {"type": "retrieval", "rewritten": ..., "sources": [...], "seconds": ..., "n_context": 20}
  {"type": "token", "text": "..."}            (nhiều lần)
  {"type": "done", "answer": "...", "refusal": bool, "sources": [... đoạn được trích trước, "cited": true], "timings": {...}}   hoặc   {"type": "error", "message": "..."}
NDJSON qua `fetch` thay vì SSE/EventSource: EventSource chỉ gửi được GET, còn câu hỏi gửi bằng POST.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from s2b_rag.web.smalltalk import smalltalk_reply

STATIC = Path(__file__).parent / "static"
MAX_QUERY_CHARS = 500
SNIPPET_CHARS = 220


def _since(start: float) -> float:
    return round(time.perf_counter() - start, 3)


# Mẫu câu từ chối của prompt ("Không có thông tin: viết \"Tài liệu hiện có không có thông tin về ...\"") và câu dự phòng.
_REFUSAL = re.compile(r"^tài liệu hiện có không có thông tin", re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    """Câu trả lời là từ chối toàn phần (tài liệu không có thông tin), không phải trả lời một phần."""
    return bool(_REFUSAL.match(answer.strip()))


def _source(d, cited: bool = False) -> dict:
    snippet = d.text[:SNIPPET_CHARS] + ("…" if len(d.text) > SNIPPET_CHARS else "")
    return {"id": d.id, "club": d.club, "topic": d.topic, "snippet": snippet, "cited": cited}


def ask_events(pipeline, query: str) -> Iterator[dict]:
    """Các sự kiện trả lời một câu hỏi. Tách khỏi HTTP để test được bằng pipeline giả."""
    start = time.perf_counter()
    # 1) câu xã giao / không phải câu hỏi: trả lời ngay, không tìm tài liệu, không gọi haiku
    if (reply := smalltalk_reply(query)) is not None:
        yield {"type": "smalltalk", "answer": reply, "rewritten": None, "seconds": _since(start)}
        return
    try:
        r = pipeline.retrieve(query)
        # 2) câu không dấu đã được haiku viết lại ("ni hao" → "Xin chào"): áp lại luật trên câu viết lại → bỏ qua lần
        #    gọi haiku trả lời (~3–4 s) và không cần hiện nguồn
        if r.rewritten and (reply := smalltalk_reply(r.rewritten)) is not None:
            yield {
                "type": "smalltalk",
                "answer": reply,
                "rewritten": r.rewritten,
                "seconds": _since(start),
            }
            return
        yield {
            "type": "retrieval",
            "rewritten": r.rewritten,
            "seconds": round(r.seconds, 2),
            "n_context": len(r.context),  # số đoạn haiku được đọc (A5: 20)
            "sources": [_source(d) for d in r.docs],
        }
        llm_start = time.perf_counter()
        first_token = None
        stream = pipeline.stream_answer(
            query, r.context
        )  # chỉ nhả phần câu trả lời (giấu "Ý HỎI"/"NGUỒN")
        for piece in stream:
            if first_token is None:
                first_token = time.perf_counter() - llm_start
            yield {"type": "token", "text": piece}
        # bản cuối: cùng hậu xử lý với bài nộp (G2 + A2); nguồn = đoạn haiku trích trước, rồi các đoạn còn lại
        answer, cited = pipeline.finish_answer(stream.raw, r.context)
        by_id = {d.id: d for d in r.context}
        sources = [_source(by_id[i], cited=True) for i in cited if i in by_id]
        sources += [_source(d) for d in r.docs if d.id not in cited]
        total = time.perf_counter() - start
        yield {
            "type": "done",
            "answer": answer,
            "refusal": is_refusal(
                answer
            ),  # giao diện hiện câu hỏi gợi ý khi tài liệu không có thông tin
            "sources": sources,
            "timings": {
                "retrieval": round(r.seconds, 2),
                "first_token": round(first_token or 0.0, 2),
                "llm": round(time.perf_counter() - llm_start, 2),
                "total": round(total, 2),
            },
        }
    except (
        Exception
    ) as exc:  # gateway lỗi, hết ngân sách...: báo gọn cho người dùng, không lộ traceback
        yield {
            "type": "error",
            "message": f"Không trả lời được lúc này ({type(exc).__name__}). Thử lại sau.",
        }


def create_app(pipeline_factory) -> Starlette:
    """`pipeline_factory()` trả về pipeline đã prepare + index; gọi một lần khi server khởi động."""
    state: dict = {}
    lock = (
        threading.Lock()
    )  # một câu hỏi một lúc: reranker trên GPU và ngân sách đều không cần chạy song song

    @asynccontextmanager
    async def lifespan(app):
        state["pipeline"] = pipeline_factory()
        yield

    async def index(request: Request):
        return FileResponse(STATIC / "index.html")

    async def health(request: Request):
        return JSONResponse({"ready": "pipeline" in state})

    async def ask(request: Request):
        body = await request.json()
        query = str(body.get("query", "")).strip()
        if not query:
            return JSONResponse({"error": "Câu hỏi trống"}, status_code=400)
        query = query[:MAX_QUERY_CHARS]

        def lines() -> Iterator[str]:
            with lock:
                for event in ask_events(state["pipeline"], query):
                    yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    routes = [
        Route("/", index),
        Route("/api/health", health),
        Route("/api/ask", ask, methods=["POST"]),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def load_pipeline():
    """Pipeline nộp bài, nạp sẵn reranker và index — mất ~30 s một lần khi khởi động."""
    from neorag.data import load_corpus
    from s2b_rag.pipeline import S2BPipeline

    pipeline = S2BPipeline()
    pipeline.prepare()
    pipeline.index(load_corpus())
    return pipeline
