"""Server cho chatbot demo (phương án B: Starlette + trang HTML tự viết). Chạy: `python -m s2b_rag.web`.

Cùng `S2BPipeline` với bài nộp: `retrieve()` (nguồn, câu viết lại, thời gian) rồi `stream_answer()` (chữ hiện dần).

API: POST /api/ask {"query": "..."} trả về NDJSON (mỗi dòng một JSON) theo thứ tự:
  {"type": "retrieval", "rewritten": ..., "sources": [...], "seconds": ...}
  {"type": "token", "text": "..."}            (nhiều lần)
  {"type": "done", "answer": "...", "timings": {...}}   hoặc   {"type": "error", "message": "..."}
NDJSON qua `fetch` thay vì SSE/EventSource: EventSource chỉ gửi được GET, còn câu hỏi gửi bằng POST.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from s2b_rag.generate import postprocess

STATIC = Path(__file__).parent / "static"
MAX_QUERY_CHARS = 500
SNIPPET_CHARS = 220


def ask_events(pipeline, query: str) -> Iterator[dict]:
    """Các sự kiện trả lời một câu hỏi. Tách khỏi HTTP để test được bằng pipeline giả."""
    start = time.perf_counter()
    try:
        r = pipeline.retrieve(query)
        yield {
            "type": "retrieval",
            "rewritten": r.rewritten,
            "seconds": round(r.seconds, 2),
            "sources": [
                {
                    "id": d.id,
                    "club": d.club,
                    "topic": d.topic,
                    "snippet": d.text[:SNIPPET_CHARS]
                    + ("…" if len(d.text) > SNIPPET_CHARS else ""),
                    "in_prompt": i < len(r.context),
                }
                for i, d in enumerate(r.docs)
            ],
        }
        llm_start = time.perf_counter()
        first_token = None
        pieces = []
        for piece in pipeline.stream_answer(query, r.context):
            if first_token is None:
                first_token = time.perf_counter() - llm_start
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        total = time.perf_counter() - start
        yield {
            "type": "done",
            "answer": postprocess("".join(pieces)),  # cùng hậu xử lý G2 với bài nộp
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
