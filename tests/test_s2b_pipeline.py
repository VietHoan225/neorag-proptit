import re

from conftest import FakeLLM
from neorag.data import Query
from neorag.predict import predict
from s2b_rag.generate import build_messages, format_context, postprocess
from s2b_rag.pipeline import S2BPipeline


def test_postprocess_removes_markdown_and_joins_lines():
    raw = "## Trả lời\n**CLB Lập Trình PTIT** thành lập ngày 09/10/2011\n- Phương châm: Chia sẻ"
    assert postprocess(raw) == (
        "Trả lời. CLB Lập Trình PTIT thành lập ngày 09/10/2011. Phương châm: Chia sẻ"
    )


def test_postprocess_keeps_numbers_links_and_plain_text():
    text = "Đăng ký tại https://mpc.proptit.com trước 10/09/2026, phí 0 đồng."
    assert postprocess(text) == text
    assert postprocess("") == ""


def test_context_lists_id_club_topic_in_rank_order(corpus):
    ctx = format_context([corpus[1], corpus[0]])
    assert ctx.startswith("[2] CLB Multimedia PTIT · Giới thiệu\n")
    assert ctx.index("[2]") < ctx.index("[1]")
    messages = build_messages("Multimedia thành lập khi nào?", corpus)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "CÂU HỎI: Multimedia thành lập khi nào?" in messages[1]["content"]


def test_pipeline_end_to_end_with_fake_llm_and_bm25(corpus):
    llm = FakeLLM(lambda messages: "**CLB Multimedia PTIT** thành lập ngày 10/11/2012.")
    pipeline = S2BPipeline(retriever="bm25_noaccent_bi", n_context=2, llm=llm)
    pipeline.prepare()
    pipeline.index(corpus)
    (pred,) = predict(pipeline, [Query("q", "clb multimedia thanh lap ngay nao")], max_retrieved=10)
    assert pred.answer == "CLB Multimedia PTIT thành lập ngày 10/11/2012."
    assert pred.retrieved_ids[0] == 2
    assert len(pred.retrieved_ids) == 3  # corpus giả chỉ có 3 đoạn
    prompt = llm.calls[0][1]["content"]
    assert re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE) == ["2", "1"]  # đúng 2 đoạn, theo hạng
    assert "temperature" not in str(llm.calls)


def test_prepare_finds_reranker_in_nested_layers(monkeypatch):
    import s2b_rag.retrieval.rerank as rerank

    loaded = []
    monkeypatch.setattr(rerank, "load_cross_encoder", lambda name=None: loaded.append(name))
    S2BPipeline(retriever="hybrid_r7", llm=FakeLLM()).prepare()  # reranker nằm sâu bên trong
    S2BPipeline(retriever="bm25_noaccent_bi", llm=FakeLLM()).prepare()  # không có reranker
    assert loaded == ["BAAI/bge-reranker-v2-m3"]


class FakeStreamClient:
    """Client giả kiểu openai: `chat.completions.create(..., stream=True)` nhả từng mẩu chữ."""

    def __init__(self, pieces):
        self.pieces, self.calls = pieces, []
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)

        def chunk(text):
            delta = type("Delta", (), {"content": text})()
            return type("Chunk", (), {"choices": [type("Choice", (), {"delta": delta})()]})()

        return [chunk(p) for p in self.pieces] + [type("Chunk", (), {"choices": []})()]


def test_retrieve_returns_docs_context_and_no_rewrite(corpus):
    pipeline = S2BPipeline(retriever="bm25_noaccent_bi", n_context=2, llm=FakeLLM())
    pipeline.index(corpus)
    r = pipeline.retrieve("clb multimedia thanh lap ngay nao")
    assert r.ids[0] == 2 and len(r.ids) == 3
    assert [d.id for d in r.context] == r.ids[:2]
    assert r.rewritten is None and r.seconds >= 0


def test_stream_answer_uses_same_prompt_as_run_and_joins_to_same_answer(corpus):
    full = "**CLB Multimedia PTIT** thành lập\nngày 10/11/2012."
    llm = FakeLLM(lambda messages: full)
    client = FakeStreamClient(["**CLB Multimedia", " PTIT** thành lập\n", "ngày 10/11/2012."])
    pipeline = S2BPipeline(retriever="bm25_noaccent_bi", n_context=2, llm=llm, stream_client=client)
    pipeline.index(corpus)
    query = "clb multimedia thanh lap ngay nao"

    expected = pipeline.run(query).answer
    r = pipeline.retrieve(query)
    streamed = postprocess("".join(pipeline.stream_answer(query, r.context)))

    assert streamed == expected
    (call,) = client.calls
    assert call["messages"] == llm.calls[0]  # cùng prompt với bài nộp
    assert call["stream"] is True and call["model"] == "claude-haiku-5-5"
    assert "temperature" not in call
