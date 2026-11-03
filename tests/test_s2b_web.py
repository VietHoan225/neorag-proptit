from neorag.data import Document
from s2b_rag.pipeline import Retrieval
from s2b_rag.web.server import ask_events

DOCS = [Document(i, "CLB Multimedia PTIT", "Tuyển thành viên", "x" * 300) for i in (91, 93, 92)]


class FakePipeline:
    def __init__(self, pieces=("**Đợt tuyển**", " mở\n", "tháng 9."), fail=False):
        self.pieces, self.fail = pieces, fail

    def retrieve(self, query):
        if self.fail:
            raise ConnectionError("gateway down")
        return Retrieval(DOCS, DOCS[:2], "CLB Multimedia PTIT tuyển thành viên khi nào?", 0.4)

    def stream_answer(self, query, context):
        assert context == DOCS[:2]  # đúng các đoạn trong prompt
        yield from self.pieces


def test_events_order_retrieval_then_tokens_then_done():
    events = list(ask_events(FakePipeline(), "clb mm tuyen tv khi nao"))
    assert [e["type"] for e in events] == ["retrieval", "token", "token", "token", "done"]
    head, done = events[0], events[-1]
    assert head["rewritten"] == "CLB Multimedia PTIT tuyển thành viên khi nào?"
    assert [s["id"] for s in head["sources"]] == [91, 93, 92]
    assert [s["in_prompt"] for s in head["sources"]] == [True, True, False]
    assert head["sources"][0]["snippet"].endswith("…") and len(head["sources"][0]["snippet"]) == 221
    assert done["answer"] == "Đợt tuyển mở. tháng 9."  # cùng hậu xử lý G2 với bài nộp
    assert set(done["timings"]) == {"retrieval", "first_token", "llm", "total"}


def test_gateway_error_becomes_friendly_error_event():
    events = list(ask_events(FakePipeline(fail=True), "câu hỏi"))
    assert [e["type"] for e in events] == ["error"]
    assert "ConnectionError" in events[0]["message"] and "Traceback" not in events[0]["message"]
