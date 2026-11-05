from neorag.data import Document
from s2b_rag.pipeline import Retrieval
from s2b_rag.web.server import ask_events

DOCS = [Document(i, "CLB Multimedia PTIT", "Tuyển thành viên", "x" * 300) for i in (91, 93, 92)]


class FakeStream:
    def __init__(self, pieces, raw):
        self.pieces, self.raw = pieces, raw

    def __iter__(self):
        yield from self.pieces


class FakePipeline:
    def __init__(self, fail=False):
        self.fail = fail

    def retrieve(self, query):
        if self.fail:
            raise ConnectionError("gateway down")
        return Retrieval(DOCS, DOCS[:2], "CLB Multimedia PTIT tuyển thành viên khi nào?", 0.4)

    def stream_answer(self, query, context):
        assert context == DOCS[:2]  # đúng các đoạn trong prompt
        return FakeStream(
            ["Đợt tuyển", " mở tháng 9."],
            raw="Ý HỎI:\nÝ 1: x | NGUỒN: 93\nTRẢ LỜI:\nĐợt tuyển mở tháng 9.",
        )

    def finish_answer(self, raw, context):
        assert "Ý HỎI" in raw  # nhận phản hồi đầy đủ, kể cả phần bị giấu
        return "Đợt tuyển mở tháng 9.", [93]


def test_events_order_retrieval_then_tokens_then_done():
    events = list(ask_events(FakePipeline(), "clb mm tuyen tv khi nao"))
    assert [e["type"] for e in events] == ["retrieval", "token", "token", "done"]
    head, done = events[0], events[-1]
    assert head["rewritten"] == "CLB Multimedia PTIT tuyển thành viên khi nào?"
    assert [s["id"] for s in head["sources"]] == [91, 93, 92] and head["n_context"] == 2
    assert head["sources"][0]["snippet"].endswith("…") and len(head["sources"][0]["snippet"]) == 221
    assert done["answer"] == "Đợt tuyển mở tháng 9."
    # đoạn haiku trích (93) lên đầu, đánh dấu cited; các đoạn còn lại giữ thứ tự retriever
    assert [(s["id"], s["cited"]) for s in done["sources"]] == [
        (93, True),
        (91, False),
        (92, False),
    ]
    assert set(done["timings"]) == {"retrieval", "first_token", "llm", "total"}


def test_gateway_error_becomes_friendly_error_event():
    events = list(ask_events(FakePipeline(fail=True), "câu hỏi"))
    assert [e["type"] for e in events] == ["error"]
    assert "ConnectionError" in events[0]["message"] and "Traceback" not in events[0]["message"]


def test_smalltalk_matches_whole_message_greetings_in_any_form():
    from s2b_rag.web.smalltalk import REPLIES, smalltalk_reply

    for q in ["Hello", "hello!!", "xin chào", "chao ban a", "Chào bạn ơi", "hi bot"]:
        assert smalltalk_reply(q) == REPLIES["greet"], q
    assert smalltalk_reply("cảm ơn nhé") == REPLIES["thanks"]
    assert smalltalk_reply("Bạn là ai?") == REPLIES["who"]
    assert smalltalk_reply("tam biet") == REPLIES["bye"]


def test_smalltalk_never_swallows_real_questions():
    from s2b_rag.web.smalltalk import smalltalk_reply

    for q in [
        "chào bạn, CLB Lập Trình PTIT tuyển thành viên khi nào?",  # chào + câu hỏi thật
        "ban dao tao cua proptit la ai",  # "ban" = Ban Đào tạo
        "ban la gi trong clb multimedia",
        "clb nao hay nhat",
        "thủ đô nước Pháp là gì",  # ngoài phạm vi: vẫn đi pipeline để từ chối dựa trên tài liệu
        "ok vậy GCC tuyển khi nào",
    ]:
        assert smalltalk_reply(q) is None, q


def test_smalltalk_event_skips_retrieval_and_llm():
    class NoCallPipeline:
        def retrieve(self, query):
            raise AssertionError("không được tìm tài liệu cho câu xã giao")

    events = list(ask_events(NoCallPipeline(), "Hello"))
    assert [e["type"] for e in events] == ["smalltalk"] and events[0]["answer"].startswith(
        "Chào bạn"
    )


def test_non_questions_get_instant_scope_reply_but_real_questions_with_numbers_do_not():
    from s2b_rag.web.smalltalk import REPLIES, smalltalk_reply

    for q in ["1 + 1", "1+1 bang may", "2 x 3 = ?", "???", "😀"]:
        assert smalltalk_reply(q) == REPLIES["scope"], q
    for q in ["nam 2026 clb tron bao nhieu nam", "top 10 thanh tich proptit", "clb nao co 2 ban"]:
        assert smalltalk_reply(q) is None, q


def test_rewritten_greeting_skips_answer_call():
    """ "ni hao" không có trong luật, nhưng haiku viết lại thành "Xin chào" → trả lời ngay, không gọi haiku trả lời."""

    class RewritesToGreeting(FakePipeline):
        def retrieve(self, query):
            return Retrieval(DOCS, DOCS[:2], "Xin chào", 1.2)

        def stream_answer(self, query, context):
            raise AssertionError("không được gọi haiku trả lời cho câu chào")

    events = list(ask_events(RewritesToGreeting(), "ni hao"))
    assert [e["type"] for e in events] == ["smalltalk"]
    assert events[0]["rewritten"] == "Xin chào" and events[0]["answer"].startswith("Chào bạn")


def test_done_event_flags_full_refusal_only():
    from s2b_rag.generate import FALLBACK_ANSWER
    from s2b_rag.web.server import is_refusal

    assert is_refusal("Tài liệu hiện có không có thông tin về thủ đô nước Pháp.")
    assert is_refusal(FALLBACK_ANSWER)
    assert not is_refusal(
        "CLB tuyển tháng 9. Tài liệu không ghi số lượng."
    )  # trả lời một phần: không phải từ chối
    events = list(ask_events(FakePipeline(), "clb mm tuyen tv khi nao"))
    assert events[-1]["refusal"] is False
