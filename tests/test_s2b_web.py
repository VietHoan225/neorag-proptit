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


# ---- Memory ngắn hạn: câu hỏi nối tiếp ----

HISTORY = [
    {
        "q": "CLB Multimedia PTIT tuyển thành viên khi nào?",
        "a": "CLB Multimedia PTIT tuyển vào tháng 8/2026.",
    }
]


def test_clean_history_keeps_last_two_valid_turns_and_truncates():
    from s2b_rag.web.memory import MAX_TURN_CHARS, clean_history

    raw = [
        {"q": "a", "a": "b"},
        "rác",
        {"q": "c"},
        {"q": "d", "a": "x" * 1000},
        {"q": "e", "a": "f"},
    ]
    assert clean_history(raw) == [{"q": "d", "a": "x" * MAX_TURN_CHARS}, {"q": "e", "a": "f"}]
    assert clean_history(None) == [] and clean_history({"q": "a"}) == []


def test_condense_uses_history_and_falls_back_to_query():
    from s2b_rag.web.memory import condense

    class LLM:
        def __init__(self, reply):
            self.reply, self.calls = reply, []

        def chat(self, messages, model):
            self.calls.append(messages)
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    llm = LLM('{"query": "CLB Nhà Sáng Tạo Game PTIT (GCC) tuyển thành viên khi nào?"}')
    assert (
        condense(llm, "m", "the con gcc", HISTORY)
        == "CLB Nhà Sáng Tạo Game PTIT (GCC) tuyển thành viên khi nào?"
    )
    assert (
        "CLB Multimedia PTIT tuyển thành viên khi nào?" in llm.calls[0][1]["content"]
    )  # có lượt trước
    assert condense(LLM(""), "m", "the con gcc", HISTORY) == "the con gcc"
    assert condense(LLM(ConnectionError()), "m", "the con gcc", HISTORY) == "the con gcc"


def test_followup_is_condensed_before_retrieval_and_answer():
    class Recording(FakePipeline):
        llm_model = "claude-haiku-5-5"

        def __init__(self):
            super().__init__()
            self.asked = []
            self.llm = type(
                "L",
                (),
                {
                    "chat": lambda s, m, model: (
                        '{"query": "CLB Nhà Sáng Tạo Game PTIT (GCC) tuyển khi nào?"}'
                    )
                },
            )()

        def retrieve(self, query):
            self.asked.append(query)
            return super().retrieve(query)

        def stream_answer(self, query, context):
            self.asked.append(query)
            return super().stream_answer(query, context)

    p = Recording()
    events = list(ask_events(p, "the con gcc thi sao", HISTORY))
    assert (
        p.asked == ["CLB Nhà Sáng Tạo Game PTIT (GCC) tuyển khi nào?"] * 2
    )  # tìm và trả lời bằng câu đầy đủ
    assert events[0]["followup"] is True and events[0]["rewritten"] == p.asked[0]
    # không có lịch sử → không viết lại theo ngữ cảnh
    p2 = Recording()
    events = list(ask_events(p2, "clb mm tuyen tv khi nao"))
    assert p2.asked[0] == "clb mm tuyen tv khi nao" and events[0]["followup"] is False


def test_with_history_every_question_is_condensed_without_history_none_is():
    """Không đoán "câu nối tiếp" bằng luật: có lịch sử là haiku viết lại (câu tự đủ nghĩa thì chỉ được thêm dấu)."""

    class Counting(FakePipeline):
        llm_model = "claude-haiku-5-5"

        def __init__(self):
            super().__init__()
            self.condense_calls = 0
            outer = self

            class L:
                def chat(self, messages, model):
                    outer.condense_calls += 1
                    return '{"query": "CLB Nhà Sáng Tạo Game PTIT (GCC) có phòng riêng không?"}'

            self.llm = L()

    p = Counting()
    list(
        ask_events(p, "GCC có không?", HISTORY)
    )  # có tên CLB, không có từ nối: vẫn phải hiểu theo ngữ cảnh
    list(ask_events(p, "GCC có phòng riêng không?"))  # không có lịch sử: không gọi
    assert p.condense_calls == 1


def test_condense_ignores_explanations_and_takes_json_query_only():
    """Lỗi thật khi chạy thử: haiku trả về lời giải thích thay vì chỉ câu hỏi → không được dùng nguyên văn."""
    from s2b_rag.web.memory import parse_condensed

    messy = (
        'Câu hỏi mới: "GCC có không?" Đây là câu hỏi nối tiếp, cần thay "GCC" bằng tên đầy đủ và làm rõ ý hỏi: '
        "CLB Nhà Sáng Tạo Game PTIT có phòng riêng để chụp ảnh không."
    )
    assert parse_condensed(messy) is None  # không có JSON → giữ câu gốc
    wrapped = 'Đây là câu viết lại:\n```json\n{"query": "CLB Nhà Sáng Tạo Game PTIT có phòng riêng không?"}\n```'
    assert parse_condensed(wrapped) == "CLB Nhà Sáng Tạo Game PTIT có phòng riêng không?"
    assert parse_condensed('{"query": ""}') is None and parse_condensed("{hỏng") is None
