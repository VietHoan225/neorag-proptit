from neorag.data import Document
from s2b_rag.aliases import GCC, MULTIMEDIA, PROPTIT, S2B, detect_clubs, enrich
from s2b_rag.retrieval.club_boost import ClubBoostRetriever, EnrichedRetriever


def test_detect_clubs_from_aliases_without_diacritics():
    assert detect_clubs("e muon vao clb mm co dc ko") == [MULTIMEDIA]
    assert detect_clubs("tv clb lt ptit phai lam gi") == [
        PROPTIT
    ]  # "clb lt" mở rộng thành "clb lập trình"
    assert detect_clubs("CLB làm game có mấy ban?") == [GCC]
    assert detect_clubs("ProPTIT với GCC khác gì nhau") == [PROPTIT, GCC]
    assert detect_clubs("Sinh nhật S2B năm 2025") == [S2B]


def test_detect_clubs_ignores_ambiguous_words():
    assert detect_clubs("học lập trình game thì cần gì") == []
    assert detect_clubs("mmo là gì") == []  # "mm" phải là một từ riêng


def test_enrich_prefixes_club_aliases_and_topic():
    out = enrich(MULTIMEDIA, "Liên hệ", "Email của CLB là ...")
    assert out.startswith("CLB Multimedia PTIT (MPC")
    assert "Liên hệ. Email của CLB là ..." in out


class Fixed:
    def __init__(self, ranking):
        self.ranking, self.indexed = ranking, None

    def index(self, corpus):
        self.indexed = corpus

    def search(self, query):
        return [(doc_id, 1.0 / rank) for rank, doc_id in enumerate(self.ranking, 1)]


CORPUS = [
    Document(1, PROPTIT, "t", "a"),
    Document(2, PROPTIT, "t", "b"),
    Document(3, PROPTIT, "t", "c"),
    Document(4, PROPTIT, "t", "d"),
    Document(5, PROPTIT, "t", "e"),
    Document(6, GCC, "t", "f"),
    Document(7, MULTIMEDIA, "t", "g"),
]


def test_boost_moves_detected_club_up_without_filtering():
    r = ClubBoostRetriever(Fixed([1, 2, 3, 4, 5, 6, 7]), weight=100.0)
    r.index(CORPUS)
    ranked = [i for i, _ in r.search("clb mm có phòng riêng không")]
    assert ranked[0] == 7
    assert sorted(ranked) == [1, 2, 3, 4, 5, 6, 7]  # không lọc bỏ đoạn nào


def test_quota_gives_each_compared_club_a_top5_slot():
    r = ClubBoostRetriever(Fixed([1, 2, 3, 4, 5, 6, 7]), weight=0.0)
    r.index(CORPUS)
    ranked = [i for i, _ in r.search("ProPTIT với GCC khác nhau thế nào")]
    assert 6 in ranked[:5]  # GCC có suất trong top-5
    assert sum(i <= 5 for i in ranked[:5]) == 4  # ProPTIT vẫn giữ suất
    assert ranked[5] == 5  # đoạn nhường chỗ xuống ngay sau top-5


def test_no_club_mentioned_keeps_order():
    r = ClubBoostRetriever(Fixed([3, 1, 2]), weight=1.0)
    r.index(CORPUS[:3])
    assert [i for i, _ in r.search("hỏi chung chung")] == [3, 1, 2]


def test_enriched_retriever_indexes_prefixed_text():
    base = Fixed([1])
    EnrichedRetriever(base).index(CORPUS[:1])
    assert base.indexed[0].text.startswith("CLB Lập Trình PTIT (ProPTIT")
    assert base.indexed[0].id == 1


def test_detect_clubs_multi_treats_three_clubs_phrase_as_all_clubs():
    from s2b_rag.aliases import detect_clubs_multi

    all_three = [PROPTIT, MULTIMEDIA, GCC]
    assert detect_clubs_multi("Trong 3 CLB thuộc S2B, CLB nào lâu đời nhất?") == [S2B, *all_three]
    assert detect_clubs_multi("gui mail hoi ca 3 clb trong s2b") == [S2B, *all_three]
    assert detect_clubs_multi("trong ba câu lạc bộ thì chỗ nào hợp với em") == all_three
    # không nói "ba CLB" thì giữ như detect_clubs; "13 clb" không phải "3 clb"
    assert detect_clubs_multi("CLB nào trong S2B tuyển ít người nhất?") == [S2B]
    assert detect_clubs_multi("có 13 clb tham gia") == []


def test_quota_after_reranker_restores_slot_that_reranker_removed():
    # tầng dưới (vd reranker) trả top-5 toàn ProPTIT dù câu hỏi so sánh ProPTIT với GCC
    r = ClubBoostRetriever(
        Fixed([1, 2, 3, 4, 5, 6, 7]), weight=0.0, detect=lambda q: [PROPTIT, GCC]
    )
    r.index(CORPUS)
    ranked = [i for i, _ in r.search("bất kỳ")]
    assert ranked[:5] == [1, 2, 3, 4, 6]  # weight=0: không đổi thứ tự nào khác ngoài suất quota
