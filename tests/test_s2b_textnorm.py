import unicodedata

from s2b_rag.textnorm import analyze, expand_query, has_diacritics, strip_accents


def test_strip_accents_vietnamese():
    assert strip_accents("Câu lạc bộ Lập Trình PTIT") == "Cau lac bo Lap Trinh PTIT"
    assert strip_accents("Đội đạt giải Nhất") == "Doi dat giai Nhat"
    assert strip_accents("thuỷ, hoà, thủy, hòa") == "thuy, hoa, thuy, hoa"


def test_strip_accents_accepts_decomposed_input():
    assert strip_accents(unicodedata.normalize("NFD", "Đa phương tiện")) == "Da phuong tien"


def test_strip_accents_keeps_ascii_unchanged():
    text = "e hoc nganh da phuong tien (dpt), 09/10/2011 ok?"
    assert strip_accents(text) == text


def test_expand_query_abbreviations_and_months():
    assert expand_query("tv clb lt ptit ko?") == "thành viên clb lập trình ptit không"
    assert expand_query("Pho CN nk 2024-2025") == "pho chủ nhiệm nhiệm kỳ 2024 2025"
    assert expand_query("cuoi t10 nop don") == "cuoi tháng 10 nop don"


def test_analyze_noaccent_with_bigrams():
    assert analyze("Thành lập ngày", accents=False, n_max=2) == [
        "thanh", "lap", "ngay", "thanh_lap", "lap_ngay",
    ]  # fmt: skip


def test_has_diacritics():
    assert has_diacritics("đi học")
    assert has_diacritics("Đ")
    assert not has_diacritics("di hoc clb lt ptit 2026?")
