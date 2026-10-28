import unicodedata

from s2b_rag.textnorm import strip_accents


def test_strip_accents_vietnamese():
    assert strip_accents("Câu lạc bộ Lập Trình PTIT") == "Cau lac bo Lap Trinh PTIT"
    assert strip_accents("Đội đạt giải Nhất") == "Doi dat giai Nhat"
    assert strip_accents("thuỷ, hoà, thủy, hòa") == "thuy, hoa, thuy, hoa"


def test_strip_accents_accepts_decomposed_input():
    assert strip_accents(unicodedata.normalize("NFD", "Đa phương tiện")) == "Da phuong tien"


def test_strip_accents_keeps_ascii_unchanged():
    text = "e hoc nganh da phuong tien (dpt), 09/10/2011 ok?"
    assert strip_accents(text) == text
