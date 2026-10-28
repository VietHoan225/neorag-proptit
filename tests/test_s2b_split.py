from collections import Counter

from neorag.data import read_jsonl
from s2b_rag.eval.split import N_HOLDOUT, TRAIN_PATH, stratified_split, without_accents


def _train():
    return read_jsonl(TRAIN_PATH)


def test_split_is_a_partition_of_train():
    rows = _train()
    dev, holdout = stratified_split(rows, N_HOLDOUT)
    dev_ids, holdout_ids = {r["id"] for r in dev}, {r["id"] for r in holdout}
    assert len(holdout) == N_HOLDOUT
    assert not dev_ids & holdout_ids
    assert dev_ids | holdout_ids == {r["id"] for r in rows}


def test_every_type_is_in_both_sets_with_similar_ratio():
    rows = _train()
    dev, holdout = stratified_split(rows, N_HOLDOUT)
    total, held = Counter(r["type"] for r in rows), Counter(r["type"] for r in holdout)
    frac = N_HOLDOUT / len(rows)
    for qtype, n in total.items():
        assert held[qtype] >= 1, qtype
        assert held[qtype] < n, qtype  # dev cũng phải có
        assert abs(held[qtype] - n * frac) <= 1, qtype


def test_split_is_deterministic_and_depends_on_seed():
    rows = _train()

    def holdout_ids(seed):
        return [r["id"] for r in stratified_split(rows, N_HOLDOUT, seed)[1]]

    assert holdout_ids(2026) == holdout_ids(2026)
    assert holdout_ids(1) != holdout_ids(2026)


def test_without_accents_only_changes_query():
    row = {"id": "x", "query": "Đăng ký ở đâu?", "answer": "Tại fanpage.", "type": "trực tiếp"}
    (out,) = without_accents([row])
    assert out == {**row, "query": "Dang ky o dau?"}
    assert row["query"] == "Đăng ký ở đâu?"  # không sửa dữ liệu gốc
