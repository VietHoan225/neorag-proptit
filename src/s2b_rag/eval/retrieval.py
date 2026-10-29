"""Đo chất lượng truy xuất MIỄN PHÍ (không gọi LLM), dùng đúng hàm chấm chính thức của BTC.

Chạy: python -m s2b_rag.eval.retrieval --retriever dense --splits dev dev_noaccent

- Chỉ chạy tầng retrieval, không sinh câu trả lời → không tốn ngân sách haiku.
- Chấm bằng `neorag.metrics.retrieval.score_query`, cùng quy tắc với bộ chấm: câu không có đáp án
  chỉ được chấm retrieval khi gold có `relevant_ids`.
- In retrieval score = mean(Recall@5, MRR@5, NDCG@5) tổng và theo type / difficulty / club.
- "public-like" = trung bình retrieval của dev và dev_noaccent (public có rất nhiều câu không dấu).
- Ghi chi tiết từng câu vào outputs/retrieval/ và một dòng tổng vào experiments/log.csv.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from neorag.data import DATA_DIR, load_corpus, read_jsonl, write_jsonl
from neorag.metrics.retrieval import score_query
from s2b_rag.aliases import enrich
from s2b_rag.eval import explog
from s2b_rag.eval.split import OUT_DIR as SPLITS_DIR
from s2b_rag.retrieval.bm25 import BM25Retriever
from s2b_rag.retrieval.club_boost import ClubBoostRetriever, EnrichedRetriever
from s2b_rag.retrieval.dense import gateway_dense
from s2b_rag.retrieval.fusion import HybridRetriever
from s2b_rag.retrieval.rerank import CrossEncoderReranker
from s2b_rag.textnorm import analyze, expand_query

K_VALUES = [3, 5, 7]
RANK_METRICS = ["recall@5", "mrr@5", "ndcg@5"]  # 3 metric vào điểm Benchmark (configs/eval.toml)
N_RETRIEVED = 10  # số id được nộp tối đa


def bm25(accents: bool, n_max: int = 1, expand: bool = True) -> BM25Retriever:
    """Một kênh BM25: có dấu hoặc bỏ dấu, n-gram tới n_max, có/không mở rộng viết tắt (R1)."""
    return BM25Retriever(
        lambda text: analyze(text, accents, n_max), expand_query if expand else str
    )


def hybrid_equal() -> HybridRetriever:
    """R3 ban đầu: RRF đều 3 kênh dense + BM25 có dấu + BM25 bỏ dấu (giữ để so sánh)."""
    return HybridRetriever(
        [gateway_dense(), bm25(accents=True, n_max=2), bm25(accents=False, n_max=2)]
    )


def hybrid(enrich: bool = False) -> HybridRetriever:
    """R3 chốt: dense + BM25 bỏ dấu (cụm 2 âm tiết); câu không dấu thì trọng số dense 0.2.
    0.2 nằm giữa vùng ổn định 0.1-0.3 khi dò trên dev (ADR-0003).
    enrich=True (R4): cả hai kênh index văn bản có tiền tố CLB + alias + chủ đề."""
    channels = [gateway_dense(), bm25(accents=False, n_max=2)]
    if enrich:
        channels = [EnrichedRetriever(c) for c in channels]
    return HybridRetriever(channels, weights=[1.0, 1.0], noaccent_weights=[0.2, 1.0])


def hybrid_club() -> ClubBoostRetriever:
    """R4 + R5 chốt: hybrid làm giàu index + boost nhẹ đoạn của CLB được nhắc (0.1) + quota top-5.
    Dò boost {0, 0.1, 0.25, 0.5, 1}: 0-0.25 tương đương, >=0.5 kém hơn -> chọn 0.1."""
    return ClubBoostRetriever(hybrid(enrich=True), weight=0.1)


def hybrid_rerank(top_n: int = 20, only_with_diacritics: bool = True) -> CrossEncoderReranker:
    """R6 chốt: hybrid_club + cross-encoder bge-reranker-v2-m3 trên top-20, chỉ cho câu có dấu.
    Đầu vào reranker: câu hỏi đã mở rộng viết tắt × đoạn đã làm giàu (tiền tố CLB + chủ đề)."""
    return CrossEncoderReranker(
        hybrid_club(),
        top_n=top_n,
        query_fn=expand_query,
        doc_text=lambda d: enrich(d.club, d.topic, d.text),
        only_with_diacritics=only_with_diacritics,
    )


# Tên retriever -> hàm tạo. Mỗi kỹ thuật mới thêm một dòng ở đây.
RETRIEVERS: dict[str, Callable] = {
    "dense": gateway_dense,  # baseline BTC
    "bm25": lambda: bm25(accents=True),  # BM25 có dấu + R1
    "bm25_noaccent": lambda: bm25(accents=False),  # R2 kênh bỏ dấu
    "bm25_noaccent_bi": lambda: bm25(accents=False, n_max=2),  # + cụm 2 âm tiết
    "hybrid_equal": hybrid_equal,  # R3 đều 3 kênh
    "hybrid": hybrid,  # R3 chốt
    "hybrid_enrich": lambda: hybrid(enrich=True),  # + R4
    "hybrid_club": hybrid_club,  # + R4 + R5
    "hybrid_rerank_all": lambda: hybrid_rerank(only_with_diacritics=False),  # R6 rerank mọi câu
    "hybrid_rerank": hybrid_rerank,  # + R6 chỉ câu có dấu (chốt hiện tại)
}


def resolve_split(name: str) -> Path:
    """ "train" -> data/train/questions.jsonl; "dev" -> experiments/splits/dev.jsonl; hoặc đường dẫn."""
    if name == "train":
        return DATA_DIR / "train" / "questions.jsonl"
    path = Path(name)
    return path if path.suffix == ".jsonl" else SPLITS_DIR / f"{name}.jsonl"


def evaluate_retriever(retriever, rows: list[dict]) -> list[dict]:
    """Chạy retriever trên từng câu, trả về kết quả từng câu (kèm metric nếu câu được chấm)."""
    results = []
    for row in rows:
        retrieved = [doc_id for doc_id, _ in retriever.search(row["query"])[:N_RETRIEVED]]
        result = {
            "id": row["id"],
            "type": row.get("type", ""),
            "difficulty": row.get("difficulty"),
            "club": row.get("club", ""),
            "query": row["query"],
            "relevant_ids": row["relevant_ids"],
            "retrieved_ids": retrieved,
        }
        if row["relevant_ids"]:  # giống bộ chấm: câu không có relevant_ids thì không tính retrieval
            scores = score_query(retrieved, set(row["relevant_ids"]), K_VALUES)
            result.update({m: scores[m] for m in RANK_METRICS})
            result["retrieval"] = mean(scores[m] for m in RANK_METRICS)
        results.append(result)
    return results


def summarize(results: list[dict]) -> dict:
    scored = [r for r in results if "retrieval" in r]
    summary = {"n": len(scored)}
    for metric in [*RANK_METRICS, "retrieval"]:
        summary[metric] = mean(r[metric] for r in scored) if scored else 0.0
    return summary


def summarize_by(results: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[str(r[field])].append(r)
    return {key: summarize(group) for key, group in groups.items() if summarize(group)["n"]}


def _print_table(title: str, summaries: dict[str, dict]) -> None:
    width = max(len(title), *(len(k) for k in summaries))
    print(f"{title:<{width}}  {'n':>3}  {'R@5':>5}  {'MRR@5':>5}  {'NDCG@5':>6}  {'retr.':>5}")
    for key, s in sorted(summaries.items(), key=lambda kv: kv[1]["retrieval"]):
        print(
            f"{key:<{width}}  {s['n']:>3}  {s['recall@5']:>5.3f}  {s['mrr@5']:>5.3f}  "
            f"{s['ndcg@5']:>6.3f}  {s['retrieval']:>5.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Đo retrieval miễn phí (không gọi LLM).")
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), default="dense")
    parser.add_argument("--splits", nargs="+", default=["dev", "dev_noaccent"])
    parser.add_argument("--note", default="", help="ghi chú cho dòng log")
    parser.add_argument("--no-log", action="store_true", help="không ghi experiments/log.csv")
    args = parser.parse_args(argv)

    retriever = RETRIEVERS[args.retriever]()
    retriever.index(load_corpus())

    overall = {}
    for split in args.splits:
        rows = read_jsonl(resolve_split(split))  # đọc thô để giữ `type`
        results = evaluate_retriever(retriever, rows)
        overall[split] = summary = summarize(results)
        write_jsonl(Path("outputs") / "retrieval" / f"{args.retriever}_{split}.jsonl", results)

        print(f"\n=== {args.retriever} · {split} ({summary['n']} câu được chấm) ===")
        _print_table("tổng", {"all": summary})
        for field in ("type", "difficulty", "club"):
            print()
            _print_table(field, summarize_by(results, field))
        if not args.no_log:
            explog.append({"system": args.retriever, "split": split, **summary, "note": args.note})

    if {"dev", "dev_noaccent"} <= overall.keys():
        public_like = mean(overall[s]["retrieval"] for s in ("dev", "dev_noaccent"))
        print(f"\npublic-like retrieval (mean dev, dev_noaccent): {public_like:.4f}")
    print("\nChi tiết từng câu: outputs/retrieval/  ·  Log: experiments/log.csv")


if __name__ == "__main__":
    main()
