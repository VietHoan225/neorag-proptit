# ADR-0004: Reranker cross-encoder chạy local, chỉ áp dụng cho câu hỏi có dấu

- **Ngày:** 2026-10-29
- **Trạng thái:** Chấp nhận

## Bối cảnh
Sau hybrid retrieval (ADR-0003) và làm giàu index + boost CLB (`hybrid_club`), recall đã cao nhưng thứ tự
trong top-5 — quyết định MRR@5 và NDCG@5 — vẫn còn cải thiện được:

| `hybrid_club` | dev | dev_noaccent | public-like | holdout / holdout_noaccent |
|---|---|---|---|---|
| Retrieval (mean R@5, MRR@5, NDCG@5) | 0.867 | 0.829 | 0.848 | 0.860 / 0.799 |

Ràng buộc: rerank bằng LLM phải dùng `claude-haiku-5-5` (tốn ngân sách 5 USD); luật cho phép cross-encoder
open-source chạy trên máy. Máy có RTX 2060 6 GB; `BAAI/bge-reranker-v2-m3` (đa ngôn ngữ) đã có trong cache.

## Các phương án
Thước đo: **public-like** = trung bình retrieval của dev và dev_noaccent (`python -m s2b_rag.eval.retrieval`).
Tất cả rerank trên top-N ứng viên của `hybrid_club`.

| # | Phương án | dev | dev_noaccent | public-like |
|---|---|---|---|---|
| 0 | Không rerank (`hybrid_club`) | 0.867 | 0.829 | 0.848 |
| 1 | Rerank mọi câu, câu hỏi/đoạn gốc, top-20, thay thứ hạng | 0.873 | 0.751 | 0.812 |
| 2 | Rerank mọi câu, gộp RRF với hybrid, top-20 | 0.888 | 0.823 | 0.855 |
| 3 | Rerank mọi câu, câu mở rộng × đoạn làm giàu, top-20 | 0.891 | 0.766 | 0.829 |
| 4 | Rerank mọi câu, bỏ dấu cả câu hỏi lẫn đoạn | 0.780 | 0.780 | 0.780 |
| 5 | Câu có dấu: như #3; câu không dấu: như #4 | 0.906 | 0.780 | 0.843 |
| 6 | **Chỉ rerank câu có dấu (như #3), câu không dấu giữ thứ hạng hybrid; top-20** | **0.905** | **0.829** | **0.867** |
| 7 | Như #6, top-40 | 0.902 | 0.829 | 0.866 |
| — | Rerank bằng LLM (haiku) | chưa thử: tốn ngân sách, chậm | | |

Phát hiện quyết định: **bge-reranker-v2-m3 gần như không hiểu câu không dấu.** Cùng một cặp:

| Câu hỏi | Điểm đoạn đúng (Multimedia) | Điểm đoạn sai (ProPTIT) |
|---|---|---|
| "CLB Multimedia có phòng riêng để chụp ảnh không?" | 0.9935 | 0.0001 |
| "clb multimedia co phong rieng de chup anh khong" | 0.0167 | 0.0030 |
| "clb mm co phong rieng de chup anh khong" | 0.0006 | **0.0019** |

fp16 và fp32 cho cùng điểm → là giới hạn của model, không phải lỗi độ chính xác. Mọi cách rerank câu không dấu
(#1–#5) đều làm dev_noaccent tụt dưới mốc 0.829.

## Quyết định
Chọn **#6**: `CrossEncoderReranker` (`src/s2b_rag/retrieval/rerank.py`) bọc `hybrid_club`:
- Câu hỏi **có ít nhất một chữ mang dấu** → chấm lại top-20 bằng `bge-reranker-v2-m3` (local, fp16 trên GPU),
  đầu vào là câu hỏi đã mở rộng viết tắt × đoạn có tiền tố CLB + chủ đề; thay hẳn thứ hạng top-20.
- Câu **không dấu** → giữ nguyên thứ hạng của `hybrid_club`.

Vì sao:
- Cross-encoder đọc cặp (câu hỏi, đoạn) cùng lúc nên xếp hạng chính xác hơn bi-encoder: dev 0.867 → 0.905
  (MRR@5 0.851 → 0.916).
- Bỏ qua câu không dấu vì có số đo cho thấy rerank làm hại chúng (0.829 → 0.751–0.823).
- top-20 ≈ top-40 về điểm (0.867 / 0.866) nhưng nhanh gấp đôi.
- **Kiểm tra trên holdout** (không dùng khi chọn):

  | | holdout | holdout_noaccent |
  |---|---|---|
  | `hybrid_club` | 0.860 | 0.799 |
  | Rerank mọi câu (#3) | 0.879 | 0.749 |
  | **#6** | **0.887** | **0.799** |

- Chi phí: 0 USD. Độ trễ retrieval p50 3 ms → 183 ms (p95 208 ms) trên RTX 2060; 2.6 GB VRAM.

## Hệ quả
- Dễ hơn: câu có dấu đạt ~0.90 retrieval mà không tốn ngân sách LLM.
- Khó hơn / rủi ro:
  - Phụ thuộc torch + sentence-transformers (`python scripts/install.py --local`), nạp model 27 s mỗi lần
    khởi động; chạy CPU sẽ chậm hơn nhiều → chatbot phải nạp model lúc khởi động (`prepare()`).
  - Câu không dấu chưa được hưởng reranker: khoảng cách 0.905 (có dấu) vs 0.829 (không dấu) là **trần lợi ích
    ~+0.04 public-like** cho việc khôi phục dấu bằng LLM (bước R7) — có số đo để quyết định có đáng tốn
    ngân sách hay không.
  - Câu viết lẫn (chỉ vài chữ có dấu) được coi là có dấu → vẫn bị rerank.
- Code: `src/s2b_rag/retrieval/rerank.py`, cấu hình `hybrid_rerank()` trong `src/s2b_rag/eval/retrieval.py`;
  số liệu: `experiments/log.csv`.
