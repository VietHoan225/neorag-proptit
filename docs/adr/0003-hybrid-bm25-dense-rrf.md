# ADR-0003: Hybrid retrieval — dense bge-m3 + BM25 bỏ dấu, gộp bằng RRF có trọng số theo dấu

- **Ngày:** 2026-10-29
- **Trạng thái:** Chấp nhận

## Bối cảnh
Retrieval chiếm 50% điểm Benchmark (mean Recall@5, MRR@5, NDCG@5). Baseline của BTC chỉ dùng dense
bge-m3 và tụt mạnh khi câu hỏi viết không dấu — kiểu câu rất phổ biến ở Public Test
("e hoc nganh da phuong tien (dpt)..."), nhưng chỉ có 26/126 câu ở train.

Đo trên tập dev (86 câu tách từ train) và bản bỏ dấu tự động của nó (`dev_noaccent`):

| | dev | dev_noaccent |
|---|---|---|
| Dense baseline | 0.708 | **0.498** |
| Riêng loại "không dấu/viết tắt" | 0.351 | — |

Phân tích từng câu: khi bỏ dấu, dense vẫn tìm đúng CLB (nhận ra tên riêng như ProPTIT, Multimedia) nhưng
chọn sai chủ đề vì mất nghĩa của từ thường ("thanh lap", "dia chi"). ProPTIT có 78/118 đoạn nên bị phạt
nặng nhất (0.633 → 0.337).

Ràng buộc: ngân sách LLM 5 USD cho cả cuộc thi → ưu tiên giải pháp không gọi `claude-haiku-5-5`.

## Các phương án
Thước đo: **public-like** = trung bình retrieval của dev và dev_noaccent (đo bằng
`python -m s2b_rag.eval.retrieval`, chấm bằng `score_query` chính thức, 0 USD).

| # | Phương án | dev | dev_noaccent | public-like |
|---|---|---|---|---|
| 0 | Dense baseline | 0.708 | 0.498 | 0.603 |
| 1 | Dense + mở rộng viết tắt cho câu hỏi | 0.701 | 0.477 | 0.589 |
| 2 | BM25 có dấu (+ mở rộng viết tắt) | 0.647 | 0.489 | 0.568 |
| 3 | BM25 bỏ dấu, âm tiết đơn | 0.707 | 0.707 | 0.707 |
| 4 | BM25 bỏ dấu + cụm 2 âm tiết | 0.738 | 0.738 | 0.738 |
| 5 | RRF đều 3 kênh (dense + BM25 có dấu + BM25 bỏ dấu) | 0.766 | 0.661 | 0.714 |
| 6 | RRF dense + BM25 bỏ dấu (#4), trọng số đều | 0.790 | 0.672 | 0.731 |
| 7 | **#6, dense trọng số 0.2 khi câu không có dấu** | **0.836** | **0.772** | **0.804** |
| — | Khôi phục dấu bằng LLM (haiku) | chưa thử | | |

- **Dense đơn (#0, #1):** hiểu nghĩa câu có dấu, nhưng yếu với câu không dấu; mở rộng viết tắt còn làm tệ hơn.
- **BM25 bỏ dấu (#3, #4):** bỏ dấu cả đoạn văn lẫn câu hỏi nên "thanh lap" khớp "thành lập"; điểm không
  phụ thuộc người dùng gõ dấu hay không. Cụm 2 âm tiết (`thanh_lap`) giữ được từ ghép tiếng Việt. Nhược:
  chỉ khớp từ bề mặt, không hiểu từ đồng nghĩa ("diễn đạt khác").
- **RRF đều nhiều kênh (#5, #6):** thua cả BM25 bỏ dấu đơn lẻ trên dev_noaccent vì các kênh yếu với câu
  không dấu (dense, BM25 có dấu) "bỏ phiếu" át kênh mạnh.
- **Khôi phục dấu bằng LLM:** tốn thêm một lần gọi haiku cho ~một nửa câu Public, thêm 1–1.5 s độ trễ,
  không tất định. Để dành (bước R7) nếu sau reranker vẫn còn khoảng cách.

## Quyết định
Chọn **#7**:

```
câu hỏi ─┬─► dense bge-m3 (câu hỏi gốc) ─────────────────────────────────┐
         └─► BM25 trên văn bản bỏ dấu, âm tiết + cụm 2 âm tiết,            ├─► RRF: Σ w / (60 + hạng)
             câu hỏi đã mở rộng viết tắt (ko→không, tv→thành viên, ...) ─┘
         w = [1.0, 1.0] nếu câu có dấu; [0.2, 1.0] nếu câu không có chữ nào mang dấu
```

Vì sao:
- **RRF thay vì cộng điểm:** chỉ dùng thứ hạng nên không phải đưa điểm BM25 (0..~30) và cosine (0..1) về
  cùng thang; k = 60 là giá trị chuẩn trong tài liệu gốc của RRF, không dò.
- **Trọng số theo dấu:** có lý do đo được (dense tụt 0.708 → 0.498 với câu không dấu), không phải mẹo theo
  từng câu. Dò w ∈ {0, 0.1, ..., 1.0}: vùng 0.1–0.3 cho public-like 0.798–0.804 (ổn định); w = 0 kém hơn
  (0.787) vì dense vẫn giúp nhận đúng CLB. Chọn **0.2 ở giữa vùng**, không lấy đỉnh, để giảm overfit dev.
- **Kiểm tra trên holdout** (40 câu, không dùng khi chọn):

  | | holdout | holdout_noaccent |
  |---|---|---|
  | Dense baseline | 0.752 | 0.445 |
  | **Hybrid #7** | **0.817** | **0.777** |

  Loại "không dấu/viết tắt": dev 0.351 → 0.736, holdout 0.463 → 0.750.
- Chi phí: **0 USD** haiku; BM25 tự viết (~30 dòng), không thêm thư viện.

## Hệ quả
- Dễ hơn: câu không dấu/viết tắt không còn là điểm yếu lớn nhất; retrieval chạy nhanh và miễn phí, đo lại
  mỗi lần đổi chỉ mất vài giây.
- Khó hơn / rủi ro:
  - Thêm 2 tham số (trọng số dense khi không dấu, n-gram) được chọn trên dev → phải tiếp tục kiểm tra
    trên holdout ở mỗi mốc.
  - Câu viết lẫn (chỉ vài chữ có dấu) bị coi là câu có dấu → dense giữ trọng số đầy đủ.
  - BM25 không xử lý từ đồng nghĩa; câu "diễn đạt khác" dựa vào dense và reranker (R6) sắp tới.
- Code: `src/s2b_rag/retrieval/{bm25,fusion,dense}.py`, `src/s2b_rag/textnorm.py`;
  số liệu: `experiments/log.csv`.
