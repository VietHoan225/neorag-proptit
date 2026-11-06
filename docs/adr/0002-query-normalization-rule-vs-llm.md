# ADR-0002: Chuẩn hoá bằng luật, dùng haiku để khôi phục dấu khi cần

- **Ngày:** 2026-11-06 (ghi lại quyết định R1 ngày 28/10 và R7 ngày 29/10)
- **Trạng thái:** Chấp nhận

## Bối cảnh

Trong train có câu viết tắt và không dấu; public có 48% câu không dấu. Dense baseline giảm từ 0.708 trên `dev`
xuống 0.498 trên `dev_noaccent`. Reranker local cũng làm hại câu không dấu (ADR-0004). Cần chuẩn hoá câu hỏi
nhưng ngân sách haiku cả cuộc thi chỉ 5 USD.

## Các phương án

| Phương án | Số đo / đánh đổi |
|---|---|
| Mở rộng viết tắt bằng luật cho cả dense và BM25 | Dense trên `dev` 0.708 → 0.701, `dev_noaccent` 0.498 → 0.477: bỏ cho dense |
| Bỏ dấu và mở rộng viết tắt cho BM25 | Kết hợp BM25 bỏ dấu với cụm 2 âm tiết và dense có trọng số theo dấu: `public-like` 0.603 → 0.804 (ADR-0003), 0 USD haiku |
| Chỉ dùng hybrid + reranker, không viết lại | Sau R6: `dev_noaccent` 0.829; câu không dấu vẫn thua câu có dấu 0.905 |
| Haiku viết lại mọi câu | Tốn tiền và có thể sửa nghĩa câu đã rõ; bản `hybrid_rewrite` thử trên `dev` chỉ 0.899 so với 0.930 của bản chỉ viết lại câu không dấu |
| **Chỉ viết lại câu không dấu, tìm cả câu viết lại và câu gốc** | `dev_noaccent` 0.829 → 0.898; public #1 75.53 → #2 80.15 |

## Quyết định

BM25 luôn chuẩn hoá bằng luật: bỏ dấu ở câu hỏi và index, thêm cụm 2 âm tiết, mở rộng những viết tắt thông dụng.
Giữ câu gốc cho dense. Chỉ khi câu **không có chữ mang dấu**, gọi `claude-haiku-5-5` viết lại có dấu rồi tìm bằng
cả câu viết lại và câu gốc; gộp thứ hạng bằng RRF, trọng số câu gốc 0.5. Lỗi viết lại thì dùng câu gốc.

R7 trên `dev`: retrieval 0.905 (R6) → 0.930; `dev_noaccent`: 0.829 → 0.898. Kiểm tra trên `holdout_noaccent`
đạt 0.887. 109 lần viết lại riêng trong thí nghiệm tốn khoảng 0.03 USD theo trang usage. Số đo và cấu hình có trong
`experiments/log.csv`; code: `textnorm.py`, `query.py`, `retrieval/rewrite.py`.

## Hệ quả

Ít cuộc gọi hơn việc viết lại mọi câu, nhưng câu không dấu chậm thêm một lần gọi và kết quả phụ thuộc haiku.
Câu có vài chữ mang dấu được xem là có dấu, nên lỗi gõ hỗn hợp vẫn có thể xảy ra.
