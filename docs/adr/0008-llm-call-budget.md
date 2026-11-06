# ADR-0008: Hai lần haiku đọc nguồn cho bài nộp A5

- **Ngày:** 2026-11-06 (ghi lại quyết định A5 ngày 04/11)
- **Trạng thái:** Chấp nhận

## Bối cảnh

Ngân sách gateway là 5 USD cho cả cuộc thi. Sau A2, prompt A4 chọn nguồn tốt hơn nhưng làm câu trả lời dài và
SP trên `dev_casual` giảm 93.59 → 92.74. Cần cải thiện retrieval mà vẫn giữ câu trả lời đã tốt của A2.

## Các phương án

| Phương án | Kết quả / chi phí |
|---|---|
| Một lần đọc prompt A (A2) | Public retrieval 91.73; một lần gọi/câu, thêm một lần viết lại nếu không dấu |
| Một lần đọc prompt A4 | `dev_casual` retrieval 98.40 → 98.65, nhưng SP giảm và câu dài 257 → 279 ký tự; bỏ |
| **A5: lần 1 viết câu trả lời, lần 2 chỉ lấy nguồn** | `dev_casual` retrieval 98.40 → 98.76; public 91.73 → 93.93; câu trả lời public giữ nguyên 42/42 |
| A6: thêm lần đọc thứ 3 | Public retrieval vẫn 93.93 dù 3/42 top-5 đổi; bỏ |

## Quyết định

Bài nộp dùng hai lần gọi `claude-haiku-5-5` với cùng 20 đoạn: prompt A cho câu trả lời và nguồn, prompt A4
chỉ lấy nguồn. Gộp nguồn theo ADR-0006, giữ nguyên câu trả lời của lần 1. Câu không dấu có thể cần thêm một lần
gọi để viết lại; câu trả lời rỗng có thể cần một lần thử lại. Không truyền `temperature`.

Diễn tập 40 câu `holdout_casual` với tất cả lần gọi thật: 0.08 USD (~0.002 USD/câu), trung vị 7.8 giây,
p90 10.5 giây, tổng 325 giây; 40/40 đủ 10 id, 0 câu rỗng. Vì đây là chi phí đo trên 40 câu và thời gian phụ
thuộc gateway, dự toán cho lượt 42 câu là khoảng 0.08–0.09 USD, không phải cam kết cố định. Chatbot chỉ dùng
lần đọc 1 vì không nộp danh sách id cho bộ chấm.

## Hệ quả

Đổi lấy +2.20 điểm phần trăm retrieval public tất định bằng một lần gọi thêm/câu. Lần 2 còn viết câu trả lời
nhưng bị bỏ, nên tốn token; thiết kế prompt chỉ chọn nguồn chưa có số đo để thay thế. Trước lần chạy mới phải báo
ước tính số lần gọi và chi phí, chờ người dùng đồng ý. Số liệu: `experiments/log.csv`, `docs/journal.md`.
