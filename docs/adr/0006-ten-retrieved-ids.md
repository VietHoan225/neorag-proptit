# ADR-0006: Đưa nguồn được dùng lên đầu rồi nộp 10 id

- **Ngày:** 2026-11-06 (ghi lại các thí nghiệm R9–A5 ngày 03–04/11)
- **Trạng thái:** Chấp nhận

## Bối cảnh

Retrieval chỉ chấm top-5, còn Groundedness đọc văn bản của **tất cả** id nộp (tối đa 10). Haiku đọc 20 đoạn
để xử lý câu nhiều ý, nên cần chọn 10 id vừa ưu tiên đoạn đúng trong top-5 vừa giữ bằng chứng cho câu trả lời.
Ở public #3, retrieval 81.84; 6/8 đoạn gold bị thiếu ở các câu nhiều ý vẫn ở hạng 6–10.

## Các phương án

| Phương án | Kết quả / rủi ro |
|---|---|
| Lấy 10 đoạn đầu từ retriever | Bỏ đoạn hạng 11–20 dù haiku đã dùng; top-5 thiếu ý phụ |
| Chỉ nộp 5 id | Không giữ được bằng chứng ngoài top-5 cho Groundedness |
| Để haiku xếp thứ tự mọi id | Trên `dev_casual`, thứ tự haiku ghi cho top-5 y hệt cách giữ thứ tự retriever trong từng tầng; không có lợi ích đo được |
| **Nguồn haiku ghi theo từng ý lên trước, giữ thứ tự retriever trong tầng, lấp đủ 10** | R9 public retrieval 81.84 → 89.56; A5 91.73 → 93.93 |

## Quyết định

Lấy 20 đoạn làm context. Chỉ chấp nhận id haiku trích nếu có trong context. Gộp nguồn của lần đọc 1 và lần
đọc 2 (A5); xếp tầng **nguồn chính của từng ý → nguồn phụ → các đoạn retriever chưa được trích**, giữ thứ tự
retriever trong mỗi tầng. Sau khi xếp mới cắt 10 id. Nếu câu trả lời dùng đoạn mà haiku quên trích, thay đoạn đó
vào vị trí 6–10; không làm xáo top-5. Nếu không trích nguồn hợp lệ, giữ thứ tự retriever.

Trên `dev_casual`, A5 retrieval 98.76; public #9 retrieval 93.93 (R@5 92.58, MRR@5 96.43, NDCG@5 92.77).
Lần đọc thứ 3 A6 đổi top-5 ở 3/42 câu public nhưng retrieval vẫn 93.93, nên dừng ở hai lần đọc. Mô phỏng
giữ cố định top-1 của retriever hoặc chỉ dùng một tầng nguồn đều thấp hơn A5 (xem `docs/journal.md`, 04/11).

## Hệ quả

Haiku phải ghi nguồn đúng định dạng; code kiểm tra id và có đường lùi khi thiếu nguồn. 20 đoạn vào prompt nhưng
chỉ nộp 10 tạo rủi ro thiếu bằng chứng: public #5 Groundedness 97.86, sau luật giữ đoạn đã dùng và A5 đạt 100
trên bài #9; metric judge có nhiễu. Chi tiết ý tưởng trích nguồn và so sánh prompt ở ADR-0005; code nằm trong
`generate.py` và `pipeline.py`.
