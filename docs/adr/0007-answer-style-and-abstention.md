# ADR-0007: Trả lời gọn theo phạm vi câu hỏi, chỉ bỏ rào đón thừa

- **Ngày:** 2026-11-06 (ghi lại G2 và A2–A4)
- **Trạng thái:** Chấp nhận

## Bối cảnh

Đáp án mẫu trung bình khoảng 210 ký tự. Đầu ra ban đầu dài 463 ký tự; thông tin thêm và rào đón làm giảm
ROUGE-L và Noise Sensitivity. Nhưng từ chối khi tài liệu thiếu thông tin là cần thiết để giữ Abstention.

## Các phương án

| Phương án | Kết quả |
|---|---|
| Prompt viết gọn + bỏ markdown (G1/G2) | `dev`: SP 0.717 → 0.869, ROUGE-L 0.419 → 0.618, độ dài 463 → 251 ký tự |
| Bỏ câu rào đón **thuần ở cuối** bằng code (A2) | `dev_casual`: ROUGE-L 0.6283 → 0.6375, SP giữ 0.9359, 8/8 câu không trả lời được vẫn từ chối |
| Cắt mạnh cả rào đón đầu/giữa câu (A3) | Public #8 87.88 so với #7 88.17; 1−Noise 92.74 → 90.17; cắt cả phần trả lời trực tiếp |
| Prompt dặn kể đủ mọi mục (A4) | `dev_casual`: SP 93.59 → 92.74, câu dài 257 → 279 ký tự; lỗi vai trò mục tiêu chưa sửa |

## Quyết định

Dùng prompt A của A2 để ghi từng ý và chỉ trả lời theo đoạn nguồn. `postprocess` bỏ markdown; `drop_tail_hedges`
chỉ bỏ câu cuối thuần rào đón không đáp ứng ý hỏi. Giữ rào đón nếu nó **trả lời trực tiếp** câu hỏi về phần
tài liệu thiếu. Khi không có căn cứ, trả lời theo dạng từ chối của prompt; nếu đầu ra rỗng, thử lại một lần rồi
dùng câu dự phòng. Không dùng cách cắt mạnh A3 hay prompt kể đủ A4 cho câu trả lời nộp bài.

Public #7 so với #6: cùng retrieval và SP, ROUGE-L +1.51 điểm phần trăm, 1−Noise 87.18 → 92.74; chỉ 6/42
câu đổi. Bài #9 giữ nguyên câu trả lời #7 ở cả 42 câu, Abstention 100. Cùng file #7 chấm lại 88.17 → 87.99,
nên khác biệt nhỏ của judge không đủ để kết luận; phần ROUGE-L là metric tất định.

## Hệ quả

Luật hậu xử lý có phạm vi hẹp; vẫn có câu dài hơn đáp án mẫu. Cắt thêm có rủi ro bỏ phần trả lời đúng, như A3.
Code: `generate.py` (`postprocess`, `drop_tail_hedges`), `pipeline.py`; số liệu: `experiments/log.csv`.
