# ADR-0009: Chatbot Starlette stream câu trả lời từ pipeline A5

- **Ngày:** 2026-11-06 (ghi lại triển khai 05/11 và kiểm tra thủ công 06/11)
- **Trạng thái:** Chấp nhận

## Bối cảnh

Demo cần chữ hiện dần, nguồn dễ hiểu và câu nối tiếp. Pipeline nộp bài A5 viết phần `Ý HỎI`/`NGUỒN` trước
`TRẢ LỜI`, nên stream trực tiếp sẽ lộ định dạng nội bộ. `run()` đã tái lập bài public #9; thay nó để phục vụ UI
có nguy cơ đổi bài nộp.

## Các phương án

| Phương án | Đánh đổi |
|---|---|
| Streamlit | Dựng giao diện nhanh; điều khiển luồng NDJSON, nguồn và bố cục kém linh hoạt hơn |
| React + backend | Giao diện linh hoạt nhưng thêm bước build và dependency trước ngày thi |
| **Starlette + HTML/CSS/JS thuần, `POST /api/ask` stream NDJSON** | Ít dependency, kiểm soát từng sự kiện; phải tự viết UI và bộ lọc stream |

## Quyết định

Giữ nguyên `run()` của A5. Thêm `retrieve()`, `stream_answer()`, `finish_answer()` cho chatbot: cùng retrieval,
20 đoạn và prompt trả lời của lần đọc 1; chỉ hiện chữ sau `TRẢ LỜI:` và chặn dòng `NGUỒN:`. Event `retrieval`
hiện nguồn trước, `token` hiện chữ dần, `done` thay bằng câu đã hậu xử lý và đánh dấu nguồn được trích. Dùng
`POST` vì cần gửi câu hỏi và tối đa hai lượt lịch sử; trình duyệt giữ lịch sử, server không lưu hội thoại.

Câu xã giao và phép tính được xử lý ngay bằng luật. Câu nối tiếp được haiku viết thành câu tự đủ nghĩa trước
khi truy xuất. Streaming gọi gateway trực tiếp, không dùng cache câu trả lời của `LLMClient` (ADR-0010).

Kiểm tra tự động 125 test pass sau bước ghép ban đầu; `run()` tái lập public #9 đúng 42/42 từ cache. Bản web
cũ G3 đo qua HTTP trên 10 câu: chữ đầu trung vị 2.46 giây có dấu, 4.67 giây không dấu; số này **không đại diện
cho A5** vì A5 phải sinh xong phần `Ý HỎI` trước chữ đầu. Ngày 06/11, người dùng đã thử thủ công bộ câu chào,
phép tính, có dấu, không dấu, nối tiếp và ngoài phạm vi và báo đều ổn; chưa ghi số đo độ trễ A5.

## Hệ quả

Demo và bài nộp dùng cùng class nhưng khác số lần gọi: chatbot chỉ cần câu trả lời, bài nộp cần thêm nguồn
lần 2. Chatbot có thể hiện chữ đầu chậm hơn web cũ; mỗi câu hỏi thật vẫn tốn gateway dù nội dung giống câu
đã hỏi. Code: `pipeline.py`, `generate.py` (`AnswerStreamFilter`), `web/`.
