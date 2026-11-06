# ADR-0010: Tái lập bài nộp bằng commit, cấu hình và cache

- **Ngày:** 2026-11-06
- **Trạng thái:** Chấp nhận

## Bối cảnh

Đầu ra haiku và judge không tất định. Cùng file nộp #7 chấm hai lần được 88.17 và 87.99, dù retrieval,
String Presence và ROUGE-L giống hệt. Muốn kiểm tra một bản code có làm đổi bài nộp không, cần giữ đầu ra
haiku đã dùng và tách sai số judge khỏi thay đổi pipeline.

## Các phương án

| Phương án | Đánh đổi |
|---|---|
| Chạy lại gateway trên checkout mới | Đơn giản nhưng có thể sinh câu khác và tốn tiền |
| Chỉ lưu điểm tổng | Không chứng minh được câu trả lời hay id nào đã đổi |
| **Ghi commit/cấu hình, giữ cache SQLite và file dự đoán/zip** | So được từng câu/id; cache và outputs chỉ lưu cục bộ, không commit |

## Quyết định

Mỗi mốc đo ghi commit, split, cấu hình và metric vào `experiments/log.csv`. Cấu hình nộp bài nằm ở class mặc
định `s2b_rag.pipeline:S2BPipeline` (= A5). `LLMClient` lưu phản hồi theo khóa gồm endpoint/model/tham số/messages
trong `.cache/pipeline.sqlite`; giữ cache để chạy lại cùng prompt và câu hỏi. Lưu outputs của bài nộp tại máy
và so **câu trả lời + thứ tự 10 id** trước khi chấm. Không commit `.cache/`, `.env`, `outputs/`.

Đã kiểm tra `S2BPipeline` trên public từ cache: cả 42/42 câu trả lời và danh sách id khớp bài #9, không gọi
gateway. Diễn tập 40 câu `holdout_casual` không có cache đo 0.08 USD và chạy trọn 325 giây (ADR-0008).

## Hệ quả

Cần sao lưu cache và file dự đoán trước ngày private. Clone mới thiếu cache không bảo đảm câu trả lời giống
byte-for-byte; judge có thể chấm khác ngay cả khi file nộp y hệt. Chatbot streaming gọi gateway trực tiếp
nên không được bảo đảm tái lập từ cache này. Code cache ở `src/neorag/llm.py` (mã BTC, không sửa); log ở
`experiments/log.csv`.
