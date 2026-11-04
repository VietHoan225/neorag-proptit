# ADR-0005: Haiku trích nguồn trong lần sinh câu trả lời để xếp lại top-5 (R9, E1)

- **Ngày:** 2026-11-03 (thí nghiệm A bổ sung 2026-11-04)
- **Trạng thái:** Chấp nhận (R9 + E1 là bài nộp #5; A đang chờ xác nhận public)

## Bối cảnh
- Sau R10 (lần nộp #3: 79.80), retrieval public là 81.8, trong khi dev là 93. Trên dev_casual (ADR-0001), câu cần
  3+ đoạn gold chỉ đạt 0.706, và public có 74% câu ≥ 2 ý (dev chỉ 42%).
- Ở các câu ≥ 2 ý bị thiếu đoạn, **6/8 đoạn thiếu đã nằm ở hạng 6–10**, tức lỗi xếp hạng chứ không phải không tìm
  thấy. Nguyên nhân: cross-encoder (ADR-0004) chấm từng đoạn riêng, nên các đoạn cùng nói về ý chính chiếm hết top-5.
- Recall/MRR/NDCG chỉ tính @5. Groundedness lại đọc text của cả 10 id nộp. Được nộp tối đa 10 id.
- Ngân sách 5 USD và độ trễ của chatbot → không muốn thêm lần gọi LLM.

## Các phương án
| # | Phương án | Lần gọi thêm | Kết quả / lý do |
|---|---|---|---|
| 1 | Tách câu con theo luật rồi tìm riêng (R8) | 0 | **Bỏ**: public-like 0.867 → 0.853, câu con mất nghĩa vì "đó", "vậy" |
| 2 | Gọi haiku riêng để rerank | +1/câu | Gấp đôi chi phí và độ trễ |
| 3 | Gộp RRF reranker với hybrid | 0 | **Bỏ**: 0.9137 → ≤ 0.9136 |
| 4 | **Trong lần sinh, haiku ghi thêm dòng "NGUỒN: id, id"; đoạn được trích lên top** | 0 | Chọn (R9) |
| 5 | Như #4, haiku đọc 20 đoạn thay vì 10 (E1) | 0 | Chọn |
| 6 | E1 nhưng giữ đoạn top-1 của retriever cố định | 0 | Bỏ: mô phỏng 0.938, thấp hơn |

## Quyết định
Chọn **#4 + #5**:

```
retrieval (ADR-0003/0004) → 20 đoạn đầu vào prompt → haiku: câu trả lời + "NGUỒN: 12, 40"
   → cắt dòng NGUỒN, chỉ giữ id có trong prompt
   → đưa đoạn được trích lên đầu (giữa chúng giữ thứ tự retriever) → MỚI cắt còn 10 id
```

- **Vì sao hiệu quả:** haiku đọc cả câu hỏi và mọi đoạn cùng lúc, nên biết mỗi ý cần một đoạn, điều mà cross-encoder
  không biết. Không thêm lần gọi nào.
- **Giữ thứ tự retriever giữa các đoạn được trích:** LLM chỉ quyết định đoạn **nào** vào top, không quyết định đoạn
  nào đứng đầu → MRR ít bị xáo trộn (dev_casual MRR 0.994).
- **Cắt 10 id SAU khi đưa đoạn được trích lên (E1):** đoạn hạng 11–20 được trích vẫn vào được top-5. Khi chỉ đọc 10
  đoạn, kết quả y hệt cắt trước (test kiểm tra trên 200 trường hợp ngẫu nhiên).
- **An toàn:** id bịa hoặc nằm ngoài prompt bị bỏ; thiếu dòng NGUỒN thì giữ thứ tự retriever; câu trả lời rỗng thì gọi
  lại 1 lần kèm câu nhắc, vẫn rỗng thì dùng câu dự phòng.
- **Hợp lệ theo luật:** chỉ dùng `claude-haiku-5-5`, không dùng nhãn gold, không hard-code id.

## Kết quả
| Bước | dev_casual retrieval | SP | ROUGE-L | Public (gốc) |
|---|---|---|---|---|
| R10 (trước R9) | 0.892 | — | — | #3: 79.80 (retrieval 81.8) |
| R9 bước 1: đọc 10 đoạn thay vì 8 | — | dev 0.9145 → 0.9338 | dev 0.6365 → 0.6658 | — |
| R9 bước 2: xếp lại theo nguồn | **0.9665** | 0.921 | 0.632 | **#4: 85.09** (retrieval 89.6) |
| E1: đọc 20 đoạn (+ E2 prompt) | 0.9815 | 0.908 | 0.637 | **#5: 87.02** (R@5 90.1, NDCG 89.6) |
| A: chọn nguồn theo từng ý (04/11) | 0.9840 | 0.936 | 0.628 | chưa nộp |

- R9 không làm tụt Recall@5 ở câu nào trên dev; 0 câu bị lọt dòng "NGUỒN" vào câu trả lời; 1−Noise public 79.1 → 83.9.
- E1, câu 3+ đoạn gold (9 câu): 0.804 → 0.856.
- Mức tăng R9 trên dev_casual (+7.5) khớp với public (+7.7) → xem ADR-0001.

## Hệ quả
- Dễ hơn: câu nhiều ý không còn là điểm yếu lớn nhất của retrieval; chi phí mỗi câu vẫn là 1 lần gọi.
- Khó hơn / rủi ro:
  - **Groundedness:** haiku đọc 20 đoạn nhưng chỉ nộp 10. Nếu nó dùng một đoạn mà không trích, đoạn đó bị cắt → judge
    thấy thông tin không có nguồn. Groundedness public 100 (#4) → 97.86 (#5); kiểm tra từ cache thấy đúng 1 câu
    (public-0016), tương đương khoảng 0.07 điểm tổng. A thêm `keep_used_sources`: đổi đoạn đã dùng mà quên trích vào
    hạng 6–10 (không đụng top-5).
  - Retrieval phụ thuộc vào việc LLM tuân theo định dạng; đầu ra không tất định (không truyền temperature).
  - Chatbot phải ẩn dòng NGUỒN khi stream câu trả lời.
  - Không tách được tác dụng của E1 với E2 và với nhiễu sinh câu trả lời trong mức +1.93 của lần nộp #5.
- Code: `src/s2b_rag/generate.py` (`split_sources`, `promote_cited`, `promote_tiers`, `split_aspects`,
  `keep_used_sources`), `src/s2b_rag/pipeline.py` (`S2BPipeline`, `S2BPipelineCtx20`, `S2BPipelineE2`,
  `S2BPipelineA`); số liệu: `experiments/log.csv`; chi tiết: `docs/journal.md`.
