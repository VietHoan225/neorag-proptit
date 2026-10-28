---
name: explain
description: Giải thích đoạn code vừa viết hoặc một file/hàm được chỉ định trong dự án NeoRAG (package src/s2b_rag hoặc code BTC trong src/neorag), rồi hỏi lại vài câu để kiểm tra người dùng đã hiểu. Dùng khi người dùng gõ /explain, hỏi "đoạn này làm gì", hoặc muốn hiểu code AI vừa sinh ra.
---

# Explain

Mục tiêu: giúp người dùng **hiểu** code để tự giải thích được ở buổi pitching, không chỉ đọc lướt.

1. **Xác định đoạn cần giải thích:** tham số người dùng đưa (file, hàm, `file:dòng`), nếu không có thì lấy
   thay đổi gần nhất (`git diff` + file chưa track). Đọc code thật trước khi giải thích, không giải thích theo trí nhớ.
2. **Giải thích theo thứ tự:**
   - **Mục đích:** đoạn này nằm ở đâu trong pipeline RAG (offline index / xử lý query / retrieve / rerank /
     dựng context / generate / hậu xử lý / đánh giá) và giúp metric nào (Recall@5, MRR@5, NDCG@5, String
     Presence, ROUGE-L, Groundedness, Noise, Relevancy, Abstention) — một đến hai câu.
   - **Ý tưởng chính:** khái niệm đằng sau (vd Unicode NFD, BM25, cosine similarity, RRF, cross-encoder,
     phân tầng khi chia tập), bằng lời đơn giản, có **ví dụ nhỏ lấy từ dữ liệu thật** của dự án
     (một câu hỏi train, một đoạn corpus) nếu hợp.
   - **Đi qua code:** chỉ những dòng quan trọng, trích `file:dòng`, không đọc từng dòng.
   - **Chỗ dễ sai:** trường hợp biên và giả định ngầm (chuỗi rỗng, chia cho 0, `đ`/Unicode, câu không có
     đáp án có `relevant_ids` rỗng, id trùng, thứ tự xếp hạng, cache làm kết quả "không đổi").
   - **Liên hệ luật thi** nếu có: chi phí LLM, không sửa `src/neorag/`, không overfit public.
3. **Hỏi 2–3 câu kiểm tra**, ví dụ: "nếu bỏ dòng thay `đ` thì `strip_accents('Đội')` ra gì?",
   "vì sao không dùng `round()` khi tính số câu holdout?", "RRF có cần chuẩn hoá điểm BM25 và cosine không?".
   Không đưa đáp án ngay, chờ người dùng trả lời rồi nhận xét từng câu.
4. Nếu người dùng trả lời sai, giải thích lại phần đó **theo cách khác** (ví dụ khác, hình dung khác),
   không lặp lại y nguyên. Nếu chỗ hiểu sai đáng nhớ, gợi ý ghi lại bằng `/journal`.

Giữ ngắn. Tiếng Việt, thuật ngữ kỹ thuật giữ nguyên tiếng Anh. Không sửa code trong skill này.
