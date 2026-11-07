# CLAUDE.md — NeoRAG Cup 2026 (bài dự thi cá nhân)

Hệ thống RAG hỏi đáp về cộng đồng S2B (CLB Lập Trình PTIT, Multimedia PTIT, GCC).
Repo tạo từ template của BTC (remote `upstream`). README gốc BTC ở `docs/competition.md`; tài liệu đề: `docs/rules.md`,
`docs/evaluation.md`, `docs/submission.md`, `docs/guide.md`, `docs/gateway.md`, `data/README.md`.

## Quy tắc bắt buộc
1. **KHÔNG tự chạy `git commit`, `git push`, `git merge`, `git rebase`.** Xong một phần việc thì chỉ in
   lệnh gợi ý để người dùng tự chạy. Commit message theo `docs/gitflow.md`.
2. **Không sửa code/cấu hình của BTC**: `src/neorag/` (bộ chấm, runner, client), `configs/eval.toml`.
   Code của mình nằm trong package riêng, kế thừa `neorag.pipelines.RAGPipeline`.
3. **Luật thi** (`docs/rules.md`): LLM duy nhất `claude-haiku-5-5` qua gateway, KHÔNG truyền `temperature`;
   embedding `bge-m3` qua gateway hoặc model open-source không sinh văn bản chạy local.
   Cấm hard-code đáp án/`retrieved_ids`, cấm gán nhãn tay tập test, cấm chèn chỉ dẫn nhắm vào judge.
4. **Ngân sách 5 USD cả cuộc thi.** Trước khi chạy lệnh gọi `claude-haiku-5-5` (predict, run, evaluate
   không có `--no-judge`, chatbot), báo trước ước tính số lần gọi và chi phí, chờ mình đồng ý.
   Đo thật: 126 câu × 1 lần gọi ≈ 0.05 USD. Ưu tiên thử retrieval bằng metric không gọi LLM.
5. Không overfit public test: mọi tối ưu phải có lý do tổng quát, kiểm tra trên tập dev tách từ train.
6. Quyết định kỹ thuật quan trọng ghi thành ADR trong `docs/adr/` (kèm số liệu) — dùng cho pitching.
7. Không commit `.env`, `outputs/`, `.cache/`.

## Mục tiêu của dự án: HỌC, không chỉ chạy được
- **Làm từng bước nhỏ.** Mỗi lần một chức năng, rồi dừng lại để mình chạy thử.
- **Giải thích ngắn lý do** của mỗi quyết định không hiển nhiên (vì sao, đánh đổi gì). Không giảng bài.
- **Không tự ý thêm tính năng** ngoài plan. Có ý hay thì đề xuất, chờ mình đồng ý.
- **Khi mình hỏi "tại sao"**, trả lời khái niệm trước, code sau.
- Không chắc về API của thư viện thì nói rõ và đề xuất cách kiểm tra, đừng đoán.
- **Báo cáo sau mỗi bước** (dựa trên `git status` + `git diff --stat` thật, không viết theo trí nhớ). Không dán lại code.
  - **File sửa:** đường dẫn, và đã sửa gì (1 dòng/file).
  - **File thêm:** đường dẫn, và file đó có ý nghĩa gì trong pipeline (1 dòng/file).
  - **Cách kiểm tra:** lệnh để mình chạy thử (pytest, script...), kèm chi phí nếu có gọi LLM.
  - **Lệnh commit gợi ý** theo `docs/gitflow.md`.
  - Trong `docs/plan.md`: tick bước vừa xong ở mục "Tiến độ", cập nhật bảng "Bản đồ file của mình"
    (thêm dòng mới, sửa trạng thái). Sửa dòng "Đang làm" ở mục Trạng thái bên dưới.
- **Nhật ký học `docs/journal.md`** (bị `.gitignore`): mỗi khi làm xong một kỹ thuật hoặc chốt một quyết định,
  thêm một mục theo mẫu ở đầu file: vấn đề (kèm số đo), khái niệm, vì sao chọn / đánh đổi, phương án đã cân
  nhắc, kết quả trước → sau (split, commit, chi phí), bài học / câu hỏi Q&A. Chỉ ghi số liệu đã đo thật.
- Giao tiếp bằng tiếng Việt.

## Môi trường & lệnh
```bash
conda activate neorag
neorag run train --no-judge                                   # predict + chấm nhanh (tốn ~0.05 USD lần đầu)
neorag run train --pipeline <module>:<Class> --no-judge       # chạy pipeline của mình
neorag evaluate train --no-judge                              # chỉ chấm lại file đã có, không gọi LLM
neorag run public_test                                        # tạo outputs/public_test_submission.zip
pytest && ruff check .
```
Code của mình nằm trong package `src/s2b_rag/` (khai báo trong `pyproject.toml`). Bản cài editable
đã trỏ vào `src/`, nên gọi được ngay, ví dụ `--pipeline s2b_rag.pipeline:MyPipeline`.
KHÔNG đặt code ở gốc repo: lệnh `neorag` không import được module ở đó.

## Cách tính điểm (tóm tắt)
score = 0.5 × mean(Recall@5, MRR@5, NDCG@5) + 0.5 × mean(String Presence, ROUGE-L, Groundedness,
Response Relevancy, 1 − Noise Sensitivity, Abstention). Groundedness tính trên text của TẤT CẢ
`retrieved_ids` (tối đa 10). Đáp án mẫu ngắn (~210 ký tự).

## Trạng thái hiện tại
- Baseline BTC: train retrieval 0.72, SP 0.74, ROUGE-L 0.43; public 52.40 (ngưỡng 53.40).
- **Tiến độ và bản đồ file: trong `docs/plan.md`** (mục "Tiến độ", "Bản đồ file của mình"). Đọc hai mục đó đầu mỗi phiên.
- Đang làm: **CHỐT `S2BPipeline` = R12MBR** (public #17 **89.99**) cho private 08/11. Lịch sử: A5 (#9 89.15); 06/11: F (prompt chép nguyên văn + ví dụ, #11 89.01) và MBR chọn câu trả lời trong 4 bản A/A4/E2/F (`S2BPipelineMBR`, #12 89.26, gấp đôi số lần gọi, dưới ngưỡng 89.45) → giữ A5. 07/11 thử thêm 5 hướng không tốn thêm lần gọi (chọn câu theo Relevancy/độ ngắn/bám tài liệu,
  hậu xử lý ROUGE, quota CLB hạng 11–20): đều ≈0 hoặc âm trên holdout (xem `docs/journal.md`) → CHỐT A5, DỪNG tối ưu điểm.
  07/11 (người dùng muốn thử tiếp): **R11** `S2BPipelineR11` = A5 + xếp lại top-5 theo đồng thuận 2 lần đọc (holdout retrieval
  96.06 → 97.58, dev không đổi, 0 USD) → **public #13 89.42** (retrieval 93.93 → 94.50) → `S2BPipeline` = R11; rồi **R12** (kiểm tra toàn corpus cho câu gắn cờ, #15 retrieval 94.82, gốc 89.36 do nhiễu judge) → **`S2BPipeline` = R12** từ 07/11. R13 (kiểm tra mọi câu, ~0.40 USD) = R12 trên dev/holdout → bỏ. **Public #16 R12+MBR 89.59** (cao nhất) → sinh lại câu trả lời khi R12 bổ sung đoạn → **public #17 89.99** (nộp lại 89.92) → **CHỐT `S2BPipeline` = R12MBR** (4 lần đọc/câu) + sửa lỗi tách bản A + đồng bộ id theo câu trả lời mới (07/11). Tiếp theo: chống sự cố ngày thi, slide, code freeze 07/11; private 08/11 (42 câu).
