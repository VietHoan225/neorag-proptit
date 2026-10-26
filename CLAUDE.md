# CLAUDE.md — NeoRAG Cup 2026 (bài dự thi cá nhân)

Hệ thống RAG hỏi đáp về cộng đồng S2B (CLB Lập Trình PTIT, Multimedia PTIT, GCC).
Repo tạo từ template của BTC (remote `upstream`). Tài liệu đề: `README.md`, `docs/rules.md`,
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
- **Code xong thì tóm tắt:** file thêm/sửa, mỗi file một dòng nói làm gì. Không dán lại code.
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
- Baseline train (`--no-judge`): Retrieval 0.72, SP 0.74, ROUGE-L 0.43. Public baseline: 52.40 (ngưỡng 53.40).
- Điểm yếu: câu "không dấu/viết tắt" (R@5 0.44); câu trả lời dài TB 452 ký tự.
- Việc tiếp theo: lên plan (Plan mode).

## Bản đồ file của mình
Chỉ ghi file do mình thêm/sửa; file của BTC xem `docs/guide.md` mục 8.

| File | Tóm tắt | Trạng thái |
|---|---|---|
| `CLAUDE.md` | Hướng dẫn cho Claude Code | Đang dùng |
| `.claude/settings.json` | Chặn git commit/push/merge/rebase/reset --hard; hỏi trước khi chạy lệnh tốn ngân sách (`neorag run/predict/evaluate`, streamlit) | Đang dùng |
| `docs/gitflow.md` | Quy ước nhánh, Conventional Commits | Đang dùng |
| `docs/adr/0000-template.md` | Mẫu ADR | Template |
| `pyproject.toml` | Thêm `src/s2b_rag` vào danh sách package được đóng gói | Đã sửa (1 dòng) |
| `src/s2b_rag/` | Package chứa pipeline của mình | Mới tạo, rỗng |
