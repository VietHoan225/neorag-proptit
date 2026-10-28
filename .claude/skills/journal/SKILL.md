---
name: journal
description: Ghi một lần vấp (bug, kết quả thí nghiệm bất ngờ, hiểu sai khái niệm) của dự án NeoRAG vào docs/journal.md theo mẫu "Vấp" có sẵn trong file. Dùng khi người dùng gõ /journal hoặc vừa giải quyết xong một vấn đề đáng nhớ.
---

# Journal

`docs/journal.md` là nhật ký học cá nhân (bị `.gitignore`). Skill này chỉ ghi mục **"Vấp"**; mục
**kỹ thuật** (vấn đề, khái niệm, đánh đổi, kết quả) vẫn ghi sau mỗi bước theo quy tắc trong `CLAUDE.md`.

1. Dựa vào tham số người dùng đưa và những gì vừa xảy ra trong phiên, soạn một mục theo mẫu "Vấp" ở đầu
   `docs/journal.md`:
   - **Chuyện gì xảy ra:** triệu chứng cụ thể — lệnh đã chạy, lỗi, số liệu (vd "R@5 dev_noaccent 0.44 → 0.41").
   - **Nguyên nhân:** chỉ ghi nguyên nhân đã xác minh; nếu chưa chắc, ghi "nghi ngờ: ...".
   - **Cách sửa:** file nào, commit nào (`git rev-parse --short HEAD` nếu đã commit).
   - **Bài học:** một câu, tổng quát hoá được.
2. Tiêu đề: `## <ngày hôm nay YYYY-MM-DD> · Vấp: <tóm tắt> (nhánh <git branch --show-current>)`.
3. **Cho người dùng xem bản nháp trước khi ghi**, mời họ viết lại phần "Bài học" bằng lời của chính họ.
   Chỉ ghi sau khi họ đồng ý.
4. Thêm vào **cuối** file, không sửa các mục cũ. Chỉ ghi số liệu đã đo thật trong phiên, không bịa.

Ngắn gọn: mỗi mục tối đa khoảng 8 dòng. Tiếng Việt, thuật ngữ kỹ thuật giữ nguyên tiếng Anh.
