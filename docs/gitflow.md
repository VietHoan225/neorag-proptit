# GitFlow & quy ước commit

## Nhánh
| Nhánh | Mục đích |
|---|---|
| `main` | Bản ổn định, chạy được (bản nộp / demo) |
| `develop` | Nhánh tích hợp, merge các feature vào đây |
| `feature/<tên>` | Mỗi tính năng/thí nghiệm một nhánh, tách từ `develop` |
| `fix/<tên>` | Sửa lỗi |
| `docs/<tên>` | Chỉ thay đổi tài liệu |

Luồng: `feature/*` → PR vào `develop` → khi ổn định, PR `develop` → `main` (gắn tag, ví dụ `v0.1-baseline`).

## Commit message (Conventional Commits)
```
<type>(<scope>): <mô tả ngắn>
```
`type`: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `exp` (thí nghiệm)
Ví dụ: `feat(retrieval): add BM25 + dense hybrid retriever`

## Quy tắc với Claude Code
Claude **không** tự chạy `git commit` / `git push` / `merge` / `rebase` (bị chặn trong `.claude/settings.json`).
Sau mỗi phần việc, Claude chỉ **gợi ý** lệnh; người dùng tự chạy.
