# ADR-0001: Đo bằng tập dev/holdout tách từ train, kèm bản bỏ dấu và bản "đời thường"

- **Ngày:** 2026-10-28 (bổ sung bản đời thường 2026-11-03)
- **Trạng thái:** Chấp nhận

## Bối cảnh
- Chỉ có 126 câu train có đáp án. Public Test (42 câu) chỉ trả về điểm tổng, không cho xem câu nào sai, và bảng
  xếp hạng lấy lần nộp gần nhất. Private Test (08/11) mới là điểm thật, và ngày đó không được sửa code.
- Nếu chọn cấu hình theo điểm public thì là overfit public. Bootstrap trên kết quả holdout (42 câu, 2000 lần) cho độ
  lệch chuẩn **±2.6 điểm**, chỉ do bốc trúng câu nào.
- Ngân sách LLM cả cuộc thi là 5 USD, nên phần lớn phép đo phải miễn phí (không gọi haiku).
- Câu hỏi train khác public: train chỉ có 26/126 câu không dấu. Đo ngày 03/11 sau lần nộp #2:

  | Đặc điểm câu hỏi | dev | public |
  |---|---|---|
  | Không dấu | 21% | **48%** |
  | Có teencode | 13% | **38%** |
  | Độ dài trung bình (âm tiết) | 17 | 22 |
  | Câu ≥ 2 ý (đếm từ hỏi) | 42% | **74%** |
  | Retrieval (`hybrid_r10`) | 0.932 | **0.822** (nộp #2) |

## Các phương án
1. **Đo thẳng trên public:** không xem được lỗi từng câu, sai số ±2.6, mỗi lần tốn một lượt nộp → chỉ dùng để xác nhận.
2. **Đo trên cả 126 câu train:** không còn tập nào chưa bị nhìn để kiểm tra overfit.
3. **k-fold cross-validation:** chính xác hơn nhưng nhân chi phí LLM lên k lần → không hợp ngân sách.
4. **Tách dev/holdout cố định + các bản biến đổi tự động của câu hỏi** (chọn).
5. **Lấy teencode từ câu hỏi public:** sát public hơn, nhưng là học từ tập thi → bỏ.

## Quyết định
Chọn **#4**. Script `python -m s2b_rag.eval.split` (`src/s2b_rag/eval/split.py`) sinh ra 6 file trong
`experiments/splits/`, seed 2026 cố định:

```
data/train/questions.jsonl (126)
 ├─ dev.jsonl (86)      ─► dev_noaccent.jsonl, dev_casual.jsonl
 └─ holdout.jsonl (40)  ─► holdout_noaccent.jsonl, holdout_casual.jsonl
```

- **Tách phân tầng:** mỗi `type` góp cùng tỉ lệ vào holdout; trong mỗi type lấy cách đều theo CLB. Mọi type có ở cả hai
  tập. Holdout 40 câu, gần bằng cỡ một bộ public.
- **Bản bỏ dấu (`*_noaccent`):** bỏ dấu toàn bộ câu hỏi, xử lý riêng chữ `đ`.
- **Bản đời thường (`*_casual`):** chữ thường, thay teencode với xác suất 70% (`không→ko/k`, `được→dc`, `em→e`,
  `câu lạc bộ→clb`...), bỏ "?" ở cuối, bỏ dấu. Bảng teencode **tự viết, không lấy từ câu public**.
- Chỉ trường `query` bị biến đổi; đoạn gold, đáp án và key facts giữ nguyên, nên vẫn chấm được bằng bộ chấm của BTC.
- **Cách dùng:** dev/dev_casual dùng để thử hằng ngày; holdout chỉ dùng ở mốc quyết định. Retrieval đo miễn phí bằng
  `python -m s2b_rag.eval.retrieval`; generation đo bằng `neorag run <file> --no-judge`, chỉ dùng judge ở mốc quan
  trọng. Quy tắc: chênh dưới 1 điểm trên holdout không coi là cải thiện; luôn so diff từng câu.

## Bằng chứng
- Thước đo cũ "public-like" (trung bình dev và dev_noaccent) đánh giá thấp vấn đề câu nhiều ý. Với teencode thì
  dev_casual chỉ kém dev_noaccent khoảng 1–2.5 điểm → giả thuyết "public khó vì teencode" sai phần lớn, tránh được
  việc tối ưu teencode vô ích. Chẩn đoán trên dev_casual chỉ ra câu nhiều ý: 3+ đoạn gold chỉ đạt 0.706, và 6/8 đoạn
  bị thiếu nằm ở hạng 6–10 → dẫn tới R9 (ADR-0005).
- **dev_casual dự đoán đúng mức tăng trên public:** R9 tăng retrieval dev_casual **+7.5** (0.892 → 0.9665), public
  thực tế tăng **+7.7** (81.84 → 89.56, lần nộp #3 → #4).

## Hệ quả
- Dễ hơn: mỗi thay đổi retrieval đo trong vài giây với 0 USD; xem được lỗi từng câu; quyết định có số liệu để giải thích.
- Khó hơn / rủi ro:
  - dev 86 câu vẫn nhỏ: nhóm câu ≥ 3 đoạn gold chỉ khoảng 9–12 câu, nên kết luận về nhóm này là bằng chứng yếu.
  - Holdout đã được nhìn nhiều lần, nên không còn hoàn toàn "chưa thấy".
  - Cache LLM giúp chạy lại 0 USD, nhưng cũng làm không đo được nhiễu sinh câu trả lời (chạy lại luôn ra câu cũ).
    Chênh lệch giữa các cấu hình khác nhau không được coi là nhiễu đo lại.
  - dev đang gần trần (retrieval dev_casual khoảng 0.98) → khó thấy các cải thiện nhỏ.
- Code: `src/s2b_rag/eval/split.py`, `src/s2b_rag/eval/retrieval.py`, `src/s2b_rag/eval/explog.py`;
  số liệu: `experiments/log.csv`.
