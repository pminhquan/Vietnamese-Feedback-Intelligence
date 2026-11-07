# Kế hoạch bằng chứng portfolio cho dự án VFI (Portfolio Evidence Plan)

> Viết ngày 2026-11-07 trên nhánh `research/phase2-readiness`. Mục tiêu: chỉ
> ra **bằng chứng cụ thể nào** trong repo chứng minh **kỹ năng nào** cho nhà
> tuyển dụng, danh sách bằng chứng cần sản xuất trong Phase 2, và cách giới
> thiệu dự án trong CV/phỏng vấn.
>
> Nguyên tắc số 0 — **không phát minh thành tích:** mọi điều nói về dự án
> phải trỏ được đến một artifact thật (file, commit, JSON). Hiện dự án chưa
> có kết quả thí nghiệm trên dữ liệu thật (dataset UIT-VSFC chờ ký thỏa
> thuận — `docs/dataset-uit-vsfc.md`), nên phần "đã có" và phần "cần sản
> xuất" được tách rời rõ ràng bên dưới. Điền mục nào xong mục đó; không điền
> trước.

---

## Phần A — Artifact nào chứng minh kỹ năng nào

| Artifact trong repo | Kỹ năng nhà tuyển dụng thấy | Lý do thuyết phục |
| --- | --- | --- |
| `src/vfi/data/validation.py` (`find_leakage`, kiểm tra phân phối, fingerprint) | ML integrity: hiểu data leakage là rủi ro số 1, không phải mục từ điển | Phỏng vấn viên kỹ luôn đào phần này; code có test riêng bảo vệ |
| `test_vocabulary_learned_from_training_data_only` trong `tests/` | Testing discipline hướng vào **quyết định thiết kế**, không phải coverage | Test không kiểm "chạy không lỗi" mà kiểm quy tắc ML có bị phá không |
| `src/vfi/evaluation/metrics.py` (`zero_division=0` khai báo, `labels=` ép 3 lớp, protocol latency có tham số ghi kèm JSON) | Đánh giá chuẩn xác + văn hóa khai báo hành vi ngầm | Nhiều ứng viên không biết hành vi mặc định của sklearn; đây là khác biệt |
| `meta.json` + `load_model` trusted-artifact policy (config hash, dataset sha256, seed, versions, cảnh báo lệch sklearn major) | MLOps mindset: provenance, reproducibility, an toàn artifact | Chứng minh bạn nghĩ đến "ai/gì đã tạo artifact này" trước khi deserialize |
| `docs/experiment-protocol.md` + `docs/dataset-uit-vsfc.md` (verdict BLOCKED có bằng chứng nguồn) | Data governance/ethics: không tự ý dùng dữ liệu khi chưa xác minh license | Rất hiếm sinh viên ghi được "blocked" có dẫn chứng thay vì tải trộm; điểm nhấn kể trong phỏng vấn |
| `MajorityBaseline` (one-hot probabilities được báo cáo "as-is" cho trung thực) | Tư duy baseline: biết nói "model của tôi chỉ tốt hơn đoán mò bao nhiêu" | Trả lời trực tiếp câu phỏng vấn "cần baseline để làm gì" |
| `SEARCH_SPACE` 4 ứng viên + tie-break có tài liệu + `SearchTrial.fit_seconds` | Kỷ luật hyperparameter search nhỏ, có lý do, ghi thời gian | Trái với kiểu "grid search 1000 tổ hợp không nghĩ" |
| `docs/learning-notes.md` + tài liệu này (bài tập owner tự làm) | Growth mindset + sở hữu kiến thức thật | Kể được khi interviewer hỏi "bạn học cái gì trong dự án này" |
| CI (ruff + pytest, CPU-only) + `scripts/verify.py` | Software engineering cơ bản: lint, test, cổng chất lượng | Yêu cầu mặc định của thực tập kỹ thuật — có là điểm cộng nền |
| PR trail: Draft PR, review, commit message chuẩn (ví dụ commit `fix(lint)` kèm diễn giải root cause gitignore ẩn `src/vfi/models/`) | Collaboration + root-cause debugging | Commit message kể được câu chuyện debug là bằng chứng sống |

---

## Phần B — Bằng chứng hiện có (Phase 1, thật) và bằng chứng cần sản xuất (Phase 2)

### B.1 Đã có ngay hôm nay (liệt kê được vào CV không cần đợi gì)

- [x] Repo có cấu trúc package `src/vfi/` (data / models / evaluation / cli),
      cài được `pip install -e ".[dev]"`, Python 3.12.
- [x] Bộ test tập trung trên synthetic fixtures (loading, validation, splitting,
      baseline, metrics, CLI) — `pytest` xanh trong CI.
- [x] Protocol đánh giá có tài liệu (split discipline, metric, latency,
      reproducibility record) — `docs/experiment-protocol.md`.
- [x] Điều tra dataset có verdict BLOCKED + nguồn + đường unblock.
- [x] Ghi chú học tập owner bằng tiếng Việt với bài tập tự kiểm.
- [x] CLI suy luận cục bộ `vfi predict` hoạt động trên artifact đã train.
- **Lưu ý chân thành:** mục này CHỨNG MINH KỸ NĂNG QUY TRÌNH, không chứng
  minh kết quả mô hình — vì chưa có run thật. Nếu CV nhắc dự án, dùng đúng
  từ như vậy.

### B.2 Checklist bằng chứng cần sản xuất trong Phase 2

> Quy tắc chia việc (như `docs/phase2/owner-learning-roadmap.md` mục 0):
> agent được viết code/docs cơ khí; **owner phải chạy, diễn giải và sở hữu
> mọi con số**. Mục nào cũng có "định nghĩa hoàn thành" để không tick ảo.
> Bắt đầu chỉ khi: (a) Phase 2 được phê duyệt, (b) dataset hợp pháp đã có.

| # | Bằng chứng | Định nghĩa hoàn thành | Ai sở hữu |
| --- | --- | --- | --- |
| 1 | `training_report.json` trên dữ liệu thật (trial TF-IDF + majority) | File nằm trong `reports/artifacts/`, có config hash + fingerprint + seed; owner giải thích được từng trường | OWNER chạy; AGENT viết schema |
| 2 | Bảng so sánh trung thực baseline vs PhoBERT | Cùng split, cùng macro-F1 protocol; 3 seed ({42, 7, 2026}) cho PhoBERT, báo mean ± std; bảng trong `reports/` + một đoạn owner viết "đọc bảng này nghĩa là gì" | OWNER viết đoạn đọc bảng |
| 3 | Confusion matrix + error analysis 15 mẫu sai | Mỗi mẫu: đầu vào rút gọn, dự đoán vs thật, giả thuyết bằng lời owner; tối thiểu 1 giả thuyết được kiểm chứng bằng mini behavioral test (kiểu CheckList: bất biến "không + tính từ") | OWNER 100% phần giả thuyết; AGENT chỉ trích xuất CSV |
| 4 | So sánh seed variance | 3 seed {42, 7, 2026} (chốt trước, không bỏ seed), bảng macro-F1 mean ± std kèm nhận xét biến động fine-tune vs TF-IDF | OWNER diễn giải; AGENT chạy lặp |
| 5 | Bảng latency theo protocol Phase 1 | `measure_latency` (batch 1, 10 warmup, 100 timed, seed) cho TF-IDF CPU / PhoBERT PyTorch / PhoBERT ONNX; protocol ghi kèm JSON | OWNER chạy + diễn giải; AGENT viết script export |
| 6 | Ablation nhỏ có tài liệu | Ví dụ: with/without word segmentation; with/without warmup — mỗi ablation 1 đoạn "tại sao làm, kết quả, quyết định" | OWNER chốt kết luận |
| 7 | Learning curve + early stopping | Bảng train/val theo epoch + epoch dừng; owner chọn patience và ghi lý do | OWNER chọn; AGENT vẽ curve |
| 8 | PR description mẫu có trade-off | Ít nhất 1 PR Phase 2 có phần "Alternatives considered / Why not X" viết tay | OWNER viết |
| 9 | README dự án cập nhật trạng thái thật | Không còn chữ "will be finalized"; có mục Results chỉ dẫn đúng nơi số thật nằm | OWNER review |
| 10 | Story bank phỏng vấn (file cá nhân, không commit) | 3 chuyện có thật: (a) gitignore ẩn models khỏi ruff; (b) mapping nhãn lệch chiều; (c) blocker license. Mỗi chuyện: bối cảnh → debug → bài học. Kiểm tra: kể lại 90 giây không nhìn | OWNER |

**Chống tick ảo:** mỗi mục trên chỉ được tick khi có artifact thật trên đĩa
hoặc trong repo + owner trả lời được câu hỏi "vì sao" của chính artifact đó.

---

## Phần C — Giới thiệu dự án trong CV / phỏng vấn

### C.1 Ba câu tiếng Việt (bản dùng khi phỏng vấn nói tiếng Việt)

> 1. "Tôi xây VFI — một hệ thống phân tích cảm xúc feedback sinh viên tiếng
>    Việt thành 3 lớp, được thiết kế theo chuẩn quy trình production nhỏ:
>    kiểm tra leakage, artifact có metadata provenance, chọn mô hình theo
>    validation macro-F1 và chỉ chạm test một lần."
> 2. "Phase 1 là baseline kỷ luật — MajorityBaseline và TF-IDF + Logistic
>    Regression với bộ test bảo vệ từng quyết định thiết kế, CI xanh — và
>    Phase 2 sẽ mở rộng sang fine-tune PhoBERT trên cùng split, cùng
>    metric để so sánh trung thực."
> 3. "Điểm tôi tâm đắc nhất là sự trung thực của quy trình: khi license
>    dataset chưa xác minh được, dự án ghi rõ BLOCKED kèm bằng chứng thay
>    vì đưa ra con số không tái lập được."

*(Câu 3 là điểm khác biệt hóa mạnh nhất — nhà tuyển dụng đọc được độ chín
chuyên nghiệp.)*

### C.2 Ba câu tiếng Anh (bản dùng cho CV tiếng Anh / interview tiếng Anh)

> 1. "I built VFI, a Vietnamese 3-class student-feedback sentiment system
>    engineered like a small production pipeline: leakage checks, provenance
>    metadata on every artifact, model selection by validation macro-F1,
>    and a single touch on the test split."
> 2. "Phase 1 ships disciplined classical baselines (majority class and
>    TF-IDF + Logistic Regression) with tests that protect each design
>    decision and a green CPU-only CI; Phase 2 will extend this to fine-tuning
>    PhoBERT on the same splits and metrics for an honest comparison."
> 3. "What I value most is process honesty: while the dataset license was
>    unverified, the project records a documented BLOCKED verdict with
>    sources instead of reporting unreproducible numbers."

### C.3 Những điều KHÔNG được nói (danh sách cấm để giữ uy tín)

- Không nói "đã đạt X% macro-F1" khi chưa có run thật trên dữ liệu hợp pháp.
- Không gọi pipeline synthetic-fixture là "đã kiểm chứng chất lượng".
- Không ghi "fine-tuned PhoBERT" vào CV trước khi điều đó xảy ra thật.
- Không che blocker license — chính nó là bằng chứng tốt nhất.
- Không nói "AI code giúp tôi toàn bộ" (tự hạ giá trị) cũng như "tôi làm
  100% một mình không dùng công cụ nào" (không đúng thực tế). Câu an toàn:
  "Tôi dùng công cụ hỗ trợ viết code cơ khí, nhưng mọi quyết định ML và toàn
  bộ kết quả là của tôi — hỏi tôi bất kỳ phần nào."

---

## Phần D — Cách trình bày ở từng kênh

1. **CV (3–4 dòng mục dự án):** dùng đúng 3 câu ở C.2, thêm tech stack:
   Python, scikit-learn, pytest, CI (GitHub Actions); Phase 2: PyTorch,
   Transformers, PhoBERT, VnCoreNLP. Kèm link repo — vì thế repo phải trình
   bày đúng trạng thái thật (checklist B.2 mục 9).
2. **Phỏng vấn kỹ thuật:** mở repo thật, đi theo dòng dữ liệu (loading →
   validation → model → metrics → artifact), dừng lâu ở 2 điểm khác biệt:
   leakage test và trusted-artifact policy; kể 1 chuyện từ story bank (B.2
   mục 10) khi có câu hỏi về debug.
3. **Email/cover letter (2 câu):** một câu VFI là gì + một câu "kỷ luật
   trung thực" (BLOCKED verdict). Đủ để interviewer tò mò, không spam.

---

**Tóm lại:** portfolio của dự án này bán **kỷ luật và trung thực**, không
bán số liệu. Mọi bằng chứng trong checklist Phần B.2 phải ra đời từ run
thật trên dữ liệu hợp pháp và do owner tự sở hữu. Đó cũng chính là thứ mà
phỏng vấn viên của internship ML đang tìm kiếm ở một sinh viên năm ba.
