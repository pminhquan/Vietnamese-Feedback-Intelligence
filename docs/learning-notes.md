# Ghi chú học tập cho chủ sở hữu dự án (Owner Learning Notes)

> Tài liệu này viết bằng tiếng Việt để bạn nắm chắc nền tảng ML của dự án.
> Mục tiêu: khi phỏng vấn thực tập, bạn TỰ giải thích được toàn bộ pipeline
> mà không cần AI trợ giúp. Đọc kỹ từng mục, sau đó làm **OWNER LEARNING
> TASKS** ở cuối mỗi phần — không làm bài hộ, không báo "đã hiểu" nếu chưa
> tự tay làm xong.

---

## 1. Văn bản trở thành đặc trưng TF-IDF như thế nào?

Máy học cổ điển không đọc được chữ — nó cần **vector số**. Với TF-IDF:

- **TF (term frequency):** một từ xuất hiện bao nhiêu lần trong câu này.
- **IDF (inverse document frequency):** một từ hiếm đến đâu trong toàn bộ
  tập huấn luyện. Từ xuất hiện ở mọi câu ("và", "của", "môn học") có IDF thấp
  — ít giá trị phân loại; từ hiếm ("quá tải") có IDF cao.
- **TF-IDF = TF × IDF**: trọng số cao khi từ đó xuất hiện nhiều trong câu
  nhưng hiếm trong corpus.

Trong dự án này, `TfidfVectorizer` được cấu hình với:

- `analyzer="word"`, `ngram_range=(1, 2)`: dùng từ đơn + cặp từ liền kề
  (bigram). Bigram quan trọng với tiếng Việt vì **phủ định** thường là cặp
  từ: "không tốt", "chưa hay" — từ đơn "tốt"/"hay" và "không" tách rời sẽ
  mất ngữ cảnh.
- `sublinear_tf=true`: tf thay bằng `1 + log(tf)` để một từ lặp 10 lần
  không chiếm ưu thế gấp 10 lần.
- `min_df=2`, `max_df=0.95`: bỏ từ xuất hiện trong <2 câu (nhiễu) hoặc
  >95% câu (quá phổ biến).
- `lowercase=True`: vectorizer tự chuyển thường (diacritics tiếng Việt
  được giữ nguyên — chỉ chữ hoa/thường thay đổi, không mất dấu).

Kết quả: mỗi câu → 1 vector độ dài bằng số từ trong **từ điển (vocabulary)**
chỉ học từ tập train. Vector thưa (sparse) vì một câu chỉ chứa vài chục từ.

**Điểm mấu chốt:** từ điển học từ train — nếu học cả từ validation/test là
rò rỉ dữ liệu (xem mục 7).

## 2. Logistic Regression học cái gì?

Không phải "hồi quy" trong ý dự đoán giá trị liên tục — nó là **bộ phân loại**:

1. Tính điểm `z = w·x + b` — mỗi đặc trưng TF-IDF `x_i` có một trọng số
   `w_i` học được. `w_i > 0` đẩy về lớp tích cực, `w_i < 0` đẩy về lớp tiêu
   cực.
2. Áp softmax thành xác suất 3 lớp (negative/neutral/positive) — tổng bằng 1.
3. Huấn luyện = tìm `w`, `b` **tối thiểu hóa cross-entropy** trên tập train,
   có điều kiện L2: hàm mất mát + `1/(2C)·‖w‖²`.

Tham số `C` là **nghịch đảo của cường độ điều chuẩn**: `C` nhỏ → phạt trọng
số nặng → mô hình đơn giản, dễ underfit; `C` lớn → ít phạt → dễ overfit.
Đó là lý do không gian tìm kiếm của chúng ta thử `C ∈ {1.0, 4.0}`.

Với TF-IDF, trọng số học được rất "đọc được": `w["không tốt"]` âm lớn,
`w["dễ hiểu"]` dương lớn. Đây là điểm hay để demo explainability đơn giản.

## 3. Vì sao train và validation phải tách rời?

Vì mục tiêu thật không phải "học thuộc dữ liệu đã thấy" mà là **tổng quát
hóa sang dữ liệu chưa thấy**:

- Train: mô hình được phép nhìn và fitting.
- Validation: **không fitting** — chỉ dùng để chọn siêu tham số/chọn mô
  hình. Nó mô phỏng "dữ liệu tương lai" trong quá trình ra quyết định.
- Test: chỉ chạm đúng một lần, sau khi chốt mô hình — con số cuối cùng.

Nếu chọn siêu tham số trên test rồi báo test accuracy, bạn đã "chọn đề thi
sau khi xem đáp án" — con số sẽ lạc quan một cách có hệ thống (selection
bias). Trong repo này, `vfi train` không hề đọc thư mục test; `vfi evaluate`
là lệnh duy nhất chạm test.

## 4. Overfitting là gì?

Mô hình **học thuộc nhiễu** của tập train thay vì quy luật chung. Dấu hiệu
kinh điển: train accuracy rất cao nhưng validation accuracy thấp (độ lệch
lớn giữa hai số). Trong dự án này:

- LR với `C` quá lớn + TF-IDF word 1-2 gram trên corpus nhỏ dễ overfit.
- `class_weight="balanced"` giúp lớp hiếm không bị bỏ rơi, nhưng không chữa
  overfit.
- Cách phát hiện trong dự án: so `validation_report` giữa các trial trong
  `reports/artifacts/training_report.json` — trial có train tốt mà val kém
  chính là kẻ overfit.

## 5. Vì sao Macro-F1 quan trọng?

- **Accuracy** bị lừa khi dữ lệch lớp: nếu 80% feedback là positive, chỉ cần
  đoán mò "positive" là được 80% — majority baseline của chúng ta tồn tại
  chính để phơi bày điều này.
- **F1** kết hợp precision (đoán positive thì có đúng không) và recall (đã
  bắt được bao nhiêu % positive thật).
- **Macro** = tính F1 từng lớp rồi lấy trung bình **không theo trọng số
  support** → lớp neutral hiếm vẫn quan trọng ngang lớp positive. Micro-F1
  (theo trọng số) sẽ che khuất lớp hiếm.

Nên: chọn mô hình theo **validation macro-F1** (`zero_division=0` được khai
báo rõ trong `src/vfi/evaluation/metrics.py`).

## 6. Đọc confusion matrix như thế nào?

Ma trận 3×3 trong `compute_metrics`: **hàng = nhãn thật, cột = nhãn dự
đoán**, thứ tự canonical `[negative, neutral, positive]`. Đường chéo = đoán
đúng. Đọc theo hàng: hàng negative nói "trong các câu negative thật, bao
nhiêu % bị đoán thành neutral/positive?".

Các mẫu lỗi đáng chú ý: negative→neutral nhiều (mô hình "né" phán đoán tiêu
cực), positive↔neutral nhầm nhau (ranh giới mờ), và nếu số lỗi tập trung
vào câu có phủ định ("không tệ") — đó là dữ liệu cho error analysis.

## 7. Vì sao data leakage làm hỏng thí nghiệm?

Leakage = thông tin từ dữ liệu đánh giá lọt vào quá trình huấn luyện. Hai
dạng thường gặp:

1. **Feature leakage:** fit từ điển TF-IDF trên toàn bộ dữ liệu (train +
   test) rồi mới chia — từ điển "biết" trước văn phạm của tập test. Trong
   repo này sklearn Pipeline fit vectorizer cùng lúc với classifier, chỉ
   trên train — kiểm chứng bằng test `test_vocabulary_learned_from_training_data_only`.
2. **Sample leakage:** cùng một câu xuất hiện ở hai split (feedback ngắn dễ
   trùng). `find_leakage` trong `src/vfi/data/validation.py` quét trùng
   chính xác sau chuẩn hóa chung. `vfi prepare-data` báo số lượng giữa train
   và validation. `vfi evaluate --preflight-only` kiểm tra cả test trước khi
   huấn luyện và chỉ đưa ra số liệu tổng hợp. `vfi train` dừng trước khi fit
   nếu train và validation trùng câu. Chốt mô hình trên validation rồi mới
   tính điểm test. Nếu xem điểm test và sửa mô hình nhiều lần dựa trên điểm
   đó, bạn đang gián tiếp tinh chỉnh trên test.

Hệ quả của leakage: chỉ số đánh giá **lạc quan giả tạo** — mô hình được điểm
cao vì đã "nhìn thấy đề" — và kết quả không tái lập được ngoài đời.

## 8. Khi chuyển từ classical ML sang PhoBERT thì gì thay đổi?

| Khía cạnh | TF-IDF + LR | PhoBERT |
| --- | --- | --- |
| Đặc trưng | sparse, thủ công theo từ điển | dense, contextual — cùng một từ có vector khác nhau theo ngữ cảnh |
| Ngôn ngữ | token hóa theo khoảng trắng | WordPiece + word segmentation tiếng Việt (VnCoreNLP/underthesea) |
| Dữ liệu | học từ đầu trên corpus nhỏ | pretrain trên kho corpus khổng lồ, fine-tune nhẹ |
| Tài nguyên | CPU vài giây | GPU khuyến nghị; hàng trăm MB weights |
| Pipeline | sklearn Pipeline | PyTorch Dataset/DataLoader/Trainer, early stopping, checkpoint |
| Rủi ro | underfit/overfit đơn giản | chi phí, seed variance giữa các lần fine-tune, cần nhiều kỷ luật hơn về checkpoint/eval |

Điều **không thay đổi**: chia split, chọn trên validation, chốt rồi mới chấm
điểm test, macro-F1 làm metric chính, và error analysis. PhoBERT không cứu được
pipeline bẩn.

---

## OWNER LEARNING TASKS — Tiền xử lý dữ liệu

1. Mở `src/vfi/data/loading.py`. Tự giải thích: vì sao dùng NFC thay vì NFD?
   Vì sao KHÔNG lowercase/strip dấu trong `normalize_text`?
2. **Bài tập:** sửa `normalize_text` thành bản lowercase tất cả, chạy
   `pytest tests/test_loading.py -q`, quan sát test nào fail và vì sao test
   được viết để bảo vệ quyết định thiết kế này. Trả lại nguyên bản sau khi
   xong.
3. **Bài tập:** viết (trên giấy hoặc notebook, không commit) ví dụ 1 câu có
   "không" đứng trước tính từ, giải thích vì sao bigram "không tốt" giúp
   phân loại hơn unigram "tốt" + "không" riêng lẻ.

## OWNER LEARNING TASKS — Huấn luyện mô hình

1. Đọc `src/vfi/models/baseline.py`. Tự giải thích vòng lặp
   `run_small_search`: ai được fit ở đâu, mô hình tốt nhất được chọn theo
   số nào, tie-break hoạt động ra sao?
2. **Bài tập:** thay `C=4.0` thành `C=1000.0` trong 1 trial (chỉ trong máy
   cục bộ, không commit), chạy train trên dữ liệu thật khi đã có, so sánh
   validation macro-F1 — ghi lại hiện tượng (nếu có) cho thấy overfit.
3. **Bài tập:** giải thích cho một người bạn (không nhìn tài liệu) vì sao
   `MajorityBaseline` có ý nghĩa khoa học dù "ngu" — gợi ý: nó là sàn tham
   chiếu (lower reference).

## OWNER LEARNING TASKS — Đánh giá

1. Tính tay trên giấy: `y_true = [positive, negative, neutral, positive]`,
   `y_pred = [positive, negative, negative, positive]` → accuracy, precision
   per class, macro-F1. Kiểm chứng bằng cách gọi `compute_metrics` trong
   Python REPL (`python -i -c "from vfi.evaluation.metrics import compute_metrics"`).
2. **Bài tập:** đọc confusion matrix trong `eval_tfidf-logreg.json` (khi có
   dữ liệu thật): lớp nào bị nhầm vào lớp nào nhiều nhất, hãy đặt 1 giả thuyết
   ngôn ngữ học cho mẫu lỗi đó.
3. **Bài tập:** giải thích khác biệt zero_division=0 so với hành vi mặc định
   của sklearn (`warn`) — và vì sao báo cáo phải khai báo lựa chọn này.

---

**Cam kết trung thực:** hoàn thành bài tập là do bạn tự đánh dấu. Người khác
(hoặc AI) không được phép tick hộ. Khi phỏng vấn, interviewer sẽ hỏi "vì sao"
— không phải "code nào".
