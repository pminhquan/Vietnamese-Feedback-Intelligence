# Lộ trình học tập Phase 2 cho chủ sở hữu dự án (Owner Learning Roadmap)

> Viết ngày 2026-11-07 trên nhánh `research/phase2-readiness`. Đây là tài liệu
> **nghiên cứu chuẩn bị Phase 2**, chưa phải lệnh bắt đầu code Phase 2 (việc
> bắt đầu Phase 2 vẫn cần phê duyệt theo `docs/roadmap.md`).
>
> Tài liệu này **bổ sung** cho `docs/learning-notes.md` (Phase 1) — không lặp
> lại nội dung cũ. Khi gặp chủ đề đã có trong Phase 1 notes, ghi chú này chỉ
> dẫn bạn **đọc lại mục nào** và **mở rộng thêm cái gì**.
>
> Sự trung thực: hiện tại dự án **chưa có** kết quả thí nghiệm thật (dataset
> UIT-VSFC còn chờ ký thỏa thuận sử dụng). Mọi bài tập dưới đây KHÔNG tạo ra
> số liệu để "khoe" — chúng tạo ra sự hiểu biết của chính bạn.

---

## 0. Quy tắc phân công: AI agent làm được gì, OWNER phải làm gì

**Quy tắc chia việc:** agent viết code và tài liệu; owner phải **tự giải
thích mọi quyết định ML** và **tự tay chạy từng bài tập**.

| Việc | Ai được làm |
| --- | --- |
| Viết code "cơ khí": glue code, test, config, CI, script xuất báo cáo, docs | **AGENT-MAY** |
| Viết khung training loop / callback / cấu hình siêu tham số theo kế hoạch đã duyệt | **AGENT-MAY**, nhưng OWNER phải review từng dòng |
| Chọn và **giải thích được** mọi quyết định ML: metric chính, learning rate, số epoch, patience, batch size, seed | **OWNER-MUST** |
| Tự tay chạy từng bài tập trong lộ trình này và tự ghi nhận kết quả | **OWNER-MUST** |
| Tự làm error analysis và **viết kết luận bằng lời của mình** (agent chỉ giúp trích xuất số liệu thô) | **OWNER-MUST** |
| Đánh dấu "đã hiểu" cho bất kỳ chủ đề nào | **CHỈ OWNER**, không ai (kể cả AI) được tick hộ |

Tại sao quy tắc này nghiêm ngặt: trong phỏng vấn thực tập, người hỏi không
quan tâm "AI nào đã viết code" — họ hỏi **"vì sao bạn chọn X"** và **"chạy
thử / sửa thử cho tôi xem"**. Nếu bạn không tự trả lời được, toàn bộ phần
code do agent viết là **nợ hiểu biết**, không phải thành tích.

Ký hiệu dùng trong tài liệu:

- `[OWNER-MUST]` — bắt buộc owner tự làm, không ủy quyền.
- `[AGENT-MAY]` — agent được viết giúp, owner review và hiểu.
- Bài tập gắn nhãn *(sau khi Phase 2 được duyệt + có dữ liệu hợp pháp)* chỉ
  thực hiện khi hai điều kiện đó xảy ra.

Tổng effort đề xuất: **~34 giờ**, chia 6 tuần (~5–6 giờ/tuần). Có thể kéo
dài tùy tiến độ đọc dataset.

---

## Tổng quan lộ trình

| # | Chủ đề | Effort | Loại |
| --- | --- | --- | --- |
| 1 | Đại số tuyến tính phía sau embeddings (dot product, cosine) | ~2.0h | Mới |
| 2 | TF-IDF thưa vs embedding dense có ngữ cảnh | ~1.5h | Ôn + mở rộng |
| 3 | Neural network cơ bản + backpropagation | ~4.0h | Mới |
| 4 | Cross-entropy loss cho 3 lớp | ~2.0h | Ôn + mở rộng |
| 5 | AdamW và vì sao weight decay tách riêng | ~2.0h | Mới |
| 6 | Learning rate, warmup, lịch decay | ~2.5h | Mới |
| 7 | Attention & Transformer (Q/K/V, self-attention, positional encoding) | ~5.0h | Mới |
| 8 | Tokenization: word vs subword; vì sao PhoBERT cần word segmentation | ~3.0h | Mới |
| 9 | Fine-tuning vs huấn luyện từ đầu | ~2.0h | Mới |
| 10 | Overfitting & early stopping | ~2.0h | Ôn + mở rộng |
| 11 | Data leakage | ~1.5h | Ôn |
| 12 | Macro-F1 (recap) | ~1.0h | Ôn |
| 13 | Error analysis | ~3.0h | Ôn + nâng cao |
| 14 | Tối ưu inference: ONNX, quantization, distillation (mức khái niệm) | ~2.5h | Mới |

Lịch tuần đề xuất: Tuần 1: #1–3 · Tuần 2: #4–6 · Tuần 3: #7 · Tuần 4: #8–9 ·
Tuần 5: #10–11 · Tuần 6: #12–14. Chủ đề #7 và #8 là nặng nhất — đừng dồn
một ngày.

---

## Phần A — Nền tảng toán và biểu diễn văn bản

### 1. Intuition đại số tuyến tính phía sau embeddings

**Mục tiêu:** hiểu vì sao "tương đồng ngữ nghĩa" thành phép nhân vector:
dot product lớn = hai vector cùng hướng; cosine chuẩn hóa độ dài để so sánh
công bằng; hai văn bản giống nhau → vector gần nhau.

**Đọc tối thiểu:**
- [Efficient Estimation of Word Representations in Vector Space (word2vec), Mikolov et al. 2013](https://arxiv.org/abs/1301.3781) — chỉ cần ý tưởng "từ ~ vector, similarity ~ cosine".
- Phần vector/similarity trong [scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) (ôn cách TF-IDF thành vector).

**Bài tập (VFI):**
1. `[OWNER-MUST]` Với `TfidfVectorizer` (như trong `src/vfi/models/baseline.py`),
   fit trên 5 câu feedback tự viết, rồi tính **cosine similarity thủ công bằng
   numpy** giữa "giảng viên dạy rất dễ hiểu" và "giảng viên giải thích rõ ràng",
   so với cặp "giảng viên dạy rất dễ hiểu" vs "học phí tăng vọt". Không dùng
   hàm cosine có sẵn — tự viết `cos(u, v) = u·v / (‖u‖‖v‖)`.
2. `[OWNER-MUST]` *(sau khi Phase 2 được duyệt + có dữ liệu hợp pháp)* Lấy
   embedding của cùng 3 câu đó từ PhoBERT (last hidden state, trung bình hoặc
   token [CLS]... lưu ý PhoBERT dùng token `<s>`), tính cosine bằng chính hàm
   numpy bạn đã viết ở bước 1. So sánh bảng TF-IDF vs PhoBERT: cặp nào đổi
   thứ hạng? Vì sao?

**Effort:** ~2 giờ.

**Tự kiểm tra:** Vì sao cosine giữa hai vector TF-IDF luôn ≥ 0, còn cosine
giữa hai embedding PhoBERT có thể âm? *(Đạt khi trả lời được: TF-IDF không có
giá trị âm nên góc nằm trong [0°, 90°]; embedding có thành phần âm nên góc
có thể > 90° → cosine < 0.)*

### 2. TF-IDF thưa vs embedding dense có ngữ cảnh

**Mục tiêu:** nắm được khác biệt căn bản: vector TF-IDF **thưa** (một chiều
mỗi từ trong từ điển, không biết ngữ cảnh), embedding **dense** (chiều cố
định, cùng một từ có vector **khác nhau theo câu chứa nó**).

**Đọc tối thiểu:** đọc lại **mục 1 và mục 8** của `docs/learning-notes.md`
(bảng so sánh TF-IDF+LR vs PhoBERT) — mục này là mở rộng, không lặp lại.

**Bài tập (VFI):** `[OWNER-MUST]` In ra 3 con số rồi giải thích: (a) số
chiều và số phần tử khác 0 (`nnz`) của vector TF-IDF cho 1 câu (dùng
`vectorizer.transform` trên câu đã fit ở bài tập #1); (b) số chiều embedding
PhoBERT-base là 768 (xem config model trên [model card vinai/phobert-base](https://huggingface.co/vinai/phobert-base));
(c) tỷ lệ "độ thưa" của TF-IDF = nnz / số chiều từ điển. *(Bài (a),(c) làm
được ngay; bài (b) chỉ cần đọc số liệu, không cần chạy model.)*

**Effort:** ~1.5 giờ.

**Tự kiểm tra:** Một câu chỉ chứa các từ bị loại bởi `max_df=0.95` và
`min_df=2` của vectorizer sẽ có vector TF-IDF như thế nào, và logistic
regression dự đoán gì cho câu đó? *(Đạt khi nói được: vector toàn số 0 →
chỉ còn bias `b` quyết định → dự đoán lớp có bias lớn nhất.)*

---

## Phần B — Mạng nơ-ron, hàm mất mát, tối ưu hóa

### 3. Neural network cơ bản + backpropagation

**Mục tiêu:** hiểu một layer là `y = f(Wx + b)`, và backpropagation chỉ là
**chain rule của đạo hàm**, tính từ mất mát ngược về từng trọng số. Đủ để
giải thích gradient descent và vì sao "zero_grad trước mỗi bước" quan trọng.

**Đọc tối thiểu:**
- [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — chain rule qua computational graph.
- [3Blue1Brown — Neural networks series](https://www.3blue1brown.com/topics/neural-networks) — xem 3 video đầu (network structure, gradient descent, backprop).
- [Karpathy: The spelled-out intro to neural networks and backpropagation (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — xem và **tự gõ theo** (không copy-paste).

**Bài tập (VFI):**
1. `[OWNER-MUST]` Trên giấy: viết mạng 1 hidden layer với input là vector
   TF-IDF 4 chiều (tự đặt số), hidden 3 nơ-ron (ReLU), output 3 lớp
   (softmax). Đạo hàm thủ công của mất mát theo MỘT trọng số cụ thể `w_11`
   (chain rule 3 bước: loss → output → hidden → w). Không cần số đẹp — cần
   đúng cấu trúc.
2. `[OWNER-MUST]` Làm theo micrograd: xây `Value` class tối thiểu (forward +
   backward cho phép nhân/cộng), chứng minh gradient tự tính trùng với kết
   quả đạo hàm tay ở bài 1 (thay ReLU bằng tanh cho khớp micrograd).
3. `[AGENT-MAY]` Agent viết một notebook khung (dữ liệu tổng hợp 3 lớp,
   MLP 1 layer, vòng train PyTorch) để owner chạy sau khi xong bài 2.
   Owner phải chạy và **sửa thử**: xóa `optimizer.zero_grad()` → quan sát
   loss diverge → giải thích vì sao.

**Effort:** ~4 giờ (nặng nhất phần B — đừng rút ngắn micrograd).

**Tự kiểm tra:** Trong `loss.backward()`, gradient của trọng số ở layer đầu
đến từ đâu? *(Đạt khi nói được chain rule: gradient layer trước = gradient
layer sau × đạo hàm cục bộ của phép biến đổi giữa hai layer; và vì sao
`zero_grad()` bắt buộc — gradient cộng dồn mặc định.)*

### 4. Cross-entropy loss cho 3 lớp

**Mục tiêu:** hiểu cross-entropy đo khoảng cách giữa phân phối dự đoán
(softmax) và nhãn one-hot; vì sao nó phạt nặng việc "tự tin sai"; kết nối
với việc logistic regression trong `baseline.py` cũng tối ưu chính hàm này.

**Đọc tối thiểu:**
- Đọc lại **mục 2** `docs/learning-notes.md` (logistic regression tối ưu cross-entropy + L2).
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — chú ý: nhận **logits**, không nhận softmax sẵn.

**Bài tập (VFI):**
1. `[OWNER-MUST]` Tính tay: `y_true = positive`, `p = [0.1, 0.2, 0.7]`
   (theo thứ tự canonical negative/neutral/positive) → CE = -ln(0.7) ≈ ?
   Tính tiếp cho `p = [0.1, 0.7, 0.2]` (tự tin sai lớp neutral) — so sánh
   mức phạt.
2. `[OWNER-MUST]` Kiểm chứng bài 1 bằng `torch.nn.functional.cross_entropy`
   và bằng `sklearn.metrics.log_loss` (đặt `labels` đủ 3 lớp). Ghi chú vì sao
   PyTorch nhận logits còn sklearn nhận xác suất.
3. `[OWNER-MUST]` Giải thích kết nối: `compute_metrics` trong
   `src/vfi/evaluation/metrics.py` chỉ **đánh giá** sau huấn luyện; còn CE
   là hàm **mục tiêu tối ưu trong lúc huấn luyện** — vì sao một mô hình có
   train loss thấp vẫn có thể macro-F1 validation thấp?

**Effort:** ~2 giờ.

**Tự kiểm tra:** Với 3 lớp, một dự đoán "phẳng" đều [1/3, 1/3, 1/3] có CE
bằng bao nhiêu, bất kể nhãn thật là gì? *(Đạt khi trả lời: ln(3) ≈ 1.0986 —
và hiểu ý nghĩa: CE của dự đoán ngẫu nhiên đều là baseline tự nhiên.)*

### 5. AdamW và vì sao decoupled weight decay

**Mục tiêu:** hiểu Adam (adaptive learning rate theo từng tham số) và điểm
mới của AdamW: weight decay được áp **trực tiếp lên trọng số**, không đi qua
gradient/adaptive moment — khác với "L2 thêm vào loss".

**Đọc tối thiểu:**
- [Decoupled Weight Decay Regularization (AdamW), Loshchilov & Hutter, ICLR 2019](https://arxiv.org/abs/1711.05101) — đọc abstract + mục 3 là đủ cho mức thực tập.
- [An overview of gradient descent optimization algorithms, Ruder](https://www.ruder.io/optimizing-gradient-descent/) — phần Adam.

**Bài tập (VFI):**
1. `[OWNER-MUST]` Trên notebook khung ở bài tập #3, train MLP với 3 cấu hình:
   SGD (không decay), Adam (weight_decay dạng L2 cộng vào gradient), AdamW.
   Giữ nguyên mọi thứ khác, so sánh train/validation loss, ghi bảng kết quả
   của chính bạn.
2. `[OWNER-MUST]` Viết 5 câu (bằng lời mình) giải thích cho một người bạn:
   vì sao cùng "regularization" mà AdamW tách decay khỏi gradient lại hợp
   lý hơn với adaptive optimizer. Liên hệ: logistic regression Phase 1 dùng
   L2 **trong** hàm mất mát (tham số `C`) — vì sao với sklearn cách đó ổn?

**Effort:** ~2 giờ.

**Tự kiểm tra:** AdamW cập nhật trọng số theo công thức nào (viết được dạng
`w ← w - lr·(...) - lr·λ·w`)? *(Đạt khi thấy rõ hạng decay KHÔNG bị chia cho
tích lũy gradient bình phương như hạng L2 trong Adam.)*

### 6. Learning rate, warmup, lịch decay (linear decay)

**Mục tiêu:** hiểu vì sao fine-tune BERT-size dùng LR nhỏ (2e-5 … 5e-5) —
con số này là **khuyến nghị của chính paper BERT**; warmup tránh bước đầu
quá mạnh khi Adam moment chưa ổn định; linear decay giúp hội tụ mịn cuối
quá trình.

**Đọc tối thiểu:**
- [BERT paper (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) — mục fine-tuning hyperparameters (LR ∈ {5e-5, 3e-5, 2e-5}, warmup 10% steps, linear decay). Đọc đúng phần đó.
- [Hugging Face: learning-rate schedulers](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) — hàm `get_linear_schedule_with_warmup`.
- [Hugging Face: Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) — `TrainingArguments` liên quan: `learning_rate`, `warmup_ratio`, `lr_scheduler_type`, `fp16`.

**Bài tập (VFI):**
1. `[OWNER-MUST]` Vẽ tay (hoặc matplotlib) đường LR khi `num_training_steps
   = 1000`, `warmup_ratio = 0.1`: LR đi từ 0 lên đỉnh ở step 100 rồi giảm
   tuyến tính về 0 ở step 1000. Tự tính 3 điểm (step 0, 100, 1000).
2. `[OWNER-MUST]` Trên notebook MLP (#3), chạy LR = 1e-3 và LR = 1e-5, so
   loss curve — quan sát 1e-3 diverge/dao động với Adam trên bài nhỏ, 1e-5
   hội tụ chậm. Ghi nhận: quy luật "LR lớn cần batch/decay cân bằng" và vì
   sao con số ổn định **phụ thuộc mô hình + dữ liệu**, không phải con số
   thần thánh.
3. *(sau khi Phase 2 được duyệt)* `[OWNER-MUST]` Fine-tune thử PhoBERT với
   LR = 2e-5 vs 1e-3 (1 epoch, 1 seed): ghi lại loss step đầu — chạy thử để
   TỰ thấy vì sao 1e-3 phá pretrain. `[AGENT-MAY]` Agent viết script chạy và
   log; owner quyết định dải LR đưa vào thí nghiệm chính thức và **viết lý
   do** vào training report.

**Effort:** ~2.5 giờ.

**Tự kiểm tra:** Vì sao warmup chỉ cần "10% số bước đầu" mà không warmup cả
chặng giữa? *(Đạt khi nói được: rủi ro lớn nhất là các bước đầu khi ước lượng
moment của Adam chưa đáng tin và trọng số đang xa vùng hội tụ tốt; giữa/quá
trình Adam đã ổn định.)*

---

## Phần C — Attention, Transformer, tokenization

### 7. Attention & Transformer (Q/K/V, self-attention, positional encoding)

**Mục tiêu:** giải thích được bằng lời: Q/K/V là ba phép chiếu của cùng một
vector từ; điểm attention = softmax(QKᵀ/√d)·V — mỗi từ "hỏi" (Q) và "nghe"
(K) các từ khác, rồi "trộn nội dung" (V) theo trọng số; vì sao cần positional
encoding (self-attention mặc định vô thứ tự).

**Đọc tối thiểu:**
- [The Illustrated Transformer, Jay Alammar](https://jalammar.github.io/illustrated-transformer/) — đọc toàn bộ, vẽ lại sơ đồ Q/K/V từ trí nhớ.
- [The Illustrated BERT, Jay Alammar](https://jalammar.github.io/illustrated-bert/) — đọc phần input embedding + [SEP]/[CLS].
- [Attention Is All You Need, Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) — chỉ mục 3.2 (Scaled Dot-Product Attention) và 3.5 (positional encoding).
- (Tùy chọn, rất đáng giờ) [Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — tự gõ theo để thấy attention sinh ra từ bigram model.

**Bài tập (VFI):**
1. `[OWNER-MUST]` Tính tay attention cho câu 3 token (dùng Q, K, V là số tự
   đặt 2 chiều): tính điểm QKᵀ, chia √d, softmax từng hàng (kiểm tra tổng
   mỗi hàng = 1), nhân V. Làm được trên giấy = hiểu cơ chế.
2. `[OWNER-MUST]` Viết 5 câu giải thích: với câu feedback "môn học không khô
   khan như tưởng tượng", self-attention làm được điều gì mà **bigram TF-IDF
   (Phase 1) làm được** và điều gì bigram **không làm được** (ví dụ: "không"
   cách "khô khan" 1 từ — bigram bắt được; nhưng "không... như tưởng tượng"
   phủ định cấu trúc cả mệnh đề thì cần truy toàn cục).
3. `[AGENT-MAY]` Agent viết đoạn code numpy kiểm chứng bài 1 (owner tự viết
   trước, rồi so kết quả — không dùng code của agent để thay bài làm tay).

**Effort:** ~5 giờ (chủ đề nặng nhất lộ trình — tách 2 buổi).

**Tự kiểm tra:** Vì sao cần chia cho √d trước softmax? *(Đạt khi giải thích:
dot product giữa vector chiều d lớn có phương sai tỉ lệ d → điểm quá lớn →
softmax saturate → gradient gần 0; chia √d giữ phương sai ổn định.)*

### 8. Tokenization: word vs subword (WordPiece/BPE); vì sao PhoBERT cần word segmentation tiếng Việt

**Mục tiêu:** hiểu ba tầng: (1) word tokenization theo khoảng trắng; (2)
subword tokenization (BERT dùng WordPiece, PhoBERT dùng BPE) — xử lý từ lạ
bằng cách cắt nhỏ; (3) đặc thù tiếng Việt: "từ" thường gồm nhiều âm tiết
("sinh viên", "giảng viên") nên trước khi vào PhoBERT phải **word
segmentation** (VnCoreNLP/underthesea) rồi nối âm tiết bằng `_`, vì corpus
pretrain của PhoBERT đã được segment như vậy — input phải khớp phân phối
pretrain.

**Đọc tối thiểu:**
- [Hugging Face: Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE vs WordPiece.
- [BERT README — tokenization](https://github.com/google-research/bert#tokenization) — WordPiece + `[CLS]`/`[SEP]`.
- [PhoBERT model card](https://huggingface.co/vinai/phobert-base) và [repo VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) — yêu cầu input đã word-segment, ví dụ dùng underscore.
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) và [underthesea](https://github.com/undertheseanlp/underthesea) — công cụ segmentation tiếng Việt (đọc README, chưa cần chạy).
- [PhoBERT paper, Nguyen & Nguyen, Findings of EMNLP 2020](https://arxiv.org/abs/2003.00744) — phần pretraining: corpus 20GB đã word-segment bằng VnCoreNLP, BPE 64K subword types, PhoBERT-base ≈ 135M tham số.

**Bài tập (VFI):**
1. `[OWNER-MUST]` Viết tay (không code): cột A = câu feedback thô "Kỳ này
   học phí tăng quá nhiều", cột B = sau segmentation "Kỳ này học_phí tăng
   quá nhiều". Đếm token theo từng cách: khoảng trắng, sau segmentation, và
   (đọc mô tả) subword BPE 64K của PhoBERT. Suy ra: segmentation làm từ điển
   subword "nhìn thấy" đơn vị nào.
2. `[OWNER-MUST]` Trả lời bằng lời: nếu quên segmentation, PhoBERT vẫn chạy
   được (không lỗi) — vậy sai ở đâu và tại sao chỉ số đánh giá sẽ kém đi?
   *(Gợi ý: mismatch với phân phối pretrain; âm tiết tách rời làm "học_phí"
   thành hai token vô nghĩa riêng lẻ.)*
3. *(sau khi Phase 2 được duyệt + có dữ liệu hợp pháp)* `[OWNER-MUST]` Chạy
   thật: tokenize 1 câu qua `AutoTokenizer.from_pretrained("vinai/phobert-base")`
   với và không có segmentation (VnCoreNLP hoặc underthesea), in token id +
   decode ngược. `[AGENT-MAY]` Agent viết script so sánh; owner tự chạy và
   diễn giải.

**Effort:** ~3 giờ.

**Tự kiểm tra:** Vì sao Phase 1 (TF-IDF) KHÔNG cần word segmentation nhưng
PhoBERT lại bắt buộc? *(Đạt khi nói được: TF-IDF tự học từ điển n-gram trên
chính corpus này nên syllable-bigram "học phí" tự hình thành trong vocabulary;
PhoBERT kế thừa tokenization cố định từ pretrain — không thể "học lại" từ
điển 64K BPE.)*

---

## Phần D — Chiến lược huấn luyện

### 9. Fine-tuning vs huấn luyện từ đầu

**Mục tiêu:** giải thích được chi phí/khả năng: pretrain từ đầu cần khối
lượng dữ liệu + GPU-hàng-tuần; fine-tune tận dụng ngôn ngữ học đã học, chỉ
điều chỉnh trên ~16k câu UIT-VSFC; biết khái niệm "frozen encoder + đầu phân
loại" như phương án tiết kiệm (fallback CPU/LLM-less).

**Đọc tối thiểu:**
- [BERT paper](https://arxiv.org/abs/1810.04805) — mục 3 (so sánh pretrain + fine-tune) và bảng so sánh với "BERT-base trained from scratch" nếu tìm được trong phần ablation.
- [PhoBERT paper](https://arxiv.org/abs/2003.00744) — phần thí nghiệm: fine-tune trên các task tiếng Việt.
- [Hugging Face: Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training).

**Bài tập (VFI):**
1. `[OWNER-MUST]` Tính ước lượng back-of-envelope (giấy): UIT-VSFC train
   ~11k câu (16k tổng), PhoBERT-base 135M tham số, so với pretrain trên
   hàng chục GB văn bản — viết ra tỷ lệ dữ liệu và kết luận "ai học cái gì
   ở giai đoạn nào".
2. *(sau khi Phase 2 được duyệt)* `[OWNER-MUST]` Chạy 2 thí nghiệm nhỏ trên
   CÙNG split: (a) full fine-tune; (b) freeze encoder + chỉ train đầu phân
   loại. So validation macro-F1 + thời gian huấn luyện. `[AGENT-MAY]` Agent
   viết cấu hình cờ `--freeze-encoder`; owner quyết định có đưa (b) vào báo
   cáo chính thức hay chỉ để học.

**Effort:** ~2 giờ.

**Tự kiểm tra:** Trong 3 tình huống, chọn fine-tune hay train từ đầu: (1)
1 triệu câu có nhãn; (2) 16k câu có nhãn; (3) 500 câu có nhãn? *(Đạt khi có
lý do theo dữ liệu + chi phí, không có đáp án tuyệt đối — quan trọng là
lý do.)*

### 10. Overfitting & early stopping

**Mục tiêu:** chuyển dấu hiệu overfit từ Phase 1 (so train/val report của
các trial TF-IDF) sang cơ chế chủ động: theo dõi validation loss mỗi epoch,
dừng khi xấu đi `patience` lần, load lại checkpoint tốt nhất.

**Đọc tối thiểu:**
- Đọc lại **mục 4** `docs/learning-notes.md`.
- [Hugging Face: Callbacks — EarlyStoppingCallback](https://huggingface.co/docs/transformers/main_classes/callback) — điều kiện đi kèm: `load_best_model_at_end=True`, `metric_for_best_model`, `eval_strategy` (trước đây là `evaluation_strategy` — kiểm tra đúng tên tham số theo phiên bản transformers đang dùng).

**Bài tập (VFI):** *(sau khi Phase 2 được duyệt)*
1. `[OWNER-MUST]` Fine-tune 6 epoch KHÔNG early stopping, ghi bảng
   train loss / validation macro-F1 từng epoch — chỉ ra epoch nào bắt đầu
   overfit trên dữ liệu thật của bạn.
2. `[OWNER-MUST]` Bật `EarlyStoppingCallback(patience=2)`, chạy lại — đối
   chiếu epoch dừng với bảng ở bài 1. Owner **tự chọn** patience và viết
   một câu lý do vào training report.
3. `[AGENT-MAY]` Agent viết script vẽ curve train/val; owner diễn giải curve
   và chốt số epoch lý tưởng.

**Effort:** ~2 giờ.

**Tự kiểm tra:** patience=2, `metric_for_best_model="macro_f1"`: nếu validation
macro-F1 đạt đỉnh ở epoch 3 rồi giảm ở epoch 4, 5 — early stopping dừng ở
epoch nào và checkpoint nào được load? *(Đạt khi trả lời: dừng sau epoch 5
(2 lần xấu liên tiếp), load checkpoint epoch 3.)*

### 11. Data leakage (ôn)

**Mục tiêu:** giữ vững kỷ luật Phase 1 khi chuyển sang deep learning: fit
tokenizer/normalization theo đúng nguyên tắc "chỉ học từ train" khi cần;
nhận diện leakage dạng near-duplicate mà `find_leakage` (exact-match) chưa
bắt được.

**Đọc tối thiểu:**
- Đọc lại **mục 3, 7** `docs/learning-notes.md` và `docs/experiment-protocol.md` (split discipline).
- [Leakage and the Reproducibility Crisis in ML-based Science, Kapoor & Narayanan](https://arxiv.org/abs/2207.07048) — đọc phần phân loại các dạng leakage (bảng tổng hợp).

**Bài tập (VFI):**
1. `[OWNER-MUST]` Trên dữ liệu **tổng hợp tự tạo** (không commit, không phải
   dữ liệu thật): cố tình fit vectorizer trên toàn bộ dữ liệu rồi mới chia,
   so validation macro-F1 với pipeline đúng chuẩn — đo "độ lạc quan giả" mà
   leakage tạo ra trên chính tay bạn.
2. `[OWNER-MUST]` Viết 5 câu: với feedback sinh viên ngắn và lặp mẫu ("thầy
   dạy hay", "thầy dạy hay!"), vì sao near-duplicate leakage là rủi ro thật
   của UIT-VSFC mà `find_leakage` hiện tại không phát hiện? Đề xuất (chỉ viết
   ý tưởng, chưa code) một kiểm tra gần-trùng hiệu quả chi phí thấp.

**Effort:** ~1.5 giờ.

**Tự kiểm tra:** Trong PhoBERT pipeline, bước nào có "học từ dữ liệu" và do
đó phải tuân theo quy tắc train-only? *(Đạt khi chỉ được: không chỉ tokenizer
(cố định từ pretrain — an toàn), mà mọi thống kê học trên corpus: chuẩn hóa
segmentation, đầu phân loại, và đặc biệt là nếu bạn tự thêm vectorizer phụ/
feature ngoài — chúng phải fit trên train.)*

---

## Phần E — Đánh giá và phân tích

### 12. Macro-F1 (recap)

**Mục tiêu:** không mất kỹ năng tính tay khi chuyển sang deep learning —
vẫn chọn mô hình theo validation macro-F1, `zero_division=0` khai báo rõ.

**Đọc tối thiểu:** đọc lại **mục 5, 6** `docs/learning-notes.md` và
`src/vfi/evaluation/metrics.py` (docstring giải thích zero_division=0).

**Bài tập (VFI):** `[OWNER-MUST]` Cho confusion matrix 3×3 tự đặt (có ô
bị nhầm), tính accuracy, precision/recall/F1 từng lớp và macro-F1 **trên
giấy**, rồi kiểm chứng bằng `compute_metrics` từ repo. Phải trùng khớp đến
chữ số thập phân thứ hai.

**Effort:** ~1 giờ.

**Tự kiểm tra:** Một lớp có support 0 ở y_pred (không bao giờ được dự đoán):
macro-F1 xử lý thế nào với `zero_division=0` và vì sao phải ghi rõ lựa chọn
này trong báo cáo? *(Đạt khi nói được: F1 lớp đó = 0, kéo trung bình macro
xuống — và việc khai báo giúp người đọc so sánh được, không đoán hành vi
ngầm.)*

### 13. Error analysis

**Mục tiêu:** nâng từ "đọc confusion matrix" (Phase 1) lên quy trình có kỷ
luật: trích mẫu sai → phân cụm lỗi → giả thuyết ngôn ngữ học → kiểm chứng
bằng thí nghiệm nhỏ → (tùy chọn) behavioral test kiểu CheckList.

**Đọc tối thiểu:**
- [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList, Ribeiro et al., ACL 2020](https://arxiv.org/abs/2005.04118) — đọc ý tưởng invariance/directional expectation (bài báo đoán giải Best Paper ACL 2020 — ý tưởng chính đọc được không cần GPU).

**Bài tập (VFI):**
1. *(sau khi có dữ liệu hợp pháp + model đã train)* `[OWNER-MUST]` Từ
   `reports/artifacts/*.json`, trích **15 mẫu sai đầu tiên theo confusion
   matrix** (ô nhiều nhất trước), tự viết từng dòng: đầu vào rút gọn, dự
   đoán vs nhãn thật, **giả thuyết của bạn** (phủ định, teencode, thiếu dấu,
   mỉa mai, ranh giới neutral...). KHÔNG để AI viết phần giả thuyết — đây là
   phần "chữ ký" của owner.
2. `[OWNER-MUST]` Chọn 1 giả thuyết kiểm chứng được: tạo mini behavioral
   test (5–10 câu tự viết): ví dụ bất biến "thêm/thay 'không' trước tính từ
   phải lật nhãn hoặc ít nhất giảm xác suất lớp cũ" — chạy trên model, ghi
   tỉ lệ vi phạm.
3. `[AGENT-MAY]` Agent viết hàm trích xuất mẫu sai từ JSON evaluation ra
   bảng CSV (mechanical); owner tự đọc và tự phân loại.

**Effort:** ~3 giờ.

**Tự kiểm tra:** Trình bày 3 mẫu sai bất kỳ trong 15 mẫu cho một người bạn —
bạn nói được giả thuyết + cách kiểm chứng mà không nhìn tài liệu? *(Đạt khi
làm được không nhìn giấy; chưa đạt = chưa "own" phần phân tích.)*

---

## Phần F — Tối ưu inference (mức khái niệm)

### 14. ONNX, quantization, distillation (concept-level)

**Mục tiêu:** biết tên và bản chất của 3 hướng giảm chi phí suy luận, đủ để
trả lời phỏng vấn "làm sao deploy model nặng": (1) ONNX — export đồ thị sang
runtime tối ưu, tách khỏi framework huấn luyện; (2) dynamic quantization —
trọng số float32 → int8, giảm bộ nhớ/độ trễ; (3) distillation — model nhỏ
học mềm (soft targets) từ model lớn.

**Đọc tối thiểu:**
- [ONNX Runtime docs — how it works](https://onnxruntime.ai/docs/).
- [PyTorch: Dynamic Quantization tutorial](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html).
- [Distilling the Knowledge in a Neural Network, Hinton et al. 2015](https://arxiv.org/abs/1503.02531) — abstract + ý tưởng soft targets.
- [DistilBERT, Sanh et al. 2019](https://arxiv.org/abs/1910.01108) — ví dụ thực tế: nhỏ hơn ~40%, nhanh hơn ~60%, giữ ~97% chất lượng (so BERT-base).

**Bài tập (VFI):** *(sau khi Phase 2 được duyệt)*
1. `[OWNER-MUST]` Dùng **chính `measure_latency` có sẵn** trong
   `src/vfi/evaluation/metrics.py` (protocol batch=1, 10 warmup, 100 timed,
   seed cố định) đo: (a) pipeline TF-IDF Phase 1 trên CPU; (b) PhoBERT
   PyTorch; (c) PhoBERT export ONNX Runtime. Bảng 3 dòng so sánh mean/p95
   — protocol ghi kèm trong JSON như Phase 1 đã làm. `[AGENT-MAY]` Agent
   viết script export ONNX; owner chạy đo và diễn giải con số.
2. `[OWNER-MUST]` Viết 10 câu: với ứng dụng VFI (dự đoán lẻ từng feedback),
   hướng nào đáng ưu tiên trước giữa quantization vs distillation, vì sao?
   (Không có đáp án đúng duy nhất — đánh giá theo lập trình: phần cứng hiện
   có, rủi ro giảm chất lượng, công sức chuyển đổi.)

**Effort:** ~2.5 giờ.

**Tự kiểm tra:** Dynamic quantization ảnh hưởng gì đến chất lượng dự đoán
và làm sao kiểm chứng bằng số, dùng protocol nào của repo? *(Đạt khi trả
lời: đo lại macro-F1 trên cùng split trước/sau + latency theo
`measure_latency`; chất lượng giảm chấp nhận được chỉ khi có số chứng minh.)*

---

## Checklist tiến độ (owner tự đánh dấu)

> Quy ước trung thực của `docs/learning-notes.md` vẫn áp dụng: hoàn thành là
> do **bạn** tự xác nhận. AI không tick hộ.

- [ ] 1. Cosine TF-IDF tự viết + (sau Phase 2) so PhoBERT
- [ ] 2. Ba con số thưa/dense + giải thích vector toàn 0
- [ ] 3. Đạo hàm tay w_11 + micrograd + demo zero_grad bug
- [ ] 4. CE tính tay 2 trường hợp + kiểm chứng 2 thư viện
- [ ] 5. Bảng SGD/Adam/AdamW + 5 câu giải thích
- [ ] 6. Vẽ schedule warmup + sweep LR (toy, sau đó fine-tune thật)
- [ ] 7. Attention 3-token tính tay + 5 câu self-attention vs bigram
- [ ] 8. Bảng đếm token 3 kiểu + chạy tokenizer thật (sau Phase 2)
- [ ] 9. Ước lượng giấy fine-tune vs scratch + (sau Phase 2) thí nghiệm freeze
- [ ] 10. Bảng 6 epoch + early stopping thật (sau Phase 2)
- [ ] 11. Thí nghiệm leakage cố tình (dữ liệu tổng hợp) + ý tưởng near-dup check
- [ ] 12. Macro-F1 tính tay trùng khớp `compute_metrics`
- [ ] 13. 15 mẫu sai + giả thuyết tự viết + 1 behavioral test
- [ ] 14. Bảng latency 3 cấu hình + 10 câu ưu tiên

---

**Cam kết:** lộ trình này là bản đề xuất học tập. Khi Phase 2 chính thức
được duyệt, các bài tập gắn nhãn *(sau khi Phase 2 được duyệt)* chuyển thành
nhiệm vụ có thật trong plan thực thi; trước đó, phần đọc + bài tập giấy vẫn
làm được ngay và không phụ thuộc dữ liệu.
