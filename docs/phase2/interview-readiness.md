# Chuẩn bị phỏng vấn thực tập dựa trên dự án VFI (Interview Readiness)

> Viết ngày 2026-11-07 trên nhánh `research/phase2-readiness`. Mục tiêu: bạn
> trả lời được câu hỏi phỏng vấn **bằng trải nghiệm thật của dự án này** —
> không học thuộc đáp án chung chung, không nhờ AI trả lời hộ.
>
> Trạng thái trung thực cần nhớ khi phỏng vấn: dự án **đã có** pipeline
> classical ML hoàn chỉnh, test, CI, protocol — nhưng **chưa có số liệu thật**
> (dataset UIT-VSFC chờ ký thỏa thuận). Tuyên bố đúng trạng thái này là điểm
> cộng về trung thực, không phải điểm trừ. Đừng nói "đã train được model X%"
> khi chưa có.
>
> Cách dùng: mỗi câu hỏi có (1) phần **câu trả lời tốt cần bao phủ**, (2)
> phần **câu trả lời sai phổ biến** để bạn tự rà. Luyện bằng cách tự nói
> thành tiếng, quay âm thanh, nghe lại — không đọc màn hình.

---

## Phần A — Câu hỏi về pipeline và dữ liệu

### Q1. "Kể toàn bộ pipeline của dự án từ dữ liệu đến dự đoán."

**Câu trả lời tốt cần bao phủ:** dữ liệu UIT-VSFC (feedback sinh viên, 3
lớp positive/neutral/negative) được tải bằng `vfi/data/loading.py` — chuẩn
hóa NFC + gộp khoảng trắng (không lowercase, không bỏ dấu vì mất tín hiệu
tiếng Việt); kiểm tra trùng lặp và rò rỉ chéo split trong
`vfi/data/validation.py` (`find_leakage`); `vfi/data/splitting.py` cho fallback
split phân tầng có seed; model: `MajorityBaseline` và `Pipeline` TF-IDF →
LogisticRegression (`models/baseline.py`), chọn siêu tham số trên validation
(`run_small_search`, 4 ứng viên); metric macro-F1 (`evaluation/metrics.py`);
artifact: `model.joblib` + `meta.json` (provenance); CLI 4 lệnh:
`prepare-data | train | evaluate | predict`.

**Câu trả lời sai phổ biến:** kể kiểu "có AI giúp tôi code" hoặc chỉ nói
"TF-IDF rồi logistic regression" mà không nhắc split discipline và artifact
metadata — nghĩa là bạn kể được công cụ nhưng không kể được **hệ thống**.

### Q2. "Vì sao chọn TF-IDF + Logistic Regression làm baseline đầu tiên thay vì chạy thẳng PhoBERT?"

**Câu trả lời tốt cần bao phủ:** (1) baseline nhanh, chạy CPU vài giây, dễ
debug toàn pipeline trước khi đắt đỏ; (2) MajorityBaseline + TF-IDF-LR là
**sàn tham chiếu** — PhoBERT chỉ có nghĩa nếu thắng được sàn này trên cùng
split/metric; (3) trọng số LR đọc được (`w["không tốt"]` âm) → explainability
đơn giản; (4) kỷ luật quy trình (test, protocol, artifact) phải đúng trước
khi thêm độ phức tạp; "PhoBERT không cứu được pipeline bẩn".

**Câu trả lời sai phổ biến:** "PhoBERT mạnh hơn nên cứ dùng PhoBERT" — người
phỏng vấn sẽ hỏi tiếp "mạnh hơn là so với cái gì?" và bạn phải có baseline
để trả lời.

### Q3. "Pipeline của bạn chống data leakage bằng cách nào?"

**Câu trả lời tốt cần bao phủ:** từ điển TF-IDF học **trong** `pipeline.fit`
chỉ trên train; validation/test chỉ được `transform`. Có test bảo vệ riêng:
`test_vocabulary_learned_from_training_data_only`. `find_leakage` quét trùng
chính xác sau chuẩn hóa chung giữa các split, `vfi prepare-data` báo cáo số
lượng. Và **nói ra giới hạn thật**: phát hiện là exact-match sau chuẩn hóa —
near-duplicate chưa bắt được (known limitation trong spec).

**Câu trả lời sai phổ biến:** "Có kiểm tra trùng lặp" — không nói được kiểm
tra nằm ở đâu, bắt dạng nào, bỏ sót dạng nào. Người phỏng vấn kỹ sẽ đào tiếp
"trùng gần giống thì sao?" — chính là dịp nói known limitation.

### Q4. "Vì sao không lowercase và không bỏ dấu tiếng Việt khi chuẩn hóa?"

**Câu trả lời tốt cần bao phủ:** NFC để hợp nhất kiểu gõ dấu khác nhau
(Unicode composed); giữ nguyên chữ hoa/thường và dấu thanh vì chúng mang tín
hiệu: phủ định ("không"), ngữ điệu, emphasis; lowercase chỉ làm bên trong
`TfidfVectorizer` (tham số `lowercase=True`), tức là một quyết định được ghi
vào cấu hình model, không phải mất mát âm thầm ở tầng dữ liệu; bỏ dấu sẽ
gộp "hỏi" với "hợ" — mất nghĩa.

**Câu trả lời sai phổ biến:** "Chuẩn hóa cho sạch dữ liệu" rồi lowercase +
bỏ dấu + xóa stopword tất cả — với tiếng Việt điều đó phá tín hiệu và là
câu trả lời của người chưa xử lý dữ liệu tiếng Việt thật.

### Q5. "Nhãn của dataset gốc là số nguyên — bạn xử lý mapping như thế nào?"

**Câu trả lời tốt cần bao phủ:** loader Phase 1 đang hard-code mapping
`0=positive, 1=negative, 2=neutral` (theo khai báo `ClassLabel` của loader
SEACrowd) — nguồn này **chưa được xác minh** và mâu thuẫn với card của chính
nhóm tác giả; cân bằng chứng nghiêng về `0=negative` làm *planning prior*,
nhưng trạng thái là **UNRESOLVED** và gate G2 quyết định (xem
`label-semantics-evidence.md`). VFI chuyển ngay khi tải sang không gian
canonical một chỗ duy nhất: `negative=0, neutral=1, positive=2`
(`CANONICAL_LABELS`, `RAW_SENTIMENT_TO_LABEL` trong `vfi/data/loading.py`)
— toàn bộ code sau đó không phải biết mapping gốc. Và điểm cộng lớn: khi có
dữ liệu thật, **kiểm tra tay vài dòng với README của tác giả** trước khi tin
mapping — đây là caveat đã ghi trong `docs/dataset-uit-vsfc.md`.

**Câu trả lời sai phổ biến:** "Có mapping sẵn trong code" — không nhớ chiều
mapping gốc, không biết điểm rủi ro là một nhãn lệch 1 bước làm toàn bộ kết
quả vô nghĩa.

### Q6. "Dữ liệu của bạn bị lệch lớp — pipeline đối phó ra sao?"

**Câu trả lời tốt cần bao phủ:** `class_weight="balanced"` cho LR; chọn mô
hình theo **macro-F1** (lớp hiếm có trọng số ngang); `MajorityBaseline` tồn
tại để phơi bày độ lệch (nếu accuracy của model thật chỉ ngang majority thì
model vô dụng); báo cáo per-class precision/recall/F1/support + confusion
matrix 3×3 để thấy lớp nào đang bị bỏ rơi.

**Câu trả lời sai phổ biến:** "Tôi sẽ oversampling/SMOTE" như câu trả lời
đầu tiên mà không nhắc metric và baseline — triệu chứng của người học thuộc
mẹo hơn là hiểu bài toán.

---

## Phần B — Câu hỏi về metric và đánh giá

### Q7. "Vì sao chọn macro-F1 làm metric chính thay vì accuracy?"

**Câu trả lời tốt cần bao phủ:** accuracy bị majority class che khuất (lớp
chiếm 80% → đoán mò được 80%); F1 kết hợp precision + recall; **macro** lấy
trung bình F1 từng lớp **không theo support** → neutral hiếm vẫn ngang positive;
kết nối với dự án: majority baseline + macro-F1 là cặp bài tròng để phát
hiện mô hình "lười". Có thể nói thêm: micro-F1 sẽ thiên về lớp lớn nên không
dùng làm metric chọn mô hình.

**Câu trả lời sai phổ biến:** "Macro-F1 tổng quát hơn" — câu rỗng. Phải nói
được precision/recall của MỘT lớp là gì và trung bình macro khác micro ra
sao.

### Q8. "zero_division=0 trong metric của bạn là gì? Bỏ nó đi thì sao?"

**Câu trả lời tốt cần bao phủ:** khi một lớp không có mẫu thật và cũng không
có mẫu dự đoán, precision/recall không xác định (0/0); sklearn mặc định
`warn` — phát cảnh báo và trả 0.0; dự án **khai báo tường minh**
`zero_division=0` trong `compute_metrics` để hành vi cố định, có tài liệu,
không phụ thuộc cảnh báo runtime; đồng thời `labels=` bị ép đủ 3 lớp canonical
để ma trận luôn 3×3.

**Câu trả lời sai phổ biến:** "Thư viện tự lo" — người phỏng vấn đánh giá
đây là người không biết hành vi mặc định và không có văn hóa khai báo quyết
định.

### Q9. "Confusion matrix của bạn đọc thế nào? Bạn kỳ vọng mẫu lỗi kiểu gì?"

**Câu trả lời tốt cần bao phủ:** hàng = nhãn thật, cột = nhãn dự đoán, thứ
tự canonical negative/neutral/positive; đọc theo hàng để biết lớp thật bị
đem đi đâu. Kỳ vọng (giả thuyết, chưa xác nhận vì chưa có dữ liệu thật):
negative→neutral nhiều (mô hình "né" phán đoán tiêu cực), positive↔neutral
ranh giới mờ, lỗi tập trung vào câu phủ định dạng "không tệ". Câu trả lời
phải nói rõ **đây là giả thuyết chờ error analysis**, chưa phải kết luận.

**Câu trả lời sai phổ biến:** đọc cột làm hàng, hoặc khẳng định chắc chắn
các mẫu lỗi như đã đo được — khi chưa có một run thật nào.

### Q10. "Latency của model được đo thế nào? Vì sao phức tạp thế?"

**Câu trả lời tốt cần bao phủ:** protocol trong `measure_latency`: batch
size 1; 10 lần warmup không tính (khớp JIT/cache/OS page); 100 lần tính giờ
bằng `time.perf_counter`, mỗi lần một câu lấy ngẫu nhiên có seed; báo
mean/median/p95 ms; **toàn bộ tham số protocol được ghi vào JSON kết quả**
cùng con số — vì số không protocol thì không tái lập và không so sánh được.
Liên hệ Phase 2: đo PhoBERT bằng đúng protocol này để so sánh công bằng với
TF-IDF.

**Câu trả lời sai phổ biến:** "Chạy timeit lấy số trung bình" — không warmup,
không p95 (tail latency quan trọng cho serving), không lưu protocol.

### Q11. "Nếu hai model có validation macro-F1 bằng nhau, bạn chọn thế nào?"

**Câu trả lời tốt cần bao phủ:** quy tắc có sẵn trong repo: tie-break về
ứng viên **đầu tiên** trong `SEARCH_SPACE` (ưu tiên cấu hình đơn giản hơn/được
liệt kê trước); ngoài đó có thể thêm tiêu chí phụ: fit time, độ phức tạp,
explainability — nhưng quyết định phải **viết ra** trong training report,
không chọn ngầm.

**Câu trả lời sai phổ biến:** "Chọn ngẫu nhiên" hoặc "chạy lại nhiều lần đến
khi phân định" — kiểu thứ hai là dùng test/validation đến mức mất ý nghĩa.

---

## Phần C — Câu hỏi về tái lập và artifact

### Q12. "`meta.json` của bạn chứa gì và chứa để làm gì?"

**Câu trả lời tốt cần bao phủ:** provenance của từng artifact:
`created_at`, config đầy đủ + `config_hash`, `dataset_fingerprints` (sha256
từng split), `sklearn_version`, `python_version`, `seed`. Mục đích: (1) tái
lập thí nghiệm; (2) chính sách **trusted artifact** — `load_model` kiểm tra
sidecar trước khi deserialize: đúng model_id, có config_hash, cảnh báo khi
sklearn version lệch major (predictions có thể đổi → retrain thay vì tin).

**Câu trả lời sai phổ biến:** "Lưu model xong có file metadata" — không nói
được load phải kiểm tra gì và vì sao chính sách đó tồn tại (joblib deserialize
từ nguồn không tin cậy là rủi ro bảo mật + version drift).

### Q13. "Làm sao bảo đảm hai lần chạy cho cùng kết quả?"

**Câu trả lời tốt cần bao phủ:** seed cố định ở config (dùng cho fallback
split, classifier, latency RNG); fingerprint dữ liệu theo sha256; pin version
thư viện trong metadata; config_hash khớp cấu hình; quy trình ghi trong
`docs/experiment-protocol.md`. Thừa nhận thật: khi lên PhoBERT (Phase 2),
fine-tune GPU có biến động seed — kế hoạch là chạy nhiều seed và báo mean ±
std, thay vì một con số đơn.

**Câu trả lời sai phổ biến:** "Set random_state=42" — chỉ đúng một phần;
người phỏng vấn muốn thấy bạn hiểu reproducibility là **hệ thống** (dữ liệu +
code + môi trường), không phải một tham số.

### Q14. "CI của bạn kiểm tra gì? Test có chứng minh model tốt không?"

**Câu trả lời tốt cần bao phủ:** CI chạy ruff + pytest (CPU, synthetic
fixtures): test loading/mapping nhãn, validation/leakage, splitting, metrics
(so với ví dụ tính tay), baseline, CLI. Và câu trả lời quan trọng: test
đơn vị **không đo chất lượng model** — chúng bảo vệ đúng đắn của quy trình;
chất lượng chỉ đến từ run thật trên dữ liệu thật. `scripts/verify.py` là
cổng lint+test+kiểm tra cấu trúc trước khi kết thúc mỗi thay đổi.

**Câu trả lời sai phổ biến:** "Test pass nên model hoạt động tốt" — nhầm
lẫn kinh điển giữa correctness của code và quality của model; đây là câu
phân biệt ứng viên kỹ.

---

## Phần D — Câu hỏi về Phase 2 (PhoBERT)

### Q15. "Vì sao fine-tune PhoBERT thay vì train một model từ đầu cho tiếng Việt?"

**Câu trả lời tốt cần bao phủ:** pretrain cần kho văn bản hàng chục GB +
GPU hàng tuần — ngoài tầm dự án; PhoBERT đã pretrain trên ~20GB văn bản
tiếng Việt đã word-segment (PhoBERT-base ≈ 135M tham số); với ~16k câu
UIT-VSFC, fine-tune tận dụng ngôn ngữ học có sẵn, chỉ điều chỉnh; phương án
trung gian nếu thiếu GPU: freeze encoder + train đầu phân loại. Nguồn:
[PhoBERT paper](https://arxiv.org/abs/2003.00744), [model card](https://huggingface.co/vinai/phobert-base).

**Câu trả lời sai phổ biến:** "Fine-tune nhanh hơn" — thiếu chiều chi phí/dữ
liệu; người phỏng vấn muốn nghe trade-off cụ thể (dữ liệu 16k so với pretrain
corpus là tỷ lệ không tưởng).

### Q16. "PhoBERT cần word segmentation tiếng Việt — giải thích vì sao, và dùng gì để segment?"

**Câu trả lời tốt cần bao phủ:** tiếng Việt "từ" gồm nhiều âm tiết
("học_phí", "giảng_viên"); PhoBERT pretrain trên corpus **đã segment**
(VnCoreNLP), tokenizer BPE vocab 64K subword học trên đơn vị đã segment → input
phải khớp phân phối đó, nối âm tiết bằng `_`; công cụ: VnCoreNLP hoặc
underthesea; nếu quên segment, model không lỗi nhưng chất lượng rơi vì âm
tiết tách rời thành token vô nghĩa. Đối lập: Phase 1 TF-IDF không cần segment
vì n-gram tự học từ corpus (bigram "học phí" tự hình thành trong từ điển).

**Câu trả lời sai phổ biến:** "PhoBERT là BERT nên tự xử lý được tiếng
Việt" / nhầm WordPiece (BERT) với BPE (PhoBERT) mà không nói được bản chất
subword là chung.

### Q17. "Bạn chọn learning rate, số epoch, batch size thế nào khi fine-tune trên GPU sinh viên?"

**Câu trả lời tốt cần bao phủ:** bắt đầu từ khuyến nghị paper BERT: LR ∈
{2e-5, 3e-5, 5e-5}, warmup ~10% bước, linear decay, batch 16/32, 2–4 epoch
(trên 16k câu overfit rất nhanh); với GPU nhỏ (ví dụ T4 16GB): fp16 mixed
precision, batch nhỏ + gradient accumulation nếu cần VRAM; **đánh giá khả
thi GPU trước** (Colab/Kaggle có hạn mức) — roadmap dự án ghi rõ "không giả
định có GPU"; mọi lựa chọn cuối phải được quyết định bằng validation
macro-F1 của chính thí nghiệm, không copy số nghe qua tai.

**Câu trả lời sai phổ biến:** "LR 1e-3, 10 epoch" — hủy hoại pretrain;
hoặc đọc thuộc một bộ hyperparameter mà không biết chúng đến từ đâu và phải
tinh chỉnh theo dữ liệu mình.

### Q18. "Làm sao biết PhoBERT của bạn THẬT SỰ tốt hơn baseline TF-IDF?"

**Câu trả lời tốt cần bao phủ:** so trên **cùng** split, cùng metric macro-F1,
cùng protocol; test chỉ chạm sau khi chốt; chạy 2–3 seed cho fine-tune và
báo mean ± std (fine-tune biến động hơn TF-IDF); kèm chi phí (thời gian
train, latency inference, bộ nhớ) — "tốt hơn" có thể đúng về chất lượng nhưng
xấu về chi phí; error analysis so 2 model chọn sai ở đâu.

**Câu trả lời sai phổ biến:** so macro-F1 của PhoBERT (chạy 1 lần, seed may
mắn) với baseline cũ — không seed variance, không cùng protocol; hoặc so
với số trên báo cáo dataset paper (khác phân phối/preprocessing).

### Q19. "Fine-tune xong mà overfit ngay epoch đầu — bạn làm gì?"

**Câu trả lời tốt cần bao phủ:** chẩn đoán trước: so train loss vs validation
macro-F1 theo epoch; can thiệp theo thứ tự giá rẻ→đắt: giảm LR, tăng weight
decay (AdamW), giảm epoch + early stopping (patience từ `EarlyStoppingCallback`,
`load_best_model_at_end`), data augmentation có kiểm soát (nếu có thời gian),
và phương án freeze encoder. Mỗi can thiệp là một trial có ghi nhận, chọn
trên validation — không "cảm giác là nó ổn hơn".

**Câu trả lời sai phổ biến:** "Tăng dữ liệu" như phát súng đầu tiên (không
có dữ liệu thêm) hoặc chỉnh ngầm nhiều thứ cùng lúc rồi không biết cái gì
có tác dụng.

### Q20. "Nếu muốn deploy PhoBERT với ngân sách thấp, bạn nghĩ đến gì?"

**Câu trả lời tốt cần bao phủ:** lần lượt: đo baseline latency bằng protocol
có sẵn (`measure_latency`) → ONNX Runtime export (tách khỏi framework, tối
ưu đồ thị) → dynamic quantization int8 (nhỏ hơn, nhanh hơn, phải đo lại
macro-F1 để chứng minh chất lượng giữ được) → distillation là phương án nặng
hơn (train model nhỏ từ soft targets — DistilBERT là ví dụ kinh điển); và
luôn có phương án fallback: TF-IDF baseline của Phase 1 chạy CPU rẻ — chọn
theo bài toán feedback lẻ từng câu không cần GPU.

**Câu trả lời sai phổ biến:** "Dùng GPU cloud" — câu trả lời bỏ qua mọi
tối ưu phần mềm và chi phí; hoặc nói tên công nghệ mà không nói được nó
đổi cái gì ở mức số bit/kích thước/độ trễ.

---

## Phần E — 5 câu hỏi "phát hiện phụ thuộc AI" và cách chuẩn bị trung thực

Nhóm câu này được một số interviewer dùng để kiểm tra ứng viên có hiểu dự
án của chính mình hay không. Cách chuẩn bị duy nhất là **thật**: đã làm thì
kể được; chưa làm thì nói chưa làm và nói bạn đã chuẩn bị đến đâu.

### D1. "Mở code lên — chỉ cho tôi chỗ từ điển TF-IDF được fit, và sửa để nó fit trên toàn bộ dữ liệu; test nào sẽ đỏ?"

**Đây là thử thách gì:** kiểm tra bạn có thực sự làm việc với code hay chỉ
kể tên thư viện. Người phụ thuộc AI sẽ lúng túng khi điều hướng file thật.

**Chuẩn bị trung thực:** tự mở `src/vfi/models/baseline.py` + `tests/`,
tự theo dõi `pipeline.fit(x_train, y_train)`; biết `Pipeline` gộp vectorizer
và classifier nên fit ở đúng một chỗ; biết trước `test_vocabulary_learned_from_training_data_only`
sẽ đỏ. Luyện tay trên máy — không xem trước đáp án.

### D2. "Tại sao chọn zero_division=0? Hành vi mặc định của sklearn là gì?"

**Đây là thử thách gì:** phát hiện quyết định do AI đưa mà owner không tra
nguồn. Ai không tự đọc docs sẽ không biết mặc định là `warn` (trả 0.0 + cảnh
báo runtime), và không giải thích nổi vì sao phải khai báo tường minh thay
vì ăn theo mặc định.

**Chuẩn bị trung thực:** đọc docstring `src/vfi/evaluation/metrics.py` và
doc sklearn một lần bằng mắt mình; luyện nói 30 giây: mặc định `warn` →
0.0 kèm cảnh báo; dự án chọn tường minh để hành vi có tài liệu, ổn định
giữa các phiên bản.

### D3. "Không nhìn ghi chú — optimizer, LR, epoch của lần fine-tune gần nhất là gì, và vì sao là những số đó?"

**Đây là thử thách gì:** câu hỏi về trải nghiệm sống. Người thật sự chạy sẽ
nhớ chuyện mình vật lộn (diverge ở 1e-3, overfit ở epoch 2...). Người không
chạy sẽ nhảy vào con số lý tưởng của tutorial, không có chuyện rơi vãi.

**Chuẩn bị trung thực:** nếu Phase 2 chưa chạy: trả lời thẳng "chưa fine-tune
thật — tôi đang ở giai đoạn baseline classical; kế hoạch của tôi là X vì Y",
rồi cho thấy bạn hiểu lựa chọn (Q17). Nếu đã chạy: mở training report
trước giờ thi và **kể lại bằng lời** từng dòng con số + điều đã sửa.

### D4. "Con số p95 latency này — protocol đo là gì? Cho tôi xem đoạn code đo."

**Đây là thử thách gì:** phát hiện số liệu "từ AI" không có chuỗi sinh ra
(provenance). Người phụ thuộc AI thường báo số mà không nhớ protocol.

**Chuẩn bị trung thực:** thuộc lòng protocol Phase 1 (batch 1, 10 warmup,
100 timed, seed, perf_counter, mean/median/p95, protocol ghi kèm JSON);
mở `measure_latency` trong `src/vfi/evaluation/metrics.py` cho interviewer
xem; nếu chưa đo PhoBERT thì nói chưa đo và chỉ protocol sẽ dùng.

### D5. "Kể một lỗi thật bạn từng gặp trong dự án này và cách bạn debug."

**Đây là thử thách gì:** thiếu một câu chuyện rơi vãi thật là tín hiệu mạnh
nhất của dự án "AI làm hộ". Người thật làm sẽ có chuyện kể ngập ngừng, có
chi tiết sai, có mùi đời thật.

**Chuẩn bị trung thực — bạn đã có chuyện có thật trong lịch sử git (xác
minh bằng `git log` trước khi kể):**
1. **Gitignore chặn lint:** pattern `models/` chưa neo (`/models/`) từng làm
   `src/vfi/models/` bị ẩn khỏi ruff chạy local — lint trong CI lộ lỗi
   (UP017 `datetime.UTC`, vài message quá dài, annotation dạng string) và
   phải sửa + pin version ruff (commit `fix(lint): satisfy ruff 0.17 on
   models package...`). Bài học: công cụ bỏ qua "im lặng" một thư mục còn
   nguy hiểm hơn công cụ báo lỗi.
2. **Mapping nhãn lệch chiều:** phát hiện loader khai báo raw mapping
   `0=positive` (theo SEACrowd) trong khi trực giác là `0=negative` — và hai
   nguồn công khai thực ra mâu thuẫn nhau, còn UNRESOLVED (xem
   `label-semantics-evidence.md`); được xử lý bằng remap một chỗ có tài
   liệu + caveát "kiểm tra tay vài dòng với README tác giả trước khi tin".
3. **Blocker dataset:** không thể tự ý tải UIT-VSFC vì license phải ký thỏa
   thuận — cách xử lý là ghi verdict BLOCKED có bằng chứng nguồn, làm hết
   code trên synthetic fixtures, không bịa số.

Nếu kể chuyện nào, bạn phải tự đọc lại nguồn thật (commit, doc) trước —
kể sai chi tiết của chính dự án mình còn tệ hơn không có chuyện để kể.

---

## Phần F — Quy trình luyện tập đề xuất

1. **Vòng 1 (đọc):** với mỗi câu Q1–Q20, tự viết dàn ý 4–5 dòng vào sổ —
   KHÔNG copy từ tài liệu này; so lại sau.
2. **Vòng 2 (nói):** mỗi câu trả lời thành tiếng trong 60–90 giây; ghi âm;
   nghe lại và đánh dấu chỗ "ngập vì không hiểu" (khác chỗ "ngập vì thiếu
   từ").
3. **Vòng 3 (thực chiến):** nhờ một người bạn hỏi ngẫu nhiên theo thứ tự
   lộn xộn + hỏi sâu "vì sao" 2 tầng ("vì sao dùng macro-F1?" → "vì sao lớp
   hiếm phải ngang lớp lớn?").
4. **Câu trả lời khi chưa có số liệu thật:** mẫu chuẩn — "Pipeline và protocol
   đã xong và có test; chưa có kết quả thật vì dataset chờ thỏa thuận license;
   khi có data, quy trình của tôi là X." Trung thực + cho thấy đã sẵn sàng.

**Lời cam kết cuối:** không dùng AI để soạn đáp án khi luyện; AI chỉ được
hỏi ngược bạn (đóng vai interviewer). Đáp án phỏng vấn phải sống trong đầu
bạn — đó là toàn bộ ý nghĩa của tài liệu này.
