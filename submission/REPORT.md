# Lab 21 — Evaluation Report

**Họ tên**: Trần Gia Thành  **MSSV**: 2A202602626  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 14.6 GB`

> Mọi con số dưới đây được lấy từ file trong `results/` và đã được kiểm tra chéo bằng
> `scripts/verify.py`.

---

## 1. Setup

Tôi sử dụng model và bộ dữ liệu mặc định để tập trung vào việc kiểm tra loss mask, thiết
kế đối chứng công bằng và đánh giá LoRA so với base model được prompt tối ưu. Bài toán là
phân loại ticket chăm sóc khách hàng tiếng Việt thành JSON gồm bốn trường `intent`,
`urgency`, `product` và `sentiment`. Model Qwen3.5-4B phù hợp với GPU T4 và cho phép chạy
cả LoRA fp16 lẫn đối chứng QLoRA.

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage bốn trường |
| Train / val | 225 / 25 (seed 42) |
| Tập đánh giá | 50 target / 15 regression, full eval |
| `max_length` | 1024 — p95 đo được là 98, gợi ý là 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

Tôi giữ `max_length=1024` theo cấu hình tier T4 để tránh cắt chuỗi và giữ cùng điều kiện
cho các run. Tuy nhiên, chuỗi dài nhất chỉ có 101 token nên 256 sẽ là lựa chọn tiết kiệm
tài nguyên hơn nếu tối ưu lại pipeline cho corpus này.

**Template có giữ khối `<think>` không?** Có. `template_check.json` ghi nhận
`open_tag_present=true`, `body_present=true` và verdict là `reasoning preserved — safe to
train on traces`. Corpus mặc định dùng câu trả lời JSON trần, nên `valid_trace_rate=0`
không có nghĩa chat template làm mất reasoning trace.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Ba dòng đầu của đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Mask `assistant-only` giám sát 39/94 token, tương đương 0.4149. Đối chứng `everything`
giám sát toàn bộ 94/94 token, bao gồm system prompt và câu hỏi. Hai assertion trong
`mask_proof.json` chứng minh câu trả lời được đưa vào loss còn câu hỏi đã bị mask, đúng
với mục tiêu huấn luyện model sinh JSON thay vì học lặp lại input.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

NB2 được chạy và đóng băng trước khi huấn luyện, trên đủ 50 mẫu target và 15 mẫu
regression. SHA của prompt tối ưu là `719e74d3b6232053`.

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3249.5 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 997.5 |
| (c) LoRA fine-tune | 0.9700 | 0.4556 | 1.0000 | 1414.9 |

**(b) có thật sự mạnh hơn (a) không?** Có. Target tăng từ 0 lên 0.765, format tăng từ 0
lên 1.0 và latency giảm từ 3249.5 xuống 997.5 ms/mẫu, trong khi regression giữ nguyên ở
0.7911. Điều này cho thấy prompt engineering đã tạo ra một baseline mạnh trước khi cần
fine-tune.

Tôi không sửa `OPTIMIZED_PROMPT`. Giữ nguyên prompt (b) giúp phép so sánh công bằng và
tránh làm yếu baseline sau khi đã nhìn thấy kết quả fine-tune. Bản LoRA tăng target thêm
0.205 nhưng latency tăng 417.4 ms/mẫu và regression giảm mạnh, nên không thể chỉ nhìn
target để quyết định triển khai.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6248 | 0.970 | 399.2 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5375 | 0.970 | 263.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 393.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 468.1 | 3.86 |

Tất cả run dùng cùng 30 optimizer steps. `attn_only` chỉ lệch 8,192 tham số so với
`correct`, khoảng 0.025%, nên đáp ứng yêu cầu khớp ngân sách dưới 5%. Do NB3 được chạy
lại, `runs.csv` còn một dòng `correct` cũ có loss 0.6256 và thời gian 403.0 giây; bảng
trên dùng dòng mới nhất 0.6248 và 399.2 giây, tương ứng với adapter được NB5 đánh giá.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Trên full target, `attn_only` hòa `correct` ở điểm 0.970. Thứ tự này không giống kết quả
theo training loss: `attn_only` đạt loss 0.5375, thấp hơn 0.6248 của `correct`, nhưng
không đạt target cao hơn. Kết quả cho thấy tăng rank lên 283 trong vị trí q,v hẹp có thể
khớp ngân sách tham số nhưng không tự động tạo lợi thế so với gắn adapter rộng ở rank 16.
Vì vậy, không thể dùng training loss để tuyên bố `attn_only` thắng.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` giảm learning rate từ 1e-4 xuống 1e-5. Loss của nó giảm chậm và kết thúc ở
1.5702, cao hơn nhiều so với 0.6248 của `correct`; target và format đều bằng 0. Nếu không
biết LR, tôi có thể kết luận sai rằng dữ liệu, mask hoặc vị trí adapter có vấn đề. Đối
chứng một biến cho thấy LR theo thang full fine-tuning quá nhỏ đối với LoRA trong ngân
sách 30 step, khiến model chưa học đủ nhiệm vụ.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

QLoRA giảm peak VRAM từ 8.78 GB xuống 3.86 GB, tiết kiệm 4.92 GB, tương đương khoảng
56.0%. Đổi lại, thời gian train tăng từ 399.2 lên 468.1 giây, loss tăng từ 0.6248 lên
0.7058 và target giảm từ 0.970 xuống 0.940. QLoRA vẫn giữ format 1.0 nên cấu hình không
bị hỏng, nhưng nó chậm hơn và kém hơn trên target trong khi LoRA fp16 vẫn vừa T4. Kết quả
ủng hộ việc ưu tiên LoRA fp16 khi đủ VRAM; QLoRA chỉ phù hợp khi giới hạn bộ nhớ quan
trọng hơn mức giảm chất lượng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`

`target Δ = +0.205` · `regression Δ = -0.336` · `valid_trace_rate = 0.00`

Fine-tune tăng target từ 0.765 lên 0.970 và giữ format 1.0, cho thấy model học tốt nhiệm
vụ phân loại ticket. Tuy nhiên, regression giảm từ 0.7911 xuống 0.4556, tức giảm khoảng
0.336, vượt xa ngưỡng cho phép 0.020. Vì vậy, model không vượt qua cổng hồi quy và chưa
phù hợp để deploy. Kết quả này cho thấy model đã chuyên môn hóa mạnh vào JSON triage
nhưng đánh đổi quá nhiều năng lực tổng quát. Nguyên nhân hợp lý là 225 mẫu train đều
thuộc miền hẹp và không có replay data tổng quát. Verdict FAILED không có nghĩa phép huấn
luyện vô ích; ngược lại, hệ thống đánh giá đã phát hiện một đánh đổi mà target accuracy
đơn lẻ sẽ che giấu. Hướng cải thiện là thêm 1–5% replay data, giữ nguyên protocol và đánh
giá lại cả target, regression, format lẫn latency.

---

## 6. Định tính — bắt buộc có cả ca THUA

`qualitative.json` lưu prediction và điểm của fine-tune, còn NB2 chỉ lưu điểm tổng hợp của
baseline (b), không lưu prediction theo từng mẫu. Vì vậy, tôi không tự dựng lại output
baseline chưa được ghi nhận.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Shipper không gọi; chỉ hỏi; shop hỗ trợ tốt | `van_chuyen`, `thap`, ốp lưng điện thoại, `tich_cuc` | Không lưu từng mẫu | Đúng 4/4 trường | ✅ FT đúng |
| 2 | Hỏi giá ốp lưng; mong phản hồi | `hoi_thong_tin`, `trung_binh`, ốp lưng điện thoại, `trung_tinh` | Không lưu từng mẫu | Đúng 4/4 trường | ✅ FT đúng |
| 3 | Bình giữ nhiệt chưa thấy tiền; khi nào tiện | `hoan_tien`, `thap`, bình giữ nhiệt, `tich_cuc` | Không lưu từng mẫu | 3/4; urgency=`trung_binh` | ❌ **FT thua** |
| 4 | Nồi chiên thiếu phụ kiện; khi nào tiện | `san_pham_loi`, `thap`, nồi chiên không dầu, `trung_tinh` | Không lưu từng mẫu | 3/4; urgency=`trung_binh` | ❌ **FT thua** |
| 5 | Áo khoác bị lỗi; khi nào tiện | `san_pham_loi`, `thap`, áo khoác gió, `tich_cuc` | Không lưu từng mẫu | 3/4; urgency=`trung_binh` | ❌ **FT thua** |

Các ca fine-tune thua có cùng mẫu lỗi: ticket chứa tín hiệu urgency thấp “khi nào tiện”,
nhưng model đều dự đoán `trung_binh`. Intent và product vẫn đúng, nên lỗi tập trung ở
ranh giới nhãn urgency thay vì trải đều trên bốn trường. Điều này gợi ý cần kiểm tra phân
bố và bổ sung cách diễn đạt của nhóm urgency thấp, thay vì chỉ tăng epoch.

---

## 7. Kết luận & điều tôi học được

Nhìn chung, bản fine-tune đạt kết quả rất tốt trên task chính nhưng chưa đủ an toàn để deploy. Model đạt target 0.970 và
format 1.0, cao hơn rõ rệt so với base model dùng prompt tối ưu, nhưng regression giảm từ
0.7911 xuống 0.4556 và latency tăng từ 997.5 lên 1414.9 ms/mẫu. Đây là đánh đổi không
chấp nhận được nếu hệ thống vẫn cần xử lý yêu cầu ngoài miền triage. Mask là điều kiện
đúng đắn đầu tiên: nếu câu hỏi cũng nằm trong loss thì mọi kết quả phía sau mất ý nghĩa.
Sau khi mask đúng, learning rate là đòn bẩy thể hiện rõ nhất, vì giảm LR mười lần làm
target và format về 0. Vị trí adapter và rank phải được so sánh dưới cùng ngân sách;
`attn_only` có loss thấp hơn nhưng chỉ hòa `correct` trên target, nên loss không thể thay
thế task metric. QLoRA tiết kiệm khoảng 56% VRAM nhưng chậm hơn và giảm target, nên không
cần thiết khi LoRA fp16 vẫn vừa T4. Trước khi triển khai, tôi cần bổ sung replay data tổng
quát và thêm các cách diễn đạt urgency thấp, sau đó chạy lại full eval để xác nhận target
vẫn cao mà regression nằm trong ngưỡng.

**Ba điều tôi học được** (cụ thể, không generic):

1. Tôi phải giải mã ngược vùng token được giám sát để chứng minh mask, không thể chỉ tin vào tên `assistant-only`.
2. Prompt tối ưu đã nâng target từ 0 lên 0.765 mà không cần train, nên fine-tune phải được so với một baseline prompt mạnh.
3. Loss thấp hơn không bảo đảm task metric cao hơn; `attn_only` có loss thấp hơn nhưng chỉ hòa `correct` ở target 0.970.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** thêm 1–5% replay data tổng quát, bổ sung các mẫu
urgency thấp như “khi nào tiện”, train lại cùng 30 step và đánh giá đủ 50 target cùng 15
regression. Tôi chỉ cân nhắc deploy nếu target vẫn cao, regression nằm trong ngưỡng và
kết quả không phụ thuộc vào một nhóm cách diễn đạt hẹp.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
