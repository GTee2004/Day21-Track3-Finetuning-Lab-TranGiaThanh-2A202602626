# Reflection

### 1. Điều gì làm bạn ngạc nhiên nhất?

Điều làm mình ngạc nhiên nhất là target tăng từ **0.765 lên 0.970**, nhưng regression lại giảm mạnh từ **0.7911 xuống 0.4556**. Fine-tune giúp model làm tốt hơn rõ rệt trên nhiệm vụ phân loại ticket, đồng thời làm suy giảm năng lực tổng quát. Kết quả này cho thấy chỉ nhìn target hoặc format là chưa đủ để quyết định một model có phù hợp để triển khai hay không.

### 2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?

Mình mất nhiều thời gian nhất ở NB4 vì phải huấn luyện ba run đối chứng `attn_only`, `wrong_lr` và `qlora`. Trong lượt chạy đầy đủ đầu tiên, riêng NB4 mất khoảng **21.6 phút**, lâu hơn NB3 vì có ba cấu hình cần kiểm tra. Ban đầu mình nghĩ huấn luyện cấu hình `correct` sẽ là phần tốn thời gian nhất, nhưng việc xây dựng các đối chứng công bằng mới là phần mất nhiều thời gian hơn.

### 3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?

Trước lab này, mình từng xem training loss thấp hơn là dấu hiệu model tốt hơn. Kết quả cho thấy `attn_only` có loss **0.5375**, thấp hơn `correct` khoảng **0.6248**, nhưng cả hai cùng đạt target **0.970** trên full eval. Vì vậy, training loss chỉ nên dùng để theo dõi quá trình học; việc lựa chọn model phải dựa thêm trên target, regression, format và latency.

### 4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?

Mình dùng AI assistant để đọc rubric, giải thích các chỉ số, kiểm tra ý nghĩa của `EVAL_LIMIT` và hỗ trợ tổ chức report. Điểm hạn chế là khi chỉ có log bị rút gọn, AI không thể khôi phục chính xác các dự đoán chưa được lưu và có thể suy đoán nếu câu hỏi không đủ cụ thể. Ngoài ra, lượt chạy `EVAL_LIMIT=8` ban đầu có thể trông như đã hoàn thành nếu chỉ nhìn kết quả model; cần đối chiếu Gatekeeper mới phát hiện đó chỉ là smoke mode. Vì vậy, mình luôn kiểm tra lại câu trả lời của AI bằng `results/`, `runs.csv`, `verdict.json` và thông báo của Gatekeeper.

### 5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?

Mình sẽ xác định và đóng băng **eval set cùng benchmark trước khi training**. Benchmark cần bao gồm target, regression, format và latency, đồng thời eval set phải được kiểm tra trùng lặp với dữ liệu train. Sau đó mình sẽ đo base model với prompt tối ưu để xác định fine-tuning có thật sự cần thiết và model sau fine-tune phải vượt qua mốc nào.
