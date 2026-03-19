# Ad Image Detector

Script Python để detect vùng quảng cáo (`ad`) đơn sắc trong ảnh hoặc video theo batch mode.

Output của mỗi lần chạy gồm:

- thông tin detect trên terminal hoặc JSON
- ảnh annotate có khoanh vùng ad
- với video: các frame được trích ra từ video trước khi detect

## Workflow mới

Input được chia thành 2 folder cố định:

- `input/images`: chứa ảnh đầu vào
- `input/videos`: chứa video đầu vào

Output được ghi ra 2 folder tương ứng:

- `output/images`
- `output/videos`

Script không còn single mode. Bạn luôn chạy theo batch mode và chọn nguồn bằng tham số `--source`.

## Cấu trúc thư mục

- `detect_ad.py`: script chính
- `input/images`: ảnh đầu vào
- `input/videos`: video đầu vào
- `output/images`: ảnh annotate cho image mode
- `output/videos`: frame trích từ video và ảnh annotate cho video mode
- `README.md`: hướng dẫn sử dụng

## Yêu cầu

Cần Python 3 và các thư viện:

```bash
pip install opencv-python numpy
```

## Cách chạy

### 1. Chạy với ảnh

```bash
python detect_ad.py --source images
```

### 2. Chạy với video

Mặc định script sẽ chụp `2 giây / 1 frame`, rồi detect toàn bộ các frame đó.

```bash
python detect_ad.py --source videos
```

Nếu muốn đổi chu kỳ trích frame:

```bash
python detect_ad.py --source videos --frame-interval 1.5
```

### 3. Chạy với JSON output

```bash
python detect_ad.py --source images --json
python detect_ad.py --source videos --json
```

## Tham số

- `--source`: bắt buộc, chọn `images` hoặc `videos`
- `--color`: màu ad ở dạng hex, ví dụ `#FFA223` hoặc `FFA223`
- `--shape`: gợi ý shape cho ad
  - `auto`: tự suy luận
  - `l`: hình chữ L
  - `rectangle`: hình chữ nhật
  - `square`: hình vuông
- `--tolerance`: độ lệch màu tối đa khi tạo mask, mặc định là `20`
- `--frame-interval`: chỉ áp dụng cho `--source videos`, mặc định là `2.0` giây
- `--json`: in kết quả dưới dạng JSON

Các tham số cũ đã bị loại bỏ:

- `--input-dir`
- `--output-dir`
- `--image`
- `--output`

## Output khi chạy với ảnh

Command:

```bash
python detect_ad.py --source images
```

Script sẽ đọc toàn bộ ảnh trong `input/images` và ghi ảnh annotate sang `output/images`.

Ví dụ summary:

```text
Source: images
Input folder: input\images
Output folder: output\images
Processed: 4 | Success: 4 | Errors: 0
```

Mỗi ảnh output sẽ có:

- vùng ad được tô highlight
- contour khoanh đúng vùng ad
- label kích thước cạnh hoặc nhánh
- `% diện tích` hiển thị bên trong ad

Ví dụ file output:

- `output/images/sample1_detected.png`
- `output/images/sample2_detected.png`
- `output/images/sample3_detected.png`
- `output/images/test_detected.png`

## Output khi chạy với video

Command:

```bash
python detect_ad.py --source videos
```

Với mỗi video trong `input/videos`, script sẽ:

1. đọc video
2. trích frame theo chu kỳ `--frame-interval`
3. lưu frame gốc vào:
   - `output/videos/<ten_video>/frames`
4. detect từng frame
5. lưu ảnh annotate vào:
   - `output/videos/<ten_video>/detections`

Ví dụ cấu trúc output cho video `demo.mp4`:

```text
output/videos/demo/
  frames/
    demo_frame_0000_t00000_00s.png
    demo_frame_0001_t00002_00s.png
  detections/
    demo_frame_0000_t00000_00s_detected.png
    demo_frame_0001_t00002_00s_detected.png
```

Summary của video mode sẽ có thêm:

- `frame_interval_seconds`
- `total_extracted_frames`
- `total_detected_frames`
- `total_frame_errors`
- danh sách `videos`

Mỗi phần tử trong `videos` sẽ chứa:

- `video_path`
- `frames_dir`
- `detections_dir`
- `extracted_frame_count`
- `success_count`
- `error_count`
- `results`
- `errors`

## Output JSON

### Image mode

```bash
python detect_ad.py --source images --json
```

Output JSON gồm:

- `source`
- `input_dir`
- `output_dir`
- `processed_count`
- `success_count`
- `error_count`
- `results`
- `errors`

### Video mode

```bash
python detect_ad.py --source videos --json
```

Output JSON gồm:

- `source`
- `input_dir`
- `output_dir`
- `frame_interval_seconds`
- `processed_count`
- `success_count`
- `error_count`
- `total_extracted_frames`
- `total_detected_frames`
- `total_frame_errors`
- `videos`
- `errors`

## Kết quả hiện tại với ảnh trong repo

Chạy:

```bash
python detect_ad.py --source images
```

Kết quả hiện tại:

- `input/images/sample1.png`: `l`, màu `#FEA11E`, diện tích `40.156%`
- `input/images/sample2.png`: `l`, màu `#000000`, diện tích `47.826%`
- `input/images/sample3.png`: `rectangle`, màu `#BB281A`, diện tích `20.403%`
- `input/images/test.png`: `rectangle`, màu `#FFA327`, diện tích `18.388%`

## Cách script hoạt động

### Với ảnh

1. Đọc toàn bộ ảnh trong `input/images`.
2. Nếu không có `--color`, script lấy các màu nổi bật từ:
   - viền ảnh
   - toàn bộ ảnh
3. Tạo mask cho từng màu candidate dựa trên khoảng cách RGB.
4. Tách các connected component.
5. Chấm điểm từng component theo:
   - diện tích
   - độ giống `L`
   - độ giống `rectangle/square`
   - shape hint nếu bạn truyền `--shape`
6. Chọn component tốt nhất.
7. Sinh ảnh annotate vào `output/images`.

### Với video

1. Đọc toàn bộ video trong `input/videos`.
2. Trích frame theo chu kỳ `--frame-interval`.
3. Lưu frame gốc vào `output/videos/<ten_video>/frames`.
4. Chạy detector cho từng frame như image mode.
5. Ghi ảnh annotate vào `output/videos/<ten_video>/detections`.

## Lưu ý

- Script phù hợp nhất khi ad là một vùng đơn sắc rõ ràng.
- Nếu ảnh hoặc frame có nhiều vùng cùng màu hoặc nền quá giống màu ad, nên truyền thêm `--color`.
- Nếu bạn đã biết shape, nên truyền thêm `--shape` để kết quả ổn định hơn.
- Nếu detect bị dư hoặc thiếu mép, hãy thử tăng hoặc giảm `--tolerance`.
- Image mode sẽ tự bỏ qua các file đã annotate có hậu tố `_detected`.
- Toàn bộ `output/` đang được ignore trong git vì đây là dữ liệu sinh ra từ quá trình chạy.

Ví dụ:

```bash
python detect_ad.py --source images --color BB281A --shape rectangle --tolerance 15
python detect_ad.py --source videos --shape l --frame-interval 2
```

## Hướng mở rộng

Có thể làm tiếp các phần sau nếu cần:

- xuất thêm mask riêng của ad
- gom kết quả video thành file report riêng theo từng video
- hỗ trợ nhiều ad trong cùng một frame
- batch processing đệ quy qua subfolder
- thêm unit test cho video sample thật
