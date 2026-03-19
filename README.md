# Ad Image Detector

Script Python để detect vùng quảng cáo (`ad`) đơn sắc trong ảnh và trả về:

- màu của ad
- shape của ad: `l`, `rectangle`, `square`
- kích thước cạnh hoặc nhánh của ad
- diện tích ad theo pixel
- tỷ lệ phần trăm diện tích ad
- ảnh output có khoanh vùng ad, hiển thị kích thước và `% diện tích`

Hiện tại repo đang dùng workflow theo folder:

- `input/`: chứa ảnh đầu vào
- `output/`: chứa ảnh kết quả đã annotate

## Cấu trúc thư mục

- `detect_ad.py`: script chính
- `input/`: folder ảnh đầu vào
- `output/`: folder ảnh đầu ra
- `README.md`: hướng dẫn sử dụng

## Yêu cầu

Cần Python 3 và các thư viện:

```bash
pip install opencv-python numpy
```

## Cách chạy

### 1. Chạy hàng loạt theo folder

Nếu không truyền `--image`, script sẽ tự động đọc toàn bộ ảnh trong `input/` và ghi kết quả sang `output/`.

```bash
python detect_ad.py
```

Bạn cũng có thể chỉ định folder khác:

```bash
python detect_ad.py --input-dir input --output-dir output
```

### 2. Chạy cho một ảnh

```bash
python detect_ad.py --image input/sample3.png --shape rectangle
```

Nếu không truyền `--output`, script sẽ:

- ghi vào `output/<ten_anh>_detected.png` nếu ảnh nằm trong folder `input/`
- hoặc ghi cạnh file gốc nếu ảnh nằm ở nơi khác

Ví dụ:

```bash
python detect_ad.py --image input/sample1.png
python detect_ad.py --image input/sample3.png --shape rectangle
python detect_ad.py --image input/sample3.png --shape rectangle --output output/sample3_result.png
python detect_ad.py --image input/sample3.png --color BB281A --shape rectangle --json
```

## Tham số

- `--image`: đường dẫn tới 1 ảnh cụ thể
- `--input-dir`: folder input cho batch mode, mặc định là `input`
- `--output-dir`: folder output cho batch mode, mặc định là `output`
- `--color`: màu ad ở dạng hex, ví dụ `#FFA223` hoặc `FFA223`
- `--shape`: gợi ý shape cho ad
  - `auto`: tự suy luận
  - `l`: hình chữ L
  - `rectangle`: hình chữ nhật
  - `square`: hình vuông
- `--tolerance`: độ lệch màu tối đa khi tạo mask, mặc định là `20`
- `--output`: đường dẫn ảnh output trong single-image mode
- `--json`: in kết quả dưới dạng JSON

## Output

Script tạo ra 2 loại output:

### 1. Kết quả trên terminal

Single-image mode sẽ in chi tiết cho 1 ảnh:

```text
Image: input\sample3.png
Detected color: #BB281A
Shape: rectangle
Ad area: 60370 px (20.403% of the image)
Bounding box: x=376, y=112, width=331, height=187
Output image: output\sample3_detected.png
Width: 331 px
Height: 187 px
```

Batch mode sẽ in summary cho cả folder:

```text
Input folder: input
Output folder: output
Processed: 4 | Success: 4 | Errors: 0
```

Sau đó script sẽ in kết quả chi tiết cho từng ảnh.

### 2. Ảnh output đã annotate

Mỗi ảnh output sẽ có:

- vùng ad được tô highlight
- contour khoanh đúng vùng ad
- label kích thước cạnh hoặc nhánh
- `% diện tích` hiển thị bên trong ad

Tên file mặc định:

- `output/sample1_detected.png`
- `output/sample2_detected.png`
- `output/sample3_detected.png`

## Output JSON

### Single-image mode

```bash
python detect_ad.py --image input/sample3.png --shape rectangle --json
```

Ví dụ:

```json
{
  "image_path": "input\\sample3.png",
  "detected_color_hex": "#BB281A",
  "shape": "rectangle",
  "inferred_shape": "rectangle",
  "bbox": {
    "x": 376,
    "y": 112,
    "width": 331,
    "height": 187
  },
  "area_pixels": 60370,
  "area_percent": 20.403,
  "dimensions": {
    "width_px": 331,
    "height_px": 187
  },
  "tolerance": 20,
  "rectangularity": 0.9753,
  "selection_score": 58880.671,
  "output_image_path": "output\\sample3_detected.png"
}
```

### Batch mode

```bash
python detect_ad.py --json
```

Batch mode trả về object gồm:

- `input_dir`
- `output_dir`
- `processed_count`
- `success_count`
- `error_count`
- `results`: danh sách kết quả thành công
- `errors`: danh sách ảnh bị lỗi nếu có

## Kết quả hiện tại trên folder input

Chạy:

```bash
python detect_ad.py
```

Kết quả đang detect được:

- `input/sample1.png`: `l`, màu `#FEA11E`, diện tích `40.156%`
- `input/sample2.png`: `l`, màu `#000000`, diện tích `47.826%`
- `input/sample3.png`: `rectangle`, màu `#BB281A`, diện tích `20.403%`
- `input/test.png`: `rectangle`, màu `#FFA327`, diện tích `18.388%`

## Cách script hoạt động

Script làm việc theo hướng đơn giản và thực dụng:

1. Đọc ảnh đầu vào.
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
7. Sinh ảnh output với contour, nhãn kích thước và `% diện tích`.

## Lưu ý

- Script phù hợp nhất khi ad là một vùng đơn sắc rõ ràng.
- Nếu ảnh có nhiều vùng cùng màu hoặc nền quá giống màu ad, nên truyền thêm `--color`.
- Nếu bạn đã biết shape, nên truyền thêm `--shape` để kết quả ổn định hơn.
- Nếu detect bị dư hoặc thiếu mép, hãy thử tăng hoặc giảm `--tolerance`.
- Batch mode sẽ tự bỏ qua các file đã annotate có hậu tố `_detected`.

Ví dụ:

```bash
python detect_ad.py --image input/sample3.png --color BB281A --shape rectangle --tolerance 15
```

## Hướng mở rộng

Có thể làm tiếp các phần sau nếu cần:

- xuất thêm mask riêng của ad
- xử lý nhiều ad trong một ảnh
- batch processing đệ quy qua subfolder
- thêm unit test cho các sample
