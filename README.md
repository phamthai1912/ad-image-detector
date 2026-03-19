# Ad Image Detector

Script Python để detect vùng quảng cáo (`ad`) đơn sắc trong ảnh và tính:

- kích thước các cạnh hoặc nhánh của ad
- diện tích ad theo pixel
- tỷ lệ phần trăm diện tích ad so với toàn ảnh

Hiện tại script hỗ trợ tốt cho các dạng:

- `l`: ad hình chữ L
- `rectangle`: ad hình chữ nhật
- `square`: ad hình vuông

Repo này đang có sẵn 3 ảnh mẫu:

- `sample1.png`: ad màu cam, hình chữ `L`
- `sample2.png`: ad màu đen, hình chữ `L`
- `sample3.png`: ad màu đỏ, hình chữ nhật

## File chính

- `detect_ad.py`: script detect ad
- `sample1.png`, `sample2.png`, `sample3.png`: ảnh mẫu để test

## Yêu cầu

Cần Python 3 và các thư viện:

```bash
pip install opencv-python numpy
```

## Cách chạy

Lệnh tổng quát:

```bash
python detect_ad.py --image <duong_dan_anh> [--color HEX] [--shape auto|l|square|rectangle] [--tolerance N] [--json]
```

Ví dụ:

```bash
python detect_ad.py --image sample1.png
python detect_ad.py --image sample2.png
python detect_ad.py --image sample3.png --shape rectangle
python detect_ad.py --image sample3.png --color BB281A --shape rectangle --json
```

## Giải thích tham số

- `--image`: đường dẫn ảnh đầu vào, bắt buộc
- `--color`: màu ad ở dạng hex, ví dụ `#FFA223` hoặc `FFA223`
- `--shape`: gợi ý shape cho ad
  - `auto`: script tự suy luận
  - `l`: hình chữ L
  - `rectangle`: hình chữ nhật
  - `square`: hình vuông
- `--tolerance`: độ lệch màu tối đa khi tạo mask, mặc định là `20`
- `--json`: in kết quả dưới dạng JSON

## Output

Khi chạy bình thường, script sẽ in ra:

- đường dẫn ảnh
- màu detect được
- shape detect được
- diện tích ad theo pixel
- phần trăm diện tích ad
- bounding box của vùng ad
- kích thước cạnh hoặc nhánh của ad

Ví dụ output text:

```bash
python detect_ad.py --image sample3.png --shape rectangle
```

```text
Image: sample3.png
Detected color: #BB281A
Shape: rectangle
Ad area: 60370 px (20.403% of the image)
Bounding box: x=376, y=112, width=331, height=187
Width: 331 px
Height: 187 px
```

Ví dụ output JSON:

```bash
python detect_ad.py --image sample3.png --shape rectangle --json
```

```json
{
  "image_path": "sample3.png",
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
  "selection_score": 131324.671
}
```

## Kết quả trên 3 ảnh mẫu

### sample1.png

```bash
python detect_ad.py --image sample1.png
```

Kết quả:

- màu detect: `#FEA11E`
- shape: `l`
- góc: `bottom-left`
- nhánh trái: `435 x 1079 px`
- nhánh dưới: `1919 x 244 px`
- diện tích: `831461 px`
- tỷ lệ diện tích: `40.156%`

### sample2.png

```bash
python detect_ad.py --image sample2.png
```

Kết quả:

- màu detect: `#000000`
- shape: `l`
- góc: `bottom-left`
- nhánh trái: `534 x 1079 px`
- nhánh dưới: `1919 x 299 px`
- diện tích: `990293 px`
- tỷ lệ diện tích: `47.826%`

### sample3.png

```bash
python detect_ad.py --image sample3.png --shape rectangle
```

Kết quả:

- màu detect: `#BB281A`
- shape: `rectangle`
- width: `331 px`
- height: `187 px`
- diện tích: `60370 px`
- tỷ lệ diện tích: `20.403%`

## Cách script hoạt động

Script làm việc theo hướng đơn giản và thực dụng:

1. Đọc ảnh đầu vào.
2. Nếu không có `--color`, script sẽ lấy các màu nổi bật từ:
   - viền ảnh
   - toàn bộ ảnh
3. Tạo mask cho từng màu candidate dựa trên khoảng cách màu RGB.
4. Tách các connected component.
5. Chấm điểm từng component theo:
   - diện tích
   - độ giống `L`
   - độ giống `rectangle/square`
   - shape hint nếu bạn truyền `--shape`
6. Chọn component tốt nhất và in kết quả.

## Lưu ý

- Script phù hợp nhất khi ad là một vùng đơn sắc rõ ràng.
- Nếu ảnh có nhiều vùng cùng màu hoặc nền quá giống màu ad, nên truyền thêm `--color`.
- Nếu bạn đã biết shape, nên truyền thêm `--shape` để kết quả ổn định hơn.
- Nếu detect bị dư hoặc thiếu mép, hãy thử tăng/giảm `--tolerance`.

Ví dụ:

```bash
python detect_ad.py --image sample3.png --color BB281A --shape rectangle --tolerance 15
```

## Hướng mở rộng

Có thể làm tiếp các phần sau nếu cần:

- xuất ảnh có vẽ bounding box
- xuất mask của ad
- xử lý nhiều ad trong một ảnh
- batch processing cho cả folder
- thêm unit test cho các sample
