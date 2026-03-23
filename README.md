# Ad Image Detector

Script Python để detect vùng quảng cáo (`ad`) đơn sắc trong ảnh hoặc video theo batch mode.

## Cấu trúc thư mục

- `input/images`: chứa ảnh đầu vào
- `input/videos`: chứa video đầu vào
- `output/images`: chứa kết quả khi chạy image mode
- `output/videos`: chứa frame trích từ video và ảnh detect của từng frame
- `detect_ad.py`: script chính

## Cài đặt

Yêu cầu Python 3 và các thư viện:

```bash
pip install opencv-python numpy
```

## Cách chạy

### Chạy với ảnh

```bash
python detect_ad.py --source images
```

### Chạy với video

Mặc định script sẽ trích `2 giây / 1 frame`, sau đó detect toàn bộ frame.

```bash
python detect_ad.py --source videos
```

Nếu muốn đổi chu kỳ trích frame:

```bash
python detect_ad.py --source videos --frame-interval 1.5
```

### In kết quả dạng JSON

```bash
python detect_ad.py --source images --json
python detect_ad.py --source videos --json
```

Trong lúc script đang chạy, console sẽ hiển thị progress bar cho:

- batch ảnh
- quá trình extract frame từ video
- quá trình detect frame trong video

## Tham số

- `--source`: bắt buộc, chọn `images` hoặc `videos`
- `--ad-mode`: chọn mode detect ad
  - `monochrome`: mặc định, dành cho ad đơn sắc
  - `multicolor`: dành cho ad đa sắc, detect theo boundary dọc/ngang
- `--color`: màu ad ở dạng hex, ví dụ `#FFA223` hoặc `FFA223`
- `--shape`: gợi ý shape cho ad: `auto`, `l`, `rectangle`, `square`
- `--tolerance`: độ lệch màu tối đa khi tạo mask, mặc định `20`
- `--frame-interval`: chỉ dùng cho `--source videos`, mặc định `2.0`
- `--json`: in kết quả dạng JSON

## Rule detect

Ad hợp lệ phải là một trong các shape sau:

- `L`
- `square`
- `rectangle`

Một ảnh hoặc frame sẽ bị report `No ads dection` nếu rơi vào một trong các trường hợp:

- vùng detect không phải `L`, `square`, hoặc `rectangle`
- tỷ lệ diện tích ad nhỏ hơn `5%`
- vùng detect phủ gần như toàn bộ frame (`>= 99.5%`)

## Multicolor Mode

Khi chạy với `--ad-mode multicolor`, script sẽ:

1. tìm các điểm đổi màu mạnh giữa `ad` và `content`
2. gom các điểm đó thành boundary thẳng theo trục dọc/ngang
3. sinh ra các object hình học từ boundary đó
4. chỉ mask object được chọn là ad

Rule chọn ad trong `multicolor` mode:

- nếu có object hình `L` thì ưu tiên object đó
- nếu không có `L`, object nhỏ hơn trong các shape hợp lệ `rectangle/square` sẽ được xem là ad
- nếu không tìm được boundary đủ thẳng hoặc boundary không tạo được shape hợp lệ thì trả `No ads dection`

## Output

### Với image mode

Script đọc toàn bộ ảnh trong `input/images` và ghi kết quả vào `output/images`.

### Với video mode

Với mỗi video trong `input/videos`, script sẽ:

1. Trích frame vào `output/videos/<ten_video>/frames`
2. Detect từng frame
3. Ghi kết quả vào `output/videos/<ten_video>/detections`
4. So sánh các frame detect thành công để tìm frame khác biệt

## Quy tắc đặt tên file trong `detections`

- nếu có ad bình thường: filename kết thúc bằng `_detected.png`
- nếu là frame warning: filename kết thúc bằng `_warning.png`
- nếu không có ad: filename kết thúc bằng `_no_ad.png`

Các frame `No ads dection` vẫn được giữ lại trong folder `detections`.

## Nội dung ảnh output

Nếu detect được ad, ảnh output sẽ có:

- vùng ad được highlight
- contour khoanh vùng ad
- kích thước cạnh hoặc nhánh
- `% diện tích` hiển thị bên trong ad

Nếu là `No ads dection`, script vẫn xuất ảnh và gắn nhãn `No ads dection`.

## Warning trong video mode

Sau khi detect xong toàn bộ frame của một video, script sẽ lấy baseline theo các nhóm frame ổn định. Nếu có nhiều frame cùng một tỷ lệ diện tích ad thì chúng sẽ được gom thành một nhóm ổn định.

Mặc định:

- một nhóm ổn định cần ít nhất `5 frame`
- nếu video có `2 ads` ở `2 thời điểm khác nhau`, và mỗi ad xuất hiện đủ nhiều frame với cùng tỷ lệ diện tích, các frame đó sẽ không bị report là warning
- frame nào không thuộc nhóm ổn định nào mới có thể bị warning

Warning có thể xuất hiện khi:

- diện tích ad lớn hơn hoặc nhỏ hơn phần còn lại
- kích thước cạnh hoặc nhánh khác các frame còn lại
- shape khác với đa số frame

Khi một frame bị warning:

- console sẽ highlight bằng nhãn `!!! WARNING FRAME ... !!!`
- file output của frame đó sẽ có hậu tố `_warning.png`

## Ví dụ

```bash
python detect_ad.py --source images --shape rectangle
python detect_ad.py --source images --color BB281A --shape rectangle --tolerance 15
python detect_ad.py --source videos --shape l --frame-interval 2
```

## Ghi chú

- Script phù hợp nhất khi ad là một vùng đơn sắc rõ ràng.
- Nếu nền gần giống màu ad, nên truyền thêm `--color`.
- Nếu đã biết trước shape, nên truyền thêm `--shape` để kết quả ổn định hơn.
- Toàn bộ `output/` là dữ liệu sinh ra khi chạy và đang được ignore trong git.
