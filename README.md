# Meal History Pipeline

Pipeline chuẩn hóa dữ liệu lịch sử suất ăn từ CSV mã hóa CP1258 sang UTF-8-SIG,
phù hợp với Excel và các hệ thống xử lý dữ liệu hiện đại.

## Chạy cục bộ

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
meal-history `
  --input MealHistory2020-01-01_2026-09-21.csv `
  --output dist/MealHistory_UTF8.csv `
  --report dist/quality-report.json
```

CLI mặc định đọc `cp1258`, gán lại schema 13 cột chuẩn, sửa các giá trị trạng
thái/lý do bị lỗi ký tự trong bản xuất và ghi kết quả bằng `utf-8-sig`.

## CI/CD

- **CI** chạy trên mọi push và pull request, kiểm thử với Python 3.11–3.13.
- **CD** chạy khi push vào `main` hoặc chạy thủ công từ GitHub Actions.
  Job tạo dữ liệu chuẩn hóa và xuất `MealHistory_UTF8.csv` cùng báo cáo chất
  lượng thành artifact `meal-history-<commit-sha>` trong environment
  `production`.

Đây là deployment target duy nhất có thể xác định từ repository hiện tại; để
đẩy sang S3, Azure Blob hoặc hệ thống nội bộ, có thể thêm bước upload dùng
secret của repository mà không đưa credential vào mã nguồn.

## Backend và frontend

Backend FastAPI và dashboard frontend được đóng gói trong cùng một service:

```powershell
meal-history-api
# Mở http://localhost:8000
```

API chính:

- `GET /api/health`: kiểm tra service và dữ liệu.
- `GET /api/summary`: tổng số dòng, hợp lệ, không hợp lệ, phòng ban và tổng tiền.
- `GET /api/options`: danh sách phòng ban/trạng thái cho bộ lọc.
- `GET /api/forecast`: dự báo số suất hợp lệ cho ngày mai theo ba nhóm Việt/Âu/
  Chay, kèm tổng suất, khoảng lịch sử và backtest `MAE`, `RMSE`, `MAPE`,
  `accuracy`; có thể truyền `target_date=YYYY-MM-DD` để kiểm tra ngày cụ thể.

Dashboard cũng có phần giải thích bằng ngôn ngữ đơn giản ngay dưới bảng metrics:
MAE/RMSE càng thấp càng tốt, MAPE càng thấp càng tốt và Accuracy tham khảo càng
cao càng tốt. Dashboard cũng hiển thị khoảng dự báo, độ bao phủ khoảng và độ
rộng khoảng. Các chỉ số khoảng này bổ sung thông tin về rủi ro, không được dùng
để làm đẹp hoặc thay đổi Accuracy/MAE/RMSE của dự báo điểm.

Backend backtest và so sánh các mô hình trước khi chọn mô hình tốt nhất: trung
bình cùng thứ, trung bình có trọng số, trung bình 7 ngày và trung vị cùng thứ.
Mục tiêu 75% chỉ được báo là đạt khi backtest thực tế đạt từ 75% trở lên.

Mô hình cũng áp dụng lịch vận hành: ngày lễ Việt Nam được cấu hình sẽ dự báo
0 suất; thứ Bảy và Chủ nhật dùng hệ số 25% so với ngày thường. Backtest dùng
cùng quy tắc này để tránh đánh giá sai khi có cuối tuần hoặc ngày nghỉ.
Hiện đã cấu hình các ngày cố định 1/1, 30/4, 1/5, 2/9 và các ngày Tết đã biết
cho năm 2025–2026; có thể bổ sung lịch nghỉ thực tế hằng năm khi doanh nghiệp
công bố lịch riêng.
- `GET /api/records?page=1&page_size=25&search=&department=&status=`: truy vấn
  có tìm kiếm, lọc và phân trang.

Chạy bằng Docker:

```powershell
docker compose up --build
```

Docker forward port `8001:8000`: truy cập `http://localhost:8001` trên máy
host sẽ được chuyển vào cổng `8000` của container.

Biến môi trường `MEAL_HISTORY_DATA` cho phép trỏ backend tới file CSV UTF-8-SIG
khác.
