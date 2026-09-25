from datetime import date
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from meal_history.pipeline import CANONICAL_COLUMNS
from meal_history.web import create_app


def _frame():
    return pd.DataFrame(
        [
            [1, 101, "Nguyen A", "Team 1", None, None, None, "Ca 1", 44000, 0, "2026-01-01", "Hợp lệ", None],
            [2, 102, "Tran B", "Team 2", None, None, None, "Ca 2", 0, 0, "2026-01-02", "Không hợp lệ", "Lỗi"],
        ],
        columns=CANONICAL_COLUMNS,
    )


def test_api_health_returns_503_when_no_data(tmp_path, monkeypatch):
    """Khi không có data trong state và file không tồn tại -> trả về HTTP 503."""
    non_existent_file = tmp_path / "does_not_exist.csv"
    monkeypatch.setenv("DATA_PATH", str(non_existent_file))
    
    app = create_app(data=None)
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 503
    assert "Dữ liệu chưa được khởi tạo. Vui lòng chạy pipeline cập nhật." in response.json()["detail"]


def test_api_validation_status_endpoint():
    """Kiểm tra endpoint /api/validation-status phản ánh đúng chuẩn kiểm tra chất lượng."""
    client = TestClient(create_app(_frame()))
    res = client.get("/api/validation-status")

    assert res.status_code == 200
    data = res.json()
    assert data["total_rows"] == 2
    assert data["date_validation"]["is_valid"] is True
    assert data["date_validation"]["parse_rate_percent"] == 100.0
    assert data["status_validation"]["is_binary"] is True
    assert data["status_validation"]["unique_count"] == 2


def test_api_summary_options_and_filters():
    client = TestClient(create_app(_frame()))

    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/summary").json()["valid_rows"] == 1
    assert client.get("/api/options").json()["departments"] == ["Team 1", "Team 2"]
    result = client.get("/api/records", params={"status": "Hợp lệ"}).json()
    assert result["total"] == 1
    assert result["items"][0]["Họ tên"] == "Nguyen A"


def test_forecast_reports_missing_meal_categories_explicitly():
    frame = pd.DataFrame(
        [
            [1, 101, "A", "Team 1", None, None, None, None, 0, 0, "2026-09-15 12:00", "Hợp lệ", None],
            [2, 102, "B", "Team 1", None, None, None, None, 0, 0, "2026-09-15 12:01", "Hợp lệ", None],
            [3, 103, "C", "Team 1", None, None, None, None, 0, 0, "2026-09-08 12:00", "Hợp lệ", None],
            [4, 104, "D", "Team 1", None, None, None, None, 0, 0, "2026-09-08 12:01", "Không hợp lệ", None],
        ],
        columns=CANONICAL_COLUMNS,
    )
    client = TestClient(create_app(frame))

    result = client.get("/api/forecast", params={"target_date": "2026-09-22"}).json()

    assert result["predicted_total"] == 2
    assert result["categorized_rows"] == 0
    assert result["uncategorized_valid_rows"] == 3
    assert "mae" in result["metrics"]
    assert all(item["status"] == "missing_data" for item in result["categories"])


def test_forecast_sums_meal_categories():
    frame = pd.DataFrame(
        [
            [1, 101, "A", "Team 1", None, "Món Việt", None, None, 0, 0, "2026-09-15 12:00", "Hợp lệ", None],
            [2, 102, "B", "Team 1", None, "Món Việt", None, None, 0, 0, "2026-09-15 12:01", "Hợp lệ", None],
            [3, 103, "C", "Team 1", None, "Món Âu", None, None, 0, 0, "2026-09-15 12:02", "Hợp lệ", None],
            [4, 104, "D", "Team 1", None, "Món chay", None, None, 0, 0, "2026-09-15 12:03", "Hợp lệ", None],
        ],
        columns=CANONICAL_COLUMNS,
    )

    result = TestClient(create_app(frame)).get(
        "/api/forecast", params={"target_date": "2026-09-22"}
    ).json()

    assert result["predicted_total"] == 4
    assert {item["category"]: item["predicted_servings"] for item in result["categories"]} == {
        "Việt": 2,
        "Âu": 1,
        "Chay": 1,
    }


def test_forecast_keeps_point_metrics_separate_from_interval_metrics():
    frame = pd.DataFrame(
        [
            [i, i, f"Employee {i}", "Team 1", None, None, None, None, 0, 0,
             f"2026-01-{i:02d} 12:00", "Hợp lệ", None]
            for i in range(1, 13)
        ],
        columns=CANONICAL_COLUMNS,
    )

    result = TestClient(create_app(frame)).get(
        "/api/forecast", params={"target_date": "2026-02-01"}
    ).json()

    assert "mae" in result["metrics"]
    assert "coverage" in result["interval_metrics"]
    assert "average_width" in result["interval_metrics"]


def test_forecast_exposes_gru_diagnostics_when_optional_model_is_disabled(monkeypatch):
    monkeypatch.delenv("MEAL_HISTORY_ENABLE_GRU", raising=False)
    result = TestClient(create_app(_frame())).get(
        "/api/forecast", params={"target_date": "2026-02-01"}
    ).json()

    gru = result["model_diagnostics"]["gru"]
    assert gru["model"] == "GRU"
    assert gru["enabled"] is False
    assert gru["status"] == "disabled"
    assert "lookback" in gru
    assert "minimum_observations" in gru


def test_forecast_applies_weekend_and_holiday_rules():
    frame = pd.DataFrame(
        [
            [i, i, f"Employee {i}", "Team 1", None, None, None, None, 0, 0,
             f"2026-09-{i:02d} 12:00", "Hợp lệ", None]
            for i in range(1, 13)
        ],
        columns=CANONICAL_COLUMNS,
    )
    client = TestClient(create_app(frame))

    weekday = client.get("/api/forecast", params={"target_date": "2026-09-18"}).json()
    sunday = client.get("/api/forecast", params={"target_date": "2026-09-20"}).json()
    holiday_national = client.get("/api/forecast", params={"target_date": "2026-09-02"}).json()

    # Kiểm thử các ngày nghỉ lễ/nghỉ bù mới bổ sung
    holiday_tet_2025 = client.get("/api/forecast", params={"target_date": "2025-01-27"}).json()
    holiday_hung_vuong_2026 = client.get("/api/forecast", params={"target_date": "2026-04-27"}).json()

    assert sunday["predicted_total"] <= weekday["predicted_total"]
    assert holiday_national["predicted_total"] == 0
    assert holiday_tet_2025["predicted_total"] == 0
    assert holiday_hung_vuong_2026["predicted_total"] == 0


def test_frontend_is_served():
    response = TestClient(create_app(_frame())).get("/")
    assert response.status_code == 200
    # Khớp tiêu đề và các thành phần trên dashboard mới
    assert "Dự Báo Suất Ăn" in response.text