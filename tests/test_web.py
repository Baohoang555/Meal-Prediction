import pandas as pd
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
    holiday = client.get("/api/forecast", params={"target_date": "2026-09-02"}).json()

    assert sunday["predicted_total"] <= weekday["predicted_total"]
    assert holiday["predicted_total"] == 0


def test_frontend_is_served():
    response = TestClient(create_app(_frame())).get("/")
    assert response.status_code == 200
    assert "Meal History Dashboard" in response.text
