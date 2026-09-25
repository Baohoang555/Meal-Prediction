"""Web API and frontend for the meal history dashboard."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .advanced_forecast import GRU_LOOKBACK, MIN_GRU_OBSERVATIONS, gru_forecast, gru_is_enabled
from .pipeline import CANONICAL_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "MealHistory_UTF8.csv"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATE_FORMAT = "%Y-%m-%d"
MEAL_CATEGORIES = ("Việt", "Âu", "Chay")
MEAL_ALIASES = {
    "Việt": ("viet", "món việt", "vietnamese"),
    "Âu": ("au", "món âu", "european", "western"),
    "Chay": ("chay", "vegetarian", "vegan"),
}

# Các ngày lễ cố định hàng năm theo Dương lịch
FIXED_HOLIDAYS = {(1, 1), (4, 30), (5, 1), (9, 2)}

# Bổ sung đầy đủ các ngày nghỉ Tết & nghỉ bù chính thức chu kỳ 2025 - 2026
OFFICIAL_HOLIDAYS_2025_2026 = {
    # Năm 2025
    date(2025, 1, 27),  # Nghỉ Tết Nguyên Đán (28 tháng Chạp)
    date(2025, 1, 28),
    date(2025, 1, 29),
    date(2025, 1, 30),
    date(2025, 1, 31),
    date(2025, 4, 7),   # Giỗ Tổ Hùng Vương 2025 (10/3 Âm lịch)
    date(2025, 9, 1),   # Nghỉ liền kề Quốc khánh 2025
    # Năm 2026
    date(2026, 2, 16),  # Nghỉ Tết Nguyên Đán (29 Tết 2026)
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
    date(2026, 2, 20),
    date(2026, 4, 27),  # Nghỉ bù Giỗ Tổ Hùng Vương 2026 (10/3 ÂL rơi vào Chủ Nhật 26/4)
    date(2026, 9, 1),   # Nghỉ liền kề Quốc khánh 2026
}


def _data_path() -> Path:
    """Xác định đường dẫn file dữ liệu linh hoạt qua DATA_PATH hoặc MEAL_HISTORY_DATA."""
    env_path = os.getenv("DATA_PATH") or os.getenv("MEAL_HISTORY_DATA")
    return Path(env_path) if env_path else DEFAULT_DATA_PATH


def _load_data(target_path: Path | None = None) -> pd.DataFrame:
    """Tải và kiểm tra tính toàn vẹn của dữ liệu; trả về HTTP 503 thân thiện nếu chưa sẵn sàng."""
    path = target_path or _data_path()
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Dữ liệu chưa được khởi tạo. Vui lòng chạy pipeline cập nhật.",
        )
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Dữ liệu chưa được khởi tạo. Vui lòng chạy pipeline cập nhật.",
        ) from exc

    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Dữ liệu thiếu cột: {', '.join(sorted(missing))}. Vui lòng chạy pipeline cập nhật.",
        )
    return frame[CANONICAL_COLUMNS]


def _meal_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).casefold()
    for category, aliases in MEAL_ALIASES.items():
        if any(alias in text for alias in aliases):
            return category
    return None


def _is_vietnamese_holiday(value: date) -> bool:
    return (value.month, value.day) in FIXED_HOLIDAYS or value in OFFICIAL_HOLIDAYS_2025_2026


def _calendar_factor(value: date) -> float:
    if _is_vietnamese_holiday(value):
        return 0.0
    return 0.25 if value.weekday() >= 5 else 1.0


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    if not actual:
        return {"mae": None, "rmse": None, "mape": None, "accuracy": None}
    errors = [prediction - value for value, prediction in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = (sum(error * error for error in errors) / len(errors)) ** 0.5
    non_zero = [(value, prediction) for value, prediction in zip(actual, predicted) if value]
    mape = (
        sum(abs(value - prediction) / abs(value) for value, prediction in non_zero)
        / len(non_zero)
        if non_zero
        else None
    )
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape * 100, 2) if mape is not None else None,
        "accuracy": round(max(0, 1 - mape) * 100, 2) if mape is not None else None,
    }


def _interval_metrics(values: list[int], window: int = 8) -> dict[str, float | None]:
    """Backtest min/max intervals without changing point-forecast metrics."""
    if len(values) <= window:
        return {"coverage": None, "average_width": None}
    covered = 0
    widths = []
    for index in range(window, len(values)):
        history = values[index - window : index]
        lower, upper = min(history), max(history)
        covered += lower <= values[index] <= upper
        widths.append(upper - lower)
    return {
        "coverage": round(covered / (len(values) - window) * 100, 2),
        "average_width": round(sum(widths) / len(widths), 2),
    }


def _model_predictions(history: pd.DataFrame, target: date) -> dict[str, int | None]:
    factor = _calendar_factor(target)
    if factor == 0:
        return {
            "Trung bình cùng thứ": 0,
            "Trung bình có trọng số": 0,
            "Trung bình 7 ngày": 0,
            "Trung vị cùng thứ": 0,
            "San bằng mũ": 0,
            "Xu hướng tuyến tính": 0,
        }

    same_weekday = history[
        history["date"].map(lambda value: value.weekday() == target.weekday())
    ]["servings"].astype(float).tolist()
    recent = history["servings"].astype(float).tolist()
    trend_values = recent[-14:]
    if len(trend_values) >= 3:
        x = list(range(len(trend_values)))
        x_mean = sum(x) / len(x)
        y_mean = sum(trend_values) / len(trend_values)
        denominator = sum((value - x_mean) ** 2 for value in x)
        slope = sum((value - x_mean) * (meal - y_mean) for value, meal in zip(x, trend_values)) / denominator
        intercept = y_mean - slope * x_mean
        trend_prediction = max(0, round(intercept + slope * len(trend_values)))
    else:
        trend_prediction = None
    exponential = None
    if recent:
        smoothed = recent[0]
        for value in recent[1:]:
            smoothed = 0.35 * value + 0.65 * smoothed
        exponential = round(smoothed)
    predictions: dict[str, int | None] = {
        "Trung bình cùng thứ": round(sum(same_weekday[-8:]) / len(same_weekday[-8:]))
        if same_weekday
        else None,
        "Trung bình có trọng số": round(
            sum(value * weight for value, weight in zip(same_weekday[-8:], range(1, len(same_weekday[-8:]) + 1)))
            / sum(range(1, len(same_weekday[-8:]) + 1))
        )
        if same_weekday
        else None,
        "Trung bình 7 ngày": round(sum(recent[-7:]) / len(recent[-7:]))
        if recent
        else None,
        "Trung vị cùng thứ": round(float(pd.Series(same_weekday[-8:]).median()))
        if same_weekday
        else None,
        "San bằng mũ": exponential,
        "Xu hướng tuyến tính": trend_prediction,
    }
    return {
        name: round(value * factor) if value is not None else None
        for name, value in predictions.items()
    }


def _select_model(daily: pd.DataFrame, target: date) -> dict[str, object]:
    rows = daily.sort_values("date").reset_index(drop=True)
    scores: dict[str, list[float]] = {}
    for index, row in rows.iterrows():
        if index < 8:
            continue
        history = rows.iloc[:index]
        predictions = _model_predictions(history, row["date"])
        for name, prediction in predictions.items():
            if prediction is not None and row["servings"] > 0:
                scores.setdefault(name, []).append(
                    abs(float(prediction) - float(row["servings"])) / float(row["servings"])
                )
    leaderboard = []
    for name, errors in scores.items():
        mape = sum(errors) / len(errors) * 100
        leaderboard.append({
            "model": name,
            "mape": round(mape, 2),
            "accuracy": round(max(0, 100 - mape), 2),
            "samples": len(errors),
        })
    leaderboard.sort(key=lambda item: item["mape"])
    chosen = leaderboard[0]["model"] if leaderboard else "Trung bình cùng thứ"
    prediction = _model_predictions(rows[rows["date"] < target], target).get(chosen)

    actual, predicted = [], []
    for index, row in rows.iterrows():
        if index < 8 or row["date"] >= target:
            continue
        value = _model_predictions(rows.iloc[:index], row["date"]).get(chosen)
        if value is not None:
            actual.append(float(row["servings"]))
            predicted.append(float(value))

    gru_values = rows["servings"].astype(float).tolist()
    gru_result = gru_forecast(gru_values) if len(gru_values) >= 30 else None
    gru_diagnostics: dict[str, object] = {
        "model": "GRU",
        "enabled": gru_is_enabled(),
        "available": gru_result is not None,
        "status": "ok" if gru_result is not None else (
            "disabled" if not gru_is_enabled() else "insufficient_data"
        ),
        "minimum_observations": MIN_GRU_OBSERVATIONS,
        "lookback": GRU_LOOKBACK,
    }
    if gru_result is not None and gru_result.predicted:
        gru_metrics = _metrics(gru_result.actual, gru_result.predicted)
        gru_mape = gru_metrics["mape"]
        gru_diagnostics.update({
            **gru_result.technical,
            "metrics": gru_metrics,
        })
        if gru_mape is not None:
            leaderboard.append({
                "model": "GRU",
                "mape": gru_mape,
                "accuracy": gru_metrics["accuracy"],
                "mae": gru_metrics["mae"],
                "rmse": gru_metrics["rmse"],
                "samples": len(gru_result.predicted),
            })
            leaderboard.sort(key=lambda item: (item["mape"], item["model"]))
            if leaderboard[0]["model"] == "GRU":
                chosen = "GRU"
                prediction = gru_result.prediction
                actual = gru_result.actual
                predicted = gru_result.predicted

    if _calendar_factor(target) == 0:
        prediction = 0

    # Đã dọn dẹp biến history không sử dụng
    return {
        "model": chosen,
        "prediction": prediction,
        "metrics": _metrics(actual, predicted),
        "leaderboard": leaderboard,
        "gru_diagnostics": gru_diagnostics,
    }


def _forecast(frame: pd.DataFrame, target_date: date | None = None) -> dict[str, object]:
    """Estimate valid servings by meal category and backtest reliability."""
    target = target_date or (date.today() + timedelta(days=1))
    dates = pd.to_datetime(frame["Thời gian ăn"], errors="coerce", dayfirst=False)
    valid = frame["Trạng thái"].eq("Hợp lệ") & dates.notna()
    categorized = frame.loc[valid, ["Món ăn"]].copy()
    categorized["date"] = dates[valid].dt.date
    categorized["category"] = categorized["Món ăn"].map(_meal_category)
    categorized = categorized.dropna(subset=["category"])
    daily = categorized.groupby(["date", "category"]).size().rename("servings").reset_index()
    overall_daily = (
        pd.DataFrame({"date": dates[valid].dt.date})
        .groupby("date")
        .size()
        .rename("servings")
        .reset_index()
        .sort_values("date")
    )
    selected = _select_model(overall_daily, target)
    overall_target_history = overall_daily[
        overall_daily["date"].map(lambda value: value.weekday() == target.weekday())
        & (overall_daily["date"] < target)
    ].tail(8)
    overall_values = overall_target_history["servings"].astype(int).tolist()
    overall_estimate = selected["prediction"]
    forecasts: list[dict[str, object]] = []
    for category in MEAL_CATEGORIES:
        history = daily[
            (daily["category"] == category)
            & daily["date"].map(lambda value: value.weekday() == target.weekday())
            & (daily["date"] < target)
        ].sort_values("date").tail(8)
        values = history["servings"].astype(int).tolist()
        if not values:
            estimate = None
        elif _calendar_factor(target) == 0:
            estimate = 0
        else:
            estimate = int(round(sum(values) / len(values)))
        actual, predicted = [], []
        for index in range(2, len(values)):
            actual.append(values[index])
            predicted.append(round(sum(values[:index]) / index))
        forecasts.append({
            "category": category,
            "predicted_servings": estimate,
            "min_servings": min(values) if values else None,
            "max_servings": max(values) if values else None,
            "sample_days": len(values),
            "metrics": _metrics(actual, predicted),
            "status": "ok" if values else "missing_data",
        })
    available = [item["predicted_servings"] for item in forecasts if item["predicted_servings"] is not None]
    total = sum(available) if available else overall_estimate

    # Đã dọn dẹp biến all_history không sử dụng
    overall_history_values = overall_daily["servings"].astype(int).tolist()
    overall_metrics = selected["metrics"]
    overall_interval_metrics = _interval_metrics(overall_history_values)
    return {
        "target_date": target.isoformat(),
        "target_weekday": target.strftime("%A"),
        "categories": forecasts,
        "predicted_total": total,
        "metrics": overall_metrics,
        "model": selected["model"],
        "model_leaderboard": selected["leaderboard"],
        "model_diagnostics": {
            "selected": selected["model"],
            "gru": selected["gru_diagnostics"],
        },
        "target_accuracy": selected["leaderboard"][0]["accuracy"] if selected["leaderboard"] else None,
        "interval_metrics": overall_interval_metrics,
        "history_days": int(overall_daily["date"].nunique()),
        "overall_min_servings": min(overall_values) if overall_values else None,
        "overall_max_servings": max(overall_values) if overall_values else None,
        "categorized_rows": int(len(categorized)),
        "uncategorized_valid_rows": int(valid.sum() - len(categorized)),
        "method": f"Tự chọn mô hình {selected['model']}; đã áp dụng lịch ngày thường, cuối tuần và ngày lễ/nghỉ bù Việt Nam",
    }


def create_app(data: pd.DataFrame | None = None) -> FastAPI:
    app = FastAPI(
        title="Meal History API",
        version="2.0.0",
        description="API and dashboard for normalized meal history.",
    )
    app.state.data = data

    @app.get("/api/health")
    def health() -> dict[str, str]:
        frame = _get_frame(app)
        return {"status": "ok", "rows": str(len(frame))}

    @app.get("/api/validation-status")
    def validation_status() -> dict[str, object]:
        """API cung cấp thông số kiểm tra sau chuẩn hóa cho Dashboard."""
        frame = _get_frame(app)
        total = len(frame)
        
        # 1. Kiểm tra parse ngày (thời gian ăn)
        parsed_dates = pd.to_datetime(frame["Thời gian ăn"], errors="coerce", dayfirst=False)
        valid_date_count = int(parsed_dates.notna().sum())
        date_rate = round((valid_date_count / total * 100), 2) if total > 0 else 0.0

        # 2. Kiểm tra cột Trạng thái (phải chuẩn hoá thành đúng 2 giá trị nhị phân)
        statuses = frame["Trạng thái"].dropna().unique().tolist()
        status_counts = {str(k): int(v) for k, v in frame["Trạng thái"].value_counts().items()}

        return {
            "total_rows": total,
            "date_validation": {
                "valid_count": valid_date_count,
                "parse_rate_percent": date_rate,
                "is_valid": date_rate >= 95.0,
                "min_date": str(parsed_dates.min().date()) if valid_date_count > 0 else None,
                "max_date": str(parsed_dates.max().date()) if valid_date_count > 0 else None,
            },
            "status_validation": {
                "unique_values": statuses,
                "unique_count": len(statuses),
                "is_binary": len(statuses) <= 2,
                "breakdown": status_counts,
            },
        }

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        frame = _get_frame(app)
        return {
            "rows": int(len(frame)),
            "valid_rows": int((frame["Trạng thái"] == "Hợp lệ").sum()),
            "invalid_rows": int((frame["Trạng thái"] == "Không hợp lệ").sum()),
            "departments": int(frame["Phòng ban"].nunique()),
            "total_amount": float(pd.to_numeric(frame["Thành tiền"], errors="coerce").fillna(0).sum()),
        }

    @app.get("/api/options")
    def options() -> dict[str, list[str]]:
        frame = _get_frame(app)
        return {
            "departments": sorted(frame["Phòng ban"].dropna().astype(str).unique().tolist()),
            "statuses": sorted(frame["Trạng thái"].dropna().astype(str).unique().tolist()),
        }

    @app.get("/api/forecast")
    def forecast(target_date: date | None = None) -> dict[str, object]:
        return _forecast(_get_frame(app), target_date)

    @app.get("/api/records")
    def records(
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 25,
        search: str = "",
        department: str = "",
        status: str = "",
    ) -> dict[str, object]:
        frame = _get_frame(app)
        filtered = frame
        if search.strip():
            needle = search.strip().casefold()
            mask = filtered.astype(str).apply(
                lambda column: column.str.casefold().str.contains(needle, regex=False, na=False)
            ).any(axis=1)
            filtered = filtered[mask]
        if department:
            filtered = filtered[filtered["Phòng ban"] == department]
        if status:
            filtered = filtered[filtered["Trạng thái"] == status]

        total = len(filtered)
        start = (page - 1) * page_size
        items = filtered.iloc[start : start + page_size]
        items = items.where(pd.notna(items), None)
        return {
            "page": page,
            "page_size": page_size,
            "total": int(total),
            "items": items.to_dict(orient="records"),
        }

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _get_frame(app: FastAPI) -> pd.DataFrame:
    """Truy xuất DataFrame trong state; nếu chưa có thì nạp từ nguồn cấu hình."""
    if app.state.data is None:
        app.state.data = _load_data()
    return app.state.data


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("meal_history.web:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()