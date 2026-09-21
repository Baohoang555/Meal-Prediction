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
FIXED_HOLIDAYS = {(1, 1), (4, 30), (5, 1), (9, 2)}
KNOWN_TET_HOLIDAYS = {
    date(2025, 1, 28),
    date(2025, 1, 29),
    date(2025, 1, 30),
    date(2025, 1, 31),
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
    date(2026, 2, 20),
}


def _data_path() -> Path:
    return Path(os.getenv("MEAL_HISTORY_DATA", str(DEFAULT_DATA_PATH)))


def _load_data() -> pd.DataFrame:
    path = _data_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại {path}. "
            "Hãy chạy pipeline trước hoặc đặt MEAL_HISTORY_DATA."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột: {', '.join(sorted(missing))}")
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
    return (value.month, value.day) in FIXED_HOLIDAYS or value in KNOWN_TET_HOLIDAYS


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
    history = rows[rows["date"] < target]
    actual, predicted = [], []
    for index, row in rows.iterrows():
        if index < 8 or row["date"] >= target:
            continue
        value = _model_predictions(rows.iloc[:index], row["date"]).get(chosen)
        if value is not None:
            actual.append(float(row["servings"]))
            predicted.append(float(value))
    return {
        "model": chosen,
        "prediction": prediction,
        "metrics": _metrics(actual, predicted),
        "leaderboard": leaderboard,
        "history": history,
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
        estimate = int(round(sum(values) / len(values))) if values else None
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
    all_history = daily[daily["date"] < target].sort_values("date")
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
        "target_accuracy": selected["leaderboard"][0]["accuracy"] if selected["leaderboard"] else None,
        "interval_metrics": overall_interval_metrics,
        "history_days": int(overall_daily["date"].nunique()),
        "overall_min_servings": min(overall_values) if overall_values else None,
        "overall_max_servings": max(overall_values) if overall_values else None,
        "categorized_rows": int(len(categorized)),
        "uncategorized_valid_rows": int(valid.sum() - len(categorized)),
        "method": f"Tự chọn mô hình {selected['model']}; đã áp dụng lịch ngày thường, cuối tuần và ngày lễ Việt Nam",
    }


def create_app(data: pd.DataFrame | None = None) -> FastAPI:
    app = FastAPI(
        title="Meal History API",
        version="1.0.0",
        description="API and dashboard for normalized meal history.",
    )
    app.state.data = data

    @app.get("/api/health")
    def health() -> dict[str, str]:
        try:
            frame = app.state.data if app.state.data is not None else _load_data()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ok", "rows": str(len(frame))}

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
    if app.state.data is None:
        try:
            app.state.data = _load_data()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return app.state.data


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("meal_history.web:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
