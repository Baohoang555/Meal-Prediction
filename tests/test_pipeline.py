from datetime import date
import pandas as pd
import pytest

from meal_history.pipeline import (
    CANONICAL_COLUMNS,
    transform_csv,
    validate_normalized_data,
)


def _make_sample_frame(rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


# -------------------------------------------------------------
# Test hàm validate_normalized_data mới
# -------------------------------------------------------------

def test_validate_normalized_data_accepts_clean_dataset():
    """Kiểm tra dữ liệu chuẩn (2 trạng thái, 100% ngày parse được) phải pass."""
    df = _make_sample_frame([
        [1, 101, "Nguyen A", "Team 1", None, None, None, None, 0, 0, "2026-01-01 12:00", "Hợp lệ", None],
        [2, 102, "Tran B", "Team 2", None, None, None, None, 0, 0, "2026-01-02 12:00", "Không hợp lệ", None],
    ])
    # Không raise exception
    validate_normalized_data(df)


def test_validate_normalized_data_rejects_more_than_two_statuses():
    """Cột Trạng thái có > 2 giá trị -> ném ValueError."""
    df = _make_sample_frame([
        [1, 101, "A", "T1", None, None, None, None, 0, 0, "2026-01-01 12:00", "Hợp lệ", None],
        [2, 102, "B", "T1", None, None, None, None, 0, 0, "2026-01-02 12:00", "Không hợp lệ", None],
        [3, 103, "C", "T1", None, None, None, None, 0, 0, "2026-01-03 12:00", "Chờ duyệt", None],  # Giá trị thứ 3
    ])

    with pytest.raises(ValueError, match="Trạng thái"):
        validate_normalized_data(df)


def test_validate_normalized_data_rejects_invalid_date_parse_rate():
    """Cột ngày có tỷ lệ parse lỗi >= 5% (ở đây 1/10 = 10% lỗi) -> ném ValueError."""
    rows = [
        [i, 100 + i, f"User {i}", "T1", None, None, None, None, 0, 0, f"2026-01-{i:02d} 12:00", "Hợp lệ", None]
        for i in range(1, 10)
    ]
    # Thêm 1 dòng có format ngày hỏng hoàn toàn (1/10 dòng = 10% lỗi parse)
    rows.append([10, 110, "Corrupt User", "T1", None, None, None, None, 0, 0, "INVALID_DATETIME_STRING", "Hợp lệ", None])
    df = _make_sample_frame(rows)

    with pytest.raises(ValueError, match="(?i)parse|ngày"):
        validate_normalized_data(df)


def test_holiday_recognition_for_2025_2026_calendar():
    """Xác minh các ngày nghỉ lễ/nghỉ bù 2025-2026 được nhận diện đúng."""
    from meal_history.web import _is_vietnamese_holiday

    new_holidays = [
        date(2025, 1, 27),  # 28 Tết Ất Tỵ
        date(2025, 9, 1),   # Nghỉ liền kề 2/9 năm 2025
        date(2026, 2, 16),  # 29 Tết Bính Ngọ
        date(2026, 4, 27),  # Nghỉ bù Giỗ Tổ Hùng Vương 2026 (Chủ nhật 26/4)
    ]

    for holiday in new_holidays:
        assert _is_vietnamese_holiday(holiday) is True, f"Ngày {holiday} phải được nhận diện là ngày nghỉ lễ."

    regular_day = date(2026, 4, 28)
    assert _is_vietnamese_holiday(regular_day) is False, f"Ngày thường {regular_day} không được đánh dấu là ngày lễ."


# -------------------------------------------------------------
# Giữ lại các test case pipeline hiện có
# -------------------------------------------------------------

def test_transform_normalizes_schema_and_known_values(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "dist" / "normalized.csv"
    report = tmp_path / "dist" / "report.json"
    pd.DataFrame(
        [
            [1, 100, "Employee A", "Team 1", None, None, None, None, 0, 0, "2026-01-01 12:00", "H?p l?", None],
            [2, 101, "Employee B", "Team 2", None, None, None, None, 0, 0, "2026-01-01 12:01", "Không h?p l?", "Ch?a ??n th?i gian ca ?n"],
        ],
        columns=[f"column_{i}" for i in range(len(CANONICAL_COLUMNS))],
    ).to_csv(source, index=False, encoding="cp1258")

    summary = transform_csv(source, output, report_path=report)

    result = pd.read_csv(output, encoding="utf-8-sig")
    assert result.columns.tolist() == CANONICAL_COLUMNS
    assert result["Trạng thái"].tolist() == ["Hợp lệ", "Không hợp lệ"]
    assert result["Lý do"].iloc[1] == "Chưa đến thời gian ca ăn"
    assert summary["rows"] == 2
    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 1
    assert report.is_file()


def test_transform_rejects_wrong_column_count(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"only": [1]}).to_csv(source, index=False, encoding="cp1258")

    with pytest.raises(ValueError, match="13 cột"):
        transform_csv(source, tmp_path / "output.csv")