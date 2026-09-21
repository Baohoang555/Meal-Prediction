import pandas as pd
import pytest

from meal_history.pipeline import CANONICAL_COLUMNS, transform_csv


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
