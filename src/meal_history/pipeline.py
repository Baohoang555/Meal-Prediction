"""CSV normalization pipeline for meal history exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Danh mục cột chuẩn hoá bắt buộc của hệ thống
CANONICAL_COLUMNS = [
    "STT",
    "Mã chấm công",
    "Họ tên",
    "Phòng ban",
    "Khẩu phần ăn",
    "Món ăn",
    "Khẩu phần ăn bồi dưỡng",
    "Ca ăn",
    "Thành tiền",
    "Tổng tích lũy",
    "Thời gian ăn",
    "Trạng thái",
    "Lý do",
]

# Danh sách ngày nghỉ lễ/nghỉ bù chính thức 2025 - 2026
VIETNAM_HOLIDAYS = {
    # 2025
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-04-07", "2025-04-30", "2025-05-01", "2025-09-01", "2025-09-02",
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-04-27", "2026-04-30", "2026-05-01", "2026-09-01", "2026-09-02",
}

STATUS_FIXES = {
    "H?p l?": "Hợp lệ",
    "Không h?p l?": "Không hợp lệ",
}

REASON_FIXES = {
    "Ch?a ??n th?i gian ca ?n": "Chưa đến thời gian ca ăn",
}


def validate_normalized_data(df: pd.DataFrame) -> None:
    """Validate DataFrame sau khi chuẩn hoá schema & data.
    
    Chặn đứng trường hợp encoding/thứ tự cột bị đoán sai nhưng pipeline vẫn
    âm thầm xuất file rác.
    """
    if "Trạng thái" not in df.columns:
        raise ValueError("Dữ liệu thiếu cột bắt buộc: 'Trạng thái'")

    # 1. Cột 'Trạng thái' chỉ được phép chứa tối đa 2 giá trị nhị phân
    unique_status = df["Trạng thái"].dropna().unique()
    if len(unique_status) > 2:
        raise ValueError(
            f"Cột 'Trạng thái' chứa nhiều hơn 2 giá trị: {list(unique_status)}"
        )

    # 2. Nhận diện cột ngày: ưu tiên 'Thời gian ăn' (schema chuẩn), fallback sang 'Ngày'
    date_col = "Thời gian ăn" if "Thời gian ăn" in df.columns else ("Ngày" if "Ngày" in df.columns else None)
    if not date_col:
        raise ValueError("Không tìm thấy cột ngày ('Thời gian ăn' hoặc 'Ngày') để kiểm tra.")

    # Tỷ lệ parse ngày thành công phải >= 95%
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    valid_ratio = parsed_dates.notna().mean() if len(df) > 0 else 0.0
    if valid_ratio < 0.95:
        raise ValueError(
            f"Tỉ lệ parse ngày hợp lệ chỉ đạt {valid_ratio * 100:.2f}% (< 95%). "
            "Có thể do đoán sai định dạng ngày hoặc encoding/thứ tự cột."
        )


def _read_source(path: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Không thể đọc {path} bằng encoding '{encoding}'. "
            "Hãy truyền --input-encoding phù hợp."
        ) from exc


def transform_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_encoding: str = "cp1258",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize an export, validate data integrity, and return a quality summary."""
    source = Path(input_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {source}")

    frame = _read_source(source, input_encoding)
    if len(frame.columns) != len(CANONICAL_COLUMNS):
        raise ValueError(
            f"CSV phải có {len(CANONICAL_COLUMNS)} cột, "
            f"nhưng nhận được {len(frame.columns)}."
        )

    frame.columns = CANONICAL_COLUMNS
    frame["Trạng thái"] = frame["Trạng thái"].replace(STATUS_FIXES)
    frame["Lý do"] = frame["Lý do"].replace(REASON_FIXES)

    # Bước validation: chặn ghi file nếu dữ liệu sau chuẩn hoá bị sai format/lệch cột
    validate_normalized_data(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "input": str(source),
        "output": str(output),
        "rows": int(len(frame)),
        "columns": CANONICAL_COLUMNS,
        "valid_rows": int((frame["Trạng thái"] == "Hợp lệ").sum()),
        "invalid_rows": int((frame["Trạng thái"] == "Không hợp lệ").sum()),
    }
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    input_encoding: str = "cp1258",
) -> pd.DataFrame:
    """Hàm wrapper cho pipeline, trả về DataFrame sau chuẩn hóa."""
    transform_csv(input_path, output_path, input_encoding=input_encoding)
    return pd.read_csv(output_path, encoding="utf-8-sig")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    parser.add_argument("--input-encoding", default="cp1258")
    parser.add_argument("--report", type=Path, dest="report_path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = transform_csv(
        args.input_path,
        args.output_path,
        input_encoding=args.input_encoding,
        report_path=args.report_path,
    )
    # Keep the CLI usable on Windows consoles that still use cp1252.
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())