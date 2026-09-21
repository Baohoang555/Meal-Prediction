"""CSV normalization pipeline for meal history exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

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

STATUS_FIXES = {
    "H?p l?": "Hợp lệ",
    "Không h?p l?": "Không hợp lệ",
}
REASON_FIXES = {
    "Ch?a ??n th?i gian ca ?n": "Chưa đến thời gian ca ăn",
}


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
    """Normalize an export and return a quality summary."""
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
