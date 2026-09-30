# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd
import pyarrow as pa

from anonymizer.engine.constants import COL_FINAL_ENTITIES
from anonymizer.engine.io.writer import write_output
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.anonymizer import _entity_label, _unwrap_entities  # private, shared with the notebook display
from anonymizer.interface.errors import AnonymizerIOError
from anonymizer.interface.results import AnonymizerResult, PreviewResult

# C0/C1 control bytes except \t and \n: record text must never drive the terminal.
_CONTROL_BYTES = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_FAILED_COLUMNS = [f.name for f in fields(FailedRecord)]
_BOLD, _DIM, _RED = "1", "2", "31"
_MIN_VALUE_WIDTH = 20
_KEY_WIDTH = 8
_DEBUG_TIP = "Tip: rerun with --debug to log each failed record."
_EXPORT_TIP = (
    "Tip: rerun with --debug to log each failed record, or pass --failed-output failed.csv to export them for triage."
)


def write_result(result: AnonymizerResult | PreviewResult, output_path: str | Path) -> Path:
    """Write the result dataframe to a file and return the resolved path."""
    return write_output(result.dataframe, output_path)


def write_trace(result: AnonymizerResult, trace_path: str | Path) -> Path:
    """Write the full pipeline trace dataset to a .csv or .parquet file.

    Args:
        result: Result of a full run.
        trace_path: Destination path; the suffix selects the format.

    Returns:
        The path written.
    """
    try:
        return write_output(result.trace_dataframe, trace_path)
    except pa.ArrowException as error:
        # Parquet rejects some trace shapes (mixed struct/str, all-empty dicts); csv always works.
        raise AnonymizerIOError(f"Failed to write trace data to path: {trace_path} ({error})") from error


def write_failed_records(result: AnonymizerResult, failed_path: str | Path) -> Path:
    """Write failed records (record_id, step, reason) to a .csv or .parquet file.

    The file is written even when nothing failed (header only), so scripts can rely on it.

    Args:
        result: Result of a full run.
        failed_path: Destination path; the suffix selects the format.

    Returns:
        The path written.
    """
    frame = pd.DataFrame([asdict(f) for f in result.failed_records], columns=pd.Index(_FAILED_COLUMNS))
    return write_output(frame, failed_path)


def print_run_summary(
    result: AnonymizerResult,
    *,
    source: str,
    output_path: Path,
    trace_path: Path | None,
    failed_path: Path | None,
    elapsed: float,
    color: bool,
    failed_export_failed: bool = False,
) -> None:
    """Print what a run did: output path, context rows, and grouped failures.

    Args:
        result: Result of a full run.
        source: Input source as the user gave it.
        output_path: Path the output file was written to (must exist).
        trace_path: Path the trace dataset was written to, if requested.
        failed_path: Path the failed records were written to, if requested.
        elapsed: Pipeline wall time in seconds.
        color: Whether styling is allowed; still off on non-TTY stdout or when NO_COLOR is set.
        failed_export_failed: True when --failed-output was given but writing it raised.
    """
    lines = _format_run_summary(
        result,
        source=source,
        output_path=output_path,
        output_size=output_path.stat().st_size,
        trace_path=trace_path,
        failed_path=failed_path,
        elapsed=elapsed,
        width=_terminal_width(),
        color=_use_color(color),
        failed_export_failed=failed_export_failed,
    )
    print("\n".join(lines))


def print_preview(result: PreviewResult, *, color: bool) -> None:
    """Print one block per preview record, then grouped failures.

    Args:
        result: Result of a preview run.
        color: Whether styling is allowed; still off on non-TTY stdout or when NO_COLOR is set.
    """
    print("\n".join(_format_preview(result, width=_terminal_width(), color=_use_color(color))))


def _use_color(color: bool) -> bool:
    return color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _terminal_width() -> int:
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def _format_run_summary(
    result: AnonymizerResult,
    *,
    source: str,
    output_path: Path,
    output_size: int,
    trace_path: Path | None,
    failed_path: Path | None,
    elapsed: float,
    width: int,
    color: bool,
    failed_export_failed: bool = False,
) -> list[str]:
    # Unstyled on purpose: scripts parse this first line for the literal path.
    lines = [f"Output written to: {_one_line(str(output_path))}"]
    rows: list[tuple[str, str]] = []
    mode = _mode(result)
    if mode is not None:
        rows.append(("Mode", mode))
    rows.append(("Source", _one_line(source)))
    n = len(result.dataframe)
    rows.append(("Output", f"{n} row{'' if n == 1 else 's'}, {_format_size(output_size)}"))
    if trace_path is not None:
        rows.append(("Trace", _one_line(str(trace_path))))
    failed = f"{len(result.failed_records)}"
    if failed_path is not None:
        failed += f" -> {_one_line(str(failed_path))}"
    elif failed_export_failed:
        failed += " (export failed)"
    rows.append(("Failed", failed))
    rows.append(("Elapsed", _format_elapsed(elapsed)))
    lines += [f"{_style(f'{key:<{_KEY_WIDTH}}', _DIM, color)}: {value}".rstrip() for key, value in rows]
    tip = _DEBUG_TIP if failed_path is not None or failed_export_failed else _EXPORT_TIP
    return lines + _format_failures(result.failed_records, tip=tip, width=width, color=color)


def _format_preview(result: PreviewResult, *, width: int, color: bool) -> list[str]:
    frame = result.dataframe
    n = len(frame)
    mode = _mode(result)
    title = f"Preview: {n} record{'' if n == 1 else 's'}" + (f", {mode}" if mode else "")
    lines = [_style(f"--- {title} ---", _BOLD, color)]
    keys = [_one_line(str(c), max(width // 3, _KEY_WIDTH)) for c in frame.columns]
    key_width = max((len(k) for k in keys), default=0)
    value_width = max(width - key_width - 5, _MIN_VALUE_WIDTH)
    for i, (_, row) in enumerate(frame.iterrows(), start=1):
        lines += ["", _style(f"Record {i}", _BOLD, color)]
        for key, column in zip(keys, frame.columns):
            value = _one_line(_cell_text(column, row[column]), value_width)
            lines.append(f"  {_style(f'{key:<{key_width}}', _DIM, color)} : {value}".rstrip())
    return lines + _format_failures(result.failed_records, tip=_DEBUG_TIP, width=width, color=color)


def _format_failures(failed: list[FailedRecord], *, tip: str, width: int, color: bool) -> list[str]:
    if not failed:
        return []
    counts = Counter((_one_line(f.step), _one_line(f.reason)) for f in failed)
    groups = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    step_width = max(len(step) for step, _ in counts)
    reason_max = max(width - step_width - 8 - len(str(groups[0][1])), _MIN_VALUE_WIDTH)
    reasons = [_one_line(reason, reason_max) for (_, reason), _ in groups]
    reason_width = max(len(r) for r in reasons)
    lines = ["", _style(f"--- Failed Records ({len(failed)}) ---", _RED, color)]
    for ((step, _), n), reason in zip(groups, reasons):
        lines.append(f"  {step:<{step_width}}  {reason:<{reason_width}}  x {n}")
    return lines + ["", tip]


def _cell_text(column: object, value: object) -> str:
    if column == COL_FINAL_ENTITIES:
        return ", ".join(label for e in _unwrap_entities(value) if (label := _entity_label(e)))
    if isinstance(value, str):
        return value
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return ""
    return str(value)


def _mode(result: AnonymizerResult | PreviewResult) -> str | None:
    if result.replace_method is not None:
        return f"replace ({type(result.replace_method).__name__.lower()})"
    if result.rewrite_config is not None:
        return "rewrite"
    return None


def _clean(text: str) -> str:
    return _CONTROL_BYTES.sub("", text)


def _one_line(text: str, max_len: int | None = None) -> str:
    flat = _clean(text).replace("\n", " ").replace("\t", " ")
    if max_len is not None and len(flat) > max_len:
        return flat[: max_len - 3] + "..."
    return flat


def _style(text: str, code: str, color: bool) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if color else text


def _format_elapsed(seconds: float) -> str:
    # Compare after rounding so 59.96 becomes "1m 00s", not "60.0s".
    if round(seconds, 1) < 60:
        return f"{seconds:.1f}s"
    total = round(seconds)
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {total % 3600 // 60:02d}m"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = size / 1024
    for unit in ("KB", "MB"):
        # Compare after rounding so 1048575 becomes "1.0 MB", not "1024.0 KB".
        if round(value, 1) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
