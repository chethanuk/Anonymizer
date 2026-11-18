# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Redact
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import COL_FINAL_ENTITIES
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.cli._output import (  # private helpers are tested directly
    _format_elapsed,
    _format_preview,
    _format_run_summary,
    _format_size,
    write_failed_records,
    write_result,
    write_trace,
)
from anonymizer.interface.cli.main import app
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def _make_result(num_rows: int = 2, num_failures: int = 0) -> AnonymizerResult:
    df = pd.DataFrame(
        {"bio": [f"original {i}" for i in range(num_rows)], "bio_replaced": [f"REDACTED_{i}" for i in range(num_rows)]}
    )
    failures = [FailedRecord(record_id=str(i), step="detect", reason="test") for i in range(num_failures)]
    return AnonymizerResult(
        dataframe=df,
        trace_dataframe=df.copy(),
        resolved_text_column="bio",
        failed_records=failures,
    )


@pytest.fixture
def csv_source(tmp_path: Path) -> Path:
    f = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello", "world"]}).to_csv(f, index=False)
    return f


# ---------------------------------------------------------------------------
# write_result unit tests
# ---------------------------------------------------------------------------


def test_write_result_csv(tmp_path: Path) -> None:
    """write_result writes a CSV file that can be read back."""
    result = _make_result(num_rows=2)
    out_path = tmp_path / "out.csv"
    returned = write_result(result, out_path)
    assert out_path.exists()
    assert returned == out_path
    loaded = pd.read_csv(out_path)
    assert list(loaded.columns) == list(result.dataframe.columns)
    assert len(loaded) == 2


def test_write_result_parquet(tmp_path: Path) -> None:
    """write_result writes a Parquet file that can be read back."""
    result = _make_result(num_rows=2)
    out_path = tmp_path / "out.parquet"
    returned = write_result(result, out_path)
    assert out_path.exists()
    assert returned == out_path
    loaded = pd.read_parquet(out_path)
    assert list(loaded.columns) == list(result.dataframe.columns)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# run subcommand output tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".csv", ".parquet"])
def test_run_default_output_path(tmp_path: Path, capsys: pytest.CaptureFixture, ext: str) -> None:
    """run with no --output writes to {stem}_anonymized{ext} next to the source."""
    source = tmp_path / f"data{ext}"
    df = pd.DataFrame({"text": ["hello"]})
    if ext == ".csv":
        df.to_csv(source, index=False)
    else:
        df.to_parquet(source, index=False)

    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result()

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(source), "--replace", "redact"])
    assert exc_info.value.code == 0

    expected = tmp_path / f"data_anonymized{ext}"
    assert expected.exists()
    assert f"data_anonymized{ext}" in capsys.readouterr().out


def test_run_rewrite_default_output_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """run --rewrite with no --output writes to {stem}_rewritten.csv."""
    source = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(source, index=False)

    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result()

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(source), "--rewrite"])
    assert exc_info.value.code == 0

    expected = tmp_path / "data_rewritten.csv"
    assert expected.exists()
    assert "data_rewritten.csv" in capsys.readouterr().out


def test_run_explicit_output(tmp_path: Path, capsys: pytest.CaptureFixture, csv_source: Path) -> None:
    """run with --output writes to the specified path and prints it."""
    out_file = tmp_path / "custom_out.csv"

    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result()

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(csv_source), "--replace", "redact", "--output", str(out_file)])
    assert exc_info.value.code == 0

    assert out_file.exists()
    assert str(out_file) in capsys.readouterr().out


def test_run_prints_context_and_grouped_failures_without_control_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """run tells the user what happened, groups failures by (step, reason), and never echoes control bytes."""
    source = tmp_path / "data.csv"
    pd.DataFrame({"text": ["a", "b", "c", "d", "e", "f"]}).to_csv(source, index=False)
    df = pd.DataFrame({"text": ["a", "b"], "text_replaced": ["[REDACTED_A]", "[REDACTED_B]"]})
    failures = [FailedRecord(record_id=str(i), step="detect", reason="LLM timeout") for i in range(3)]
    failures.append(FailedRecord(record_id="9", step="replace", reason="\x1b]0;pwned\x07\x1b[2Jboom"))
    result = AnonymizerResult(
        dataframe=df,
        trace_dataframe=df.copy(),
        resolved_text_column="text",
        failed_records=failures,
        replace_method=Redact(),
    )
    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = result

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(source), "--replace", "redact", "--output", str(tmp_path / "out.csv")])

    out = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "Output written to:" in out
    assert "replace (redact)" in out
    assert "Elapsed" in out
    detect_line = next(line for line in out.splitlines() if "LLM timeout" in line)
    assert "detect" in detect_line and "x 3" in detect_line
    assert "boom" in out
    assert "\x1b" not in out and "\x07" not in out


# ---------------------------------------------------------------------------
# preview subcommand output tests
# ---------------------------------------------------------------------------


def test_preview_prints_dataframe(tmp_path: Path, capsys: pytest.CaptureFixture, csv_source: Path) -> None:
    """preview prints the result dataframe to stdout."""
    result = _make_result(num_rows=2)
    preview_result = PreviewResult(
        dataframe=result.dataframe,
        trace_dataframe=result.trace_dataframe,
        resolved_text_column="bio",
        failed_records=[],
        preview_num_records=2,
    )

    mock_anonymizer = MagicMock()
    mock_anonymizer.preview.return_value = preview_result

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["preview", "--source", str(csv_source), "--replace", "redact"])
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    assert "bio_replaced" in out
    assert "REDACTED_0" in out


# ---------------------------------------------------------------------------
# side files, formatters, colour gating and flag wiring
# ---------------------------------------------------------------------------

_GOAL = PrivacyGoal(protect="names and home addresses of people", preserve="the overall meaning of each record")


def _failures(groups: list[tuple[str, str, int]]) -> list[FailedRecord]:
    return [
        FailedRecord(record_id=f"{step}-{i}", step=step, reason=reason) for step, reason, n in groups for i in range(n)
    ]


def _summary_lines(result: AnonymizerResult, trace: bool = False, failed: bool = False) -> list[str]:
    return _format_run_summary(
        result,
        source="data.csv",
        output_path=Path("/work/out.csv"),
        output_size=118,
        trace_path=Path("/work/trace.parquet") if trace else None,
        failed_path=Path("/work/failed.csv") if failed else None,
        elapsed=83.0,
        width=80,
        color=False,
    )


def _preview_result(df: pd.DataFrame, failures: list[FailedRecord] | None = None) -> PreviewResult:
    return PreviewResult(
        dataframe=df,
        trace_dataframe=df.copy(),
        resolved_text_column="text",
        failed_records=failures or [],
        preview_num_records=len(df),
    )


@pytest.mark.parametrize(
    ("suffix", "num_failures"),
    [(".csv", 2), (".parquet", 2), (".PARQUET", 1), (".csv", 0)],
    ids=["csv", "parquet", "upper-case-parquet", "no-failures-header-only"],
)
def test_write_side_files_round_trip(tmp_path: Path, suffix: str, num_failures: int) -> None:
    """--trace writes the trace dataset and --failed-output writes one row per failed record."""
    result = _make_result(num_rows=2, num_failures=num_failures)
    trace = write_trace(result, tmp_path / f"trace{suffix}")
    failed = write_failed_records(result, tmp_path / f"failed{suffix}")

    read = pd.read_csv if suffix == ".csv" else pd.read_parquet
    assert list(read(trace).columns) == list(result.trace_dataframe.columns)
    loaded = read(failed, dtype=str) if suffix == ".csv" else read(failed)
    assert list(loaded.columns) == ["record_id", "step", "reason"]
    assert list(loaded.itertuples(index=False, name=None)) == [
        (r.record_id, r.step, r.reason) for r in result.failed_records
    ]


@pytest.mark.parametrize(
    ("result_kwargs", "trace", "failed", "present", "absent"),
    [
        ({"replace_method": Redact()}, False, False, ["Mode    : replace (redact)"], []),
        ({"rewrite_config": _GOAL}, False, False, ["Mode    : rewrite"], ["replace ("]),
        ({}, False, False, ["Source  : data.csv", "Elapsed : 1m 23s"], ["Mode"]),
        ({}, False, False, ["Output  : 2 rows, 118 B", "Failed  : 0"], ["Failed Records", "Tip"]),
        ({"num_rows": 0}, False, False, ["Output  : 0 rows"], []),
        (
            {"failures": [("detect", "LLM timeout", 1)]},
            True,
            True,
            ["Trace   : /work/trace.parquet", "Failed  : 1 -> /work/failed.csv", "--- Failed Records (1) ---"],
            ["--failed-output"],
        ),
        (
            {"failures": [("detect", "LLM timeout", 1)]},
            False,
            False,
            ["pass --failed-output failed.csv", "Failed  : 1\n"],
            [],
        ),
        ({}, False, True, ["Failed  : 0 -> /work/failed.csv"], []),
    ],
    ids=[
        "replace-mode",
        "rewrite-mode",
        "no-mode",
        "no-failures",
        "zero-rows",
        "trace-and-failed-paths",
        "tip-suggests-export",
        "zero-failed-with-export-path",
    ],
)
def test_run_summary_shows_context_rows(
    result_kwargs: dict, trace: bool, failed: bool, present: list[str], absent: list[str]
) -> None:
    """The run summary lists mode, source, output, side files and elapsed time; failures only when present."""
    result_kwargs = dict(result_kwargs)
    num_rows = result_kwargs.pop("num_rows", 2)
    failures = _failures(result_kwargs.pop("failures", []))
    df = pd.DataFrame({"text": ["x"] * num_rows})
    result = AnonymizerResult(
        dataframe=df, trace_dataframe=df, resolved_text_column="text", failed_records=failures, **result_kwargs
    )
    lines = _summary_lines(result, trace=trace, failed=failed)
    text = "\n".join(lines)

    assert lines[0] == "Output written to: /work/out.csv"
    for expected in present:
        assert expected in text
    for unexpected in absent:
        assert unexpected not in text
    assert not any("\x1b" in line for line in lines)
    assert all(line == line.rstrip() for line in lines)
    assert not any(f"{step}-" in text for step in ("detect", "replace"))  # record ids never printed


def test_failures_group_by_count_then_step_and_reason() -> None:
    """Twelve failures in three groups print one line per group, largest first."""
    failures = _failures([("replace", "boom", 1), ("detect", "LLM timeout", 8), ("detect", "bad json", 3)])
    df = pd.DataFrame({"text": ["x"]})
    result = AnonymizerResult(dataframe=df, trace_dataframe=df, resolved_text_column="text", failed_records=failures)
    lines = _summary_lines(result)

    start = lines.index("--- Failed Records (12) ---")
    groups = lines[start + 1 : start + 4]
    assert [g.split()[-1] for g in groups] == ["8", "3", "1"]
    assert "LLM timeout" in groups[0] and "bad json" in groups[1] and "boom" in groups[2]
    assert (
        lines[-1]
        == "Tip: rerun with --debug to log each failed record, or pass --failed-output failed.csv to export them for triage."
    )


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("text", "a" * 120, "a" * 68 + "..."),
        ("text", "line one\nline\ttwo", "line one line two"),
        ("text", "safe\x9b31m\x00text\x1b[2J", "safe31mtext[2J"),
        ("text", float("nan"), ""),
        ("text", None, ""),
        ("text", 0.75, "0.75"),
        ("text", True, "True"),
        (COL_FINAL_ENTITIES, {"entities": [{"label": "first_name", "value": "Maria"}]}, "first_name"),
        (COL_FINAL_ENTITIES, [{"label": "first_name"}, {"label": "company_name"}], "first_name, company_name"),
    ],
    ids=[
        "long-value-truncated",
        "newline-and-tab-flattened",
        "c0-c1-control-bytes-removed",
        "nan-empty",
        "none-empty",
        "float-cast",
        "bool-cast",
        "entities-dict",
        "entities-list",
    ],
)
def test_preview_renders_one_clean_line_per_value(column: str, value: object, expected: str) -> None:
    """Each preview value is one sanitized line that fits the terminal width."""
    lines = _format_preview(_preview_result(pd.DataFrame({column: [value]})), width=80, color=False)

    assert lines[0] == "--- Preview: 1 record ---"
    assert f"  {column} : {expected}".rstrip() in lines
    assert all(len(line) <= 80 for line in lines)


@pytest.mark.parametrize(
    ("df", "failures", "present", "absent"),
    [
        (pd.DataFrame({"text": []}), [], ["--- Preview: 0 records ---"], ["Record 1", "Failed Records"]),
        (
            pd.DataFrame({"text": ["a", "b"]}),
            _failures([("detect", "LLM timeout", 2)]),
            ["Record 2", "--- Failed Records (2) ---", "Tip: rerun with --debug to log each failed record."],
            ["--failed-output"],
        ),
    ],
    ids=["zero-rows", "failures-with-debug-tip"],
)
def test_preview_blocks_and_failures(
    df: pd.DataFrame, failures: list[FailedRecord], present: list[str], absent: list[str]
) -> None:
    """Preview prints a header, one block per record, and grouped failures with a --debug tip."""
    text = "\n".join(_format_preview(_preview_result(df, failures), width=80, color=False))

    for expected in present:
        assert expected in text
    for unexpected in absent:
        assert unexpected not in text


@pytest.mark.parametrize(
    ("command", "isatty", "no_color", "extra_args", "expect_escape"),
    [
        ("preview", True, None, [], True),
        ("preview", True, "1", [], False),
        ("preview", True, "", [], True),
        ("preview", True, None, ["--no-color"], False),
        ("preview", False, None, [], False),
        ("run", True, None, [], True),
        ("run", True, None, ["--no-color"], False),
    ],
    ids=[
        "tty",
        "no-color-env",
        "empty-no-color-env-is-unset",
        "no-color-flag",
        "not-a-tty",
        "run-tty",
        "run-no-color-flag",
    ],
)
def test_color_follows_tty_no_color_env_and_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    csv_source: Path,
    command: str,
    isatty: bool,
    no_color: str | None,
    extra_args: list[str],
    expect_escape: bool,
) -> None:
    """Styling appears only on a terminal, and NO_COLOR or --no-color turns it off."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: isatty)
    if no_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", no_color)
    result = _make_result(num_rows=1)
    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = result
    mock_anonymizer.preview.return_value = _preview_result(result.dataframe)

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app([command, "--source", str(csv_source), "--replace", "redact", *extra_args])

    assert exc_info.value.code == 0
    assert ("\x1b[" in capsys.readouterr().out) == expect_escape


def test_validate_does_not_accept_color_flag(capsys: pytest.CaptureFixture, csv_source: Path) -> None:
    """validate prints nothing styled, so --no-color is not one of its options."""
    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=MagicMock()):
        with pytest.raises(SystemExit) as exc_info:
            app(["validate", "--source", str(csv_source), "--replace", "redact", "--no-color"])

    assert exc_info.value.code != 0
    assert "Config is valid." not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("formatter", "value", "expected"),
    [
        (_format_elapsed, 59.94, "59.9s"),
        (_format_elapsed, 59.96, "1m 00s"),
        (_format_size, 1023, "1023 B"),
        (_format_size, 1024**2 - 1, "1.0 MB"),
    ],
)
def test_summary_units_roll_over_after_rounding(formatter: Callable[[float], str], value: float, expected: str) -> None:
    """A value that rounds up to the unit boundary is shown in the next unit, never as 60.0s or 1024.0 KB."""
    assert formatter(value) == expected


def test_run_output_path_line_stays_literal_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: Path, csv_source: Path
) -> None:
    """Scripts parsing the first line get the bare path even when colors are on."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    out_path = tmp_path / "out.csv"
    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result(num_rows=1)

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit):
            app(["run", "--source", str(csv_source), "--replace", "redact", "--output", str(out_path)])

    assert capsys.readouterr().out.splitlines()[0] == f"Output written to: {out_path}"


def test_preview_long_column_name_fits_terminal() -> None:
    """A very long column name is truncated so preview lines still fit the width."""
    lines = _format_preview(_preview_result(pd.DataFrame({"k" * 200: ["value"]})), width=80, color=False)

    assert all(len(line) <= 80 for line in lines)
    assert any(line.endswith(" : value") for line in lines)


def test_run_writes_trace_and_failed_output_files(
    tmp_path: Path, capsys: pytest.CaptureFixture, csv_source: Path
) -> None:
    """--trace and --failed-output write both files and list them in the summary."""
    trace, failed = tmp_path / "trace.parquet", tmp_path / "failed.csv"
    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result(num_rows=2, num_failures=3)

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(
                ["run", "--source", str(csv_source), "--replace", "redact"]
                + ["--trace", str(trace), "--failed-output", str(failed)]
            )

    out = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert trace.exists() and failed.exists()
    assert len(pd.read_csv(failed)) == 3
    assert f"Trace   : {trace}" in out and f"Failed  : 3 -> {failed}" in out
    assert out.splitlines()[0].startswith("Output written to:")


@pytest.mark.parametrize(
    ("side_args", "message"),
    [
        (["--trace", "{tmp}/trace.txt"], "Unsupported --trace format"),
        (["--failed-output", "{tmp}/out.csv"], "--failed-output path must differ from --output"),
        (["--trace", "{tmp}/x.csv", "--failed-output", "{tmp}/x.csv"], "--failed-output path must differ from --trace"),
        (["--trace", "{source}"], "--trace path must differ from --source"),
        (["--failed-output", "{source}/failed.csv"], "--failed-output parent is not a writable directory"),
        (["--trace", "{tmp}/dir.csv"], "--trace path is a directory"),
        (["--trace", "{tmp}/readonly.csv"], "--trace file is not writable"),
        (["--output", "{tmp}/readonly.csv"], "--output file is not writable"),
        (["--output", "{tmp}/out.txt"], "Unsupported --output format"),
        (["--output", "{source}"], "--output path must differ from --source"),
    ],
    ids=[
        "unsupported-suffix",
        "failed-equals-output",
        "trace-equals-failed",
        "trace-equals-source",
        "parent-is-a-file",
        "path-is-a-directory",
        "trace-existing-file-read-only",
        "output-existing-file-read-only",
        "output-unsupported-suffix",
        "output-equals-source",
    ],
)
def test_run_rejects_bad_side_file_paths_before_running(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    csv_source: Path,
    side_args: list[str],
    message: str,
) -> None:
    """A bad --trace/--failed-output path fails fast, before any pipeline work."""
    if "read-only" in request.node.callspec.id and os.geteuid() == 0:
        pytest.skip("root ignores file permission bits, so the read-only precondition cannot hold")
    (tmp_path / "dir.csv").mkdir()
    (tmp_path / "readonly.csv").write_text("x")
    (tmp_path / "readonly.csv").chmod(0o444)
    args = [a.format(tmp=tmp_path, source=csv_source) for a in side_args]
    mock_anonymizer = MagicMock()

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(
                [
                    "run",
                    "--source",
                    str(csv_source),
                    "--replace",
                    "redact",
                    *([] if "--output" in args else ["--output", str(tmp_path / "out.csv")]),
                    *args,
                ]
            )

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:") and message in err
    mock_anonymizer.run.assert_not_called()


@pytest.mark.parametrize(
    "trace_column",
    [
        [{"span": 1}, "plain text"],  # mixed struct/str: ArrowInvalid
        [{}, {}],  # all-empty dicts: ArrowNotImplementedError
    ],
    ids=["mixed-struct-and-str", "all-empty-dict"],
)
def test_run_unwritable_trace_keeps_failed_output_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture, csv_source: Path, trace_column: list
) -> None:
    """A trace that parquet cannot store gives a normal CLI error, after the failure report and summary."""
    result = _make_result(num_rows=2, num_failures=2)
    result.trace_dataframe = pd.DataFrame({"trace": trace_column})
    trace, failed = tmp_path / "trace.parquet", tmp_path / "failed.csv"
    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = result

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(
                ["run", "--source", str(csv_source), "--replace", "redact"]
                + ["--trace", str(trace), "--failed-output", str(failed)]
            )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.err.startswith("Error:") and "Traceback" not in captured.err and str(trace) in captured.err
    assert len(pd.read_csv(failed)) == 2
    assert "Output written to:" in captured.out and f"Failed  : 2 -> {failed}" in captured.out
    assert "Trace   :" not in captured.out


def test_run_failed_output_write_error_does_not_suggest_the_flag_again(
    tmp_path: Path, capsys: pytest.CaptureFixture, csv_source: Path
) -> None:
    """If the --failed-output write itself fails, the summary says so and does not tell the user to pass the flag."""
    failed = tmp_path / "failed.csv"
    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result(num_rows=2, num_failures=2)

    with (
        patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer),
        patch("anonymizer.interface.cli.main.write_failed_records", side_effect=OSError("disk full")),
    ):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(csv_source), "--replace", "redact", "--failed-output", str(failed)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "disk full" in captured.err
    assert "Failed  : 2 (export failed)" in captured.out
    assert "--failed-output" not in captured.out


def test_run_with_failures_keeps_stdout_contract(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A run with failures still prints the output path, never per-record fields, and nothing on stderr."""
    source = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(source, index=False)
    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result(num_rows=1, num_failures=2)

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(source), "--replace", "redact"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "Output written to:" in captured.out
    assert "record_id" not in captured.out
    assert "reason" not in captured.out
    assert captured.err == ""
