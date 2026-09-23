#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two benchmark output directories group by group.

Usage:
    uv run python tools/measurement/compare_benchmark_output.py benchmark-runs/baseline benchmark-runs/candidate
    uv run python tools/measurement/compare_benchmark_output.py benchmark-runs/baseline benchmark-runs/candidate \
      --output comparison --format csv
    uv run python tools/measurement/compare_benchmark_output.py benchmark-runs/baseline benchmark-runs/candidate --json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
import pandas as pd
from analyze_benchmark_output import analyze_benchmark_output, read_jsonl_table
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.tables import AnalysisExportResult, ExportFormat, ModelTableSpec, write_analysis_tables
from pydantic import BaseModel, Field

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark_comparison")

Direction = Literal["higher", "lower"]
Verdict = Literal["improved", "regressed", "unchanged", "not_comparable"]
GroupKey = tuple[str | None, str | None]

# Render and export order. empty_detection_rate sits directly after utility on purpose:
# records with no detected entities skip rewrite with utility 1.0, so mean utility
# rises when detection recall falls. Never read one without the other.
_METRIC_DIRECTION: dict[str, Direction] = {
    "utility_score_mean": "higher",
    "empty_detection_rate": "lower",
    "leakage_mass_mean": "lower",
    "weighted_leakage_rate_mean": "lower",
    "needs_human_review_rate": "lower",
    "needs_repair_rate": "lower",
    "micro_entity_precision": "higher",
    "micro_entity_recall": "higher",
    "micro_entity_f1": "higher",
    "micro_detection_valid_rate": "higher",
    "sum_original_value_leak_count": "lower",
    "sum_original_value_leak_unique_value_count": "lower",
    "failed_case_rate": "lower",
    "median_pipeline_elapsed_sec": "lower",
    "median_observed_total_tokens": "lower",
}
# Per-record rewrite fields in measurements.jsonl, averaged per group here because the
# analyzer's group rows do not carry them. Every other metric is a group-row field.
_RECORD_METRIC_COLUMNS = {
    "utility_score_mean": "utility_score",
    "leakage_mass_mean": "leakage_mass",
    "weighted_leakage_rate_mean": "weighted_leakage_rate",
    "needs_human_review_rate": "needs_human_review",
    "needs_repair_rate": "needs_repair",
}


class MetricComparisonRow(BaseModel):
    workload_id: str | None = None
    config_id: str | None = None
    metric: str
    direction: Direction
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    # None when the baseline is 0 or either side is missing; the absolute delta still applies.
    delta_pct: float | None = None
    verdict: Verdict


class BenchmarkComparison(BaseModel):
    baseline_dir: str
    candidate_dir: str
    measurement_schema_version: int | None = None
    unmatched_baseline: list[str] = Field(default_factory=list)
    unmatched_candidate: list[str] = Field(default_factory=list)
    rows: list[MetricComparisonRow] = Field(default_factory=list)


def compare_benchmark_output(baseline_dir: Path, candidate_dir: Path) -> BenchmarkComparison:
    baseline, baseline_versions = _load_run(baseline_dir)
    candidate, candidate_versions = _load_run(candidate_dir)
    versions = baseline_versions | candidate_versions
    if len(versions) > 1:
        raise ValueError(
            f"baseline uses measurement schema versions {sorted(baseline_versions)} and candidate uses "
            f"{sorted(candidate_versions)}; compare versions separately"
        )
    matched = [key for key in baseline if key in candidate]
    rows = [
        _compare_metric(key, metric, direction, baseline[key].get(metric), candidate[key].get(metric))
        for key in matched
        for metric, direction in _METRIC_DIRECTION.items()
        if baseline[key].get(metric) is not None or candidate[key].get(metric) is not None
    ]
    return BenchmarkComparison(
        baseline_dir=str(baseline_dir),
        candidate_dir=str(candidate_dir),
        measurement_schema_version=next(iter(versions), None),
        unmatched_baseline=[_label(key) for key in baseline if key not in candidate],
        unmatched_candidate=[_label(key) for key in candidate if key not in baseline],
        rows=rows,
    )


def _load_run(benchmark_dir: Path) -> tuple[dict[GroupKey, dict[str, float | None]], set[int]]:
    analysis = analyze_benchmark_output(benchmark_dir)
    metrics: dict[GroupKey, dict[str, float | None]] = {}
    for group in analysis.groups:
        key = (group.workload_id, group.config_id)
        if key in metrics:
            raise ValueError(f"{benchmark_dir} has more than one group for {_label(key)}")
        metrics[key] = {name: getattr(group, name) for name in _METRIC_DIRECTION if name not in _RECORD_METRIC_COLUMNS}
    for key, values in _record_metric_means(benchmark_dir).items():
        metrics.setdefault(key, {}).update(values)
    versions = {
        group.measurement_schema_version for group in analysis.groups if group.measurement_schema_version is not None
    }
    return metrics, versions


def _record_metric_means(benchmark_dir: Path) -> dict[GroupKey, dict[str, float | None]]:
    measurements = read_jsonl_table(benchmark_dir / "measurements.jsonl", required=True)
    if "record_type" not in measurements.columns:
        return {}
    records = measurements[measurements["record_type"] == "record"]
    means: dict[GroupKey, dict[str, float | None]] = {}
    for keys, group in records.groupby(["run_tags.workload_id", "run_tags.config_id"], dropna=False):
        workload_id, config_id = keys
        means[(_str_or_none(workload_id), _str_or_none(config_id))] = {
            name: _mean_or_none(group, column) for name, column in _RECORD_METRIC_COLUMNS.items()
        }
    return means


def _mean_or_none(rows: pd.DataFrame, column: str) -> float | None:
    if column not in rows.columns:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _str_or_none(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


def _compare_metric(
    key: GroupKey,
    metric: str,
    direction: Direction,
    baseline: float | None,
    candidate: float | None,
) -> MetricComparisonRow:
    delta: float | None = None
    delta_pct: float | None = None
    if baseline is not None and candidate is not None:
        delta = candidate - baseline
        delta_pct = None if baseline == 0 else delta / abs(baseline) * 100
    if delta is None:
        verdict: Verdict = "not_comparable"
    elif delta == 0:
        verdict = "unchanged"
    elif (delta > 0) == (direction == "higher"):
        verdict = "improved"
    else:
        verdict = "regressed"
    return MetricComparisonRow(
        workload_id=key[0],
        config_id=key[1],
        metric=metric,
        direction=direction,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        delta_pct=delta_pct,
        verdict=verdict,
    )


def _label(key: GroupKey) -> str:
    return f"{key[0]}/{key[1]}"


def write_comparison_table(
    result: BenchmarkComparison, output_dir: Path, export_format: ExportFormat
) -> AnalysisExportResult:
    return write_analysis_tables(
        output_dir, export_format, [ModelTableSpec("metric_comparison", result.rows, MetricComparisonRow)]
    )


def render_result(result: BenchmarkComparison, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    matched = sorted({_label((row.workload_id, row.config_id)) for row in result.rows})
    lines = [
        f"Compared {len(matched)} matched group(s); "
        f"only in baseline={result.unmatched_baseline}, only in candidate={result.unmatched_candidate}"
    ]
    for row in result.rows:
        pct = "n/a" if row.delta_pct is None else f"{row.delta_pct:+.1f}%"
        delta = "n/a" if row.delta is None else f"{row.delta:+.4g}"
        lines.append(
            f"- {_label((row.workload_id, row.config_id))} {row.metric}: "
            f"{row.baseline} -> {row.candidate} (delta={delta}, {pct}) {row.verdict}"
        )
    return "\n".join(lines)


@app.default
def main(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.parquet,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = compare_benchmark_output(baseline_dir, candidate_dir)
        if output is not None:
            write_comparison_table(result, output, format)
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
