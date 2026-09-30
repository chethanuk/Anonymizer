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
import math
import sys
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
import pandas as pd

# Private import: upstream PR 226 moves this to a public rows_for_case, so update this
# import when that lands.
from analyze_benchmark_output import _rows_for_case, analyze_benchmark_output, read_jsonl_table
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.stats import mean_or_none
from measurement_tools.tables import AnalysisExportResult, ExportFormat, ModelTableSpec, write_analysis_tables
from pydantic import BaseModel, Field

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark_comparison")

Direction = Literal["higher", "lower"]
Verdict = Literal["improved", "regressed", "unchanged", "not_comparable"]
GroupKey = tuple[str | None, str | None]

# Render and export order. The ground-truth empty-detection rate sits directly after utility
# on purpose: records with no detected entities skip rewrite with utility 1.0, so mean utility
# rises when detection recall falls. Never read one without the other. It counts only records
# that have ground truth; an empty detection on a record with no PII is correct.
_METRIC_DIRECTION: dict[str, Direction] = {
    "utility_score_mean": "higher",
    "empty_detection_with_ground_truth_rate": "lower",
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
# Group-row fields carried for the context warning only; they get no verdict.
_CONTEXT_COLUMNS = ("total_record_count",)


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
    matched_groups: list[str] = Field(default_factory=list)
    unmatched_baseline: list[str] = Field(default_factory=list)
    unmatched_candidate: list[str] = Field(default_factory=list)
    rows: list[MetricComparisonRow] = Field(default_factory=list)


def compare_benchmark_output(baseline_dir: Path, candidate_dir: Path) -> BenchmarkComparison:
    """Compare two benchmark output directories group by group.

    Args:
        baseline_dir: Benchmark output directory used as the reference run.
        candidate_dir: Benchmark output directory to judge against the baseline.

    Returns:
        Per-metric rows for groups present in both runs, plus the unmatched group labels.

    Raises:
        ValueError: If a run has no record rows, the runs share no group, mixes schema versions, has duplicate
            group keys, or the two runs use different schema versions.
    """
    baseline, baseline_versions = _load_run(baseline_dir)
    candidate, candidate_versions = _load_run(candidate_dir)
    for label, run_dir, run_versions in (
        ("baseline", baseline_dir, baseline_versions),
        ("candidate", candidate_dir, candidate_versions),
    ):
        if len(run_versions) > 1:
            raise ValueError(f"{label} {run_dir} mixes measurement schema versions {sorted(run_versions)}")
    # _load_run reads unversioned as 1, the way the analyzer reads its leak counts, so
    # unversioned vs v1 compares and unversioned vs v2 is rejected.
    if baseline_versions != candidate_versions:
        raise ValueError(
            f"baseline uses measurement schema versions {sorted(baseline_versions)} and candidate uses "
            f"{sorted(candidate_versions)}; compare versions separately"
        )
    versions = baseline_versions
    matched = [key for key in baseline if key in candidate]
    if not matched:
        # No shared groups would print an empty comparison, which reads as a pass.
        raise ValueError(f"{baseline_dir} and {candidate_dir} share no workload/config groups")
    for key in matched:
        counts = (baseline[key].get("total_record_count"), candidate[key].get("total_record_count"))
        if counts[0] != counts[1]:
            logger.warning(
                "%s covers %s baseline record(s) but %s candidate record(s); leak counts are sums "
                "and are not comparable across different records",
                _label(key),
                *counts,
            )
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
        matched_groups=[_label(key) for key in matched],
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
        metrics[key] = {
            name: getattr(group, name)
            for name in (*_METRIC_DIRECTION, *_CONTEXT_COLUMNS)
            if name not in _RECORD_METRIC_COLUMNS
        }
    records = _record_rows(benchmark_dir)
    if records.empty:
        # No record rows would compare as all "unchanged", which reads as a pass.
        raise ValueError(f"{benchmark_dir} has no record rows to compare")
    # Key records by the analyzer's resolved case group, not raw run_tags: the analyzer
    # falls back to top-level fields, detection artifacts and traces when tags are missing.
    record_index: dict[GroupKey, set[object]] = {}
    for case in analysis.cases:
        record_index.setdefault((case.workload_id, case.config_id), set()).update(
            _rows_for_case(records, case.case_id).index
        )
    for key, index in record_index.items():
        group_records = records[records.index.isin(index)]
        metrics[key].update(
            {name: mean_or_none(group_records, column) for name, column in _RECORD_METRIC_COLUMNS.items()}
        )
    versions = {group.measurement_schema_version or 1 for group in analysis.groups}
    return metrics, versions


def _record_rows(benchmark_dir: Path) -> pd.DataFrame:
    measurements = read_jsonl_table(benchmark_dir / "measurements.jsonl", required=True)
    if "record_type" not in measurements.columns:
        return measurements.iloc[0:0]
    return measurements[measurements["record_type"] == "record"]


def _compare_metric(
    key: GroupKey,
    metric: str,
    direction: Direction,
    baseline: float | None,
    candidate: float | None,
) -> MetricComparisonRow:
    delta: float | None = None
    delta_pct: float | None = None
    unchanged = False
    if baseline is not None and candidate is not None:
        delta = candidate - baseline
        # Record-level means depend on summation order, so identical data can differ by a few ulps.
        unchanged = math.isclose(candidate, baseline, rel_tol=1e-9, abs_tol=1e-12)
        if baseline != 0:
            delta_pct = 0.0 if unchanged else delta / abs(baseline) * 100
    if delta is None:
        verdict: Verdict = "not_comparable"
    elif unchanged:
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
    lines = [
        f"Compared {len(result.matched_groups)} matched group(s); "
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
