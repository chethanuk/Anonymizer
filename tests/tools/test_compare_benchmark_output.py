# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

Mutation = Callable[[dict[str, object]], None]


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(REPO_ROOT / "tests/fixtures/measurement/benchmark-output", destination)
    return destination


def _mutate_records(benchmark_dir: Path, mutate: Mutation) -> None:
    path = benchmark_dir / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        mutate(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _bio_record(row: dict[str, object]) -> bool:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    return row["record_type"] == "record" and tags["workload_id"] == "bio"


def _set_bio_record(**fields: float) -> Mutation:
    def mutate(row: dict[str, object]) -> None:
        if _bio_record(row):
            row.update(fields)

    return mutate


def _drop_bio_true_positives(row: dict[str, object]) -> None:
    if _bio_record(row) and row["entity_true_positive_count"] == 10:
        row["entity_true_positive_count"] = 5
        row["entity_false_negative_count"] = 15


def _improve_bio_rewrite_and_drop_recall(row: dict[str, object]) -> None:
    _set_bio_record(utility_score=0.9, leakage_mass=0.1)(row)
    _drop_bio_true_positives(row)


def _rename_shell_config(row: dict[str, object]) -> None:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    if tags["workload_id"] == "shell":
        tags["config_id"] = "native-local-renamed"


def _duplicate_bio_group(row: dict[str, object]) -> None:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    if tags["workload_id"] == "shell":
        tags.update(workload_id="bio", config_id="default", gliner_threshold=0.5)


def _schema_version(version: int) -> Mutation:
    def mutate(row: dict[str, object]) -> None:
        row["schema_version"] = version

    return mutate


def _noop(row: dict[str, object]) -> None:
    return None


@dataclass(frozen=True)
class CompareCase:
    baseline: Mutation = _noop
    candidate: Mutation = _noop
    # (workload_id, config_id, metric) -> (verdict, delta_pct)
    expected: dict[tuple[str, str, str], tuple[str, float | None]] = field(default_factory=dict)
    unmatched_baseline: list[str] = field(default_factory=list)
    unmatched_candidate: list[str] = field(default_factory=list)
    all_unchanged: bool = False
    exit_code: int | None = None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            CompareCase(
                expected={
                    ("bio", "default", "micro_entity_recall"): ("unchanged", 0.0),
                    ("bio", "default", "empty_detection_rate"): ("unchanged", 0.0),
                    ("shell", "native-local", "sum_original_value_leak_count"): ("unchanged", 0.0),
                },
                all_unchanged=True,
            ),
            id="identity",
        ),
        pytest.param(
            CompareCase(
                baseline=_set_bio_record(utility_score=0.8, leakage_mass=0.4),
                candidate=_improve_bio_rewrite_and_drop_recall,
                expected={
                    ("bio", "default", "utility_score_mean"): ("improved", 12.5),
                    ("bio", "default", "leakage_mass_mean"): ("improved", -75.0),
                    ("bio", "default", "micro_entity_recall"): ("regressed", -50.0),
                },
            ),
            id="direction",
        ),
        pytest.param(
            CompareCase(
                baseline=_set_bio_record(leakage_mass=0.0),
                candidate=_set_bio_record(leakage_mass=0.2),
                expected={("bio", "default", "leakage_mass_mean"): ("regressed", None)},
            ),
            id="zero-baseline",
        ),
        pytest.param(
            CompareCase(
                candidate=_rename_shell_config,
                expected={("bio", "default", "micro_entity_recall"): ("unchanged", 0.0)},
                unmatched_baseline=["shell/native-local"],
                unmatched_candidate=["shell/native-local-renamed"],
            ),
            id="unmatched-key",
        ),
        pytest.param(
            CompareCase(baseline=_schema_version(1), candidate=_schema_version(2), exit_code=125),
            id="mixed-schema",
        ),
        pytest.param(
            CompareCase(baseline=_duplicate_bio_group, exit_code=125),
            id="duplicate-group-key",
        ),
    ],
)
def test_compare_benchmark_output_reports_metric_deltas_between_runs(
    case: CompareCase,
    compare_benchmark_output_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = compare_benchmark_output_tool
    baseline_dir = _copy_fixture(tmp_path, "baseline")
    candidate_dir = _copy_fixture(tmp_path, "candidate")
    _mutate_records(baseline_dir, case.baseline)
    _mutate_records(candidate_dir, case.candidate)

    if case.exit_code is not None:
        with pytest.raises(SystemExit) as raised:
            tool.main(baseline_dir, candidate_dir, json_output=True)
        assert raised.value.code == case.exit_code
        return

    tool.main(baseline_dir, candidate_dir, json_output=True)
    result = json.loads(capsys.readouterr().out)

    assert result["unmatched_baseline"] == case.unmatched_baseline
    assert result["unmatched_candidate"] == case.unmatched_candidate
    rows = {(row["workload_id"], row["config_id"], row["metric"]): row for row in result["rows"]}
    for key, (verdict, delta_pct) in case.expected.items():
        assert rows[key]["verdict"] == verdict, key
        if delta_pct is None:
            assert rows[key]["delta_pct"] is None, key
        else:
            assert rows[key]["delta_pct"] == pytest.approx(delta_pct), key
    metrics = [row["metric"] for row in result["rows"]]
    for index, metric in enumerate(metrics):
        if metric == "utility_score_mean":
            assert metrics[index + 1] == "empty_detection_rate"
    if case.all_unchanged:
        assert {row["verdict"] for row in result["rows"]} == {"unchanged"}


def test_compare_benchmark_output_exports_metric_comparison_table(
    compare_benchmark_output_tool: ModuleType, tmp_path: Path
) -> None:
    tool = compare_benchmark_output_tool
    baseline_dir = _copy_fixture(tmp_path, "baseline")
    candidate_dir = _copy_fixture(tmp_path, "candidate")
    output_dir = tmp_path / "comparison"

    tool.main(baseline_dir, candidate_dir, output=output_dir, format=tool.ExportFormat.csv)

    table = output_dir / "metric_comparison.csv"
    assert table.exists()
    header = table.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == [
        "workload_id",
        "config_id",
        "metric",
        "direction",
        "baseline",
        "candidate",
        "delta",
        "delta_pct",
        "verdict",
    ]
    assert (output_dir / "manifest.json").exists()
