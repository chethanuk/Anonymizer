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

Mutation = Callable[[dict[str, object]], object]
_DROP = object()


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(REPO_ROOT / "tests/fixtures/measurement/benchmark-output", destination)
    return destination


def _mutate_records(benchmark_dir: Path, mutate: Mutation) -> None:
    path = benchmark_dir / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if mutate(row) is not _DROP]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# Float means depend on summation order, so the same values in reverse must still compare unchanged.
_REWRITE_VALUES = [0.1, 0.2, 0.7, 0.3, 0.9, 0.11]


def _write_bio_rewrite_records(benchmark_dir: Path, *, reverse: bool) -> None:
    path = benchmark_dir / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    template = next(row for row in rows if _bio_record(row))
    values = _REWRITE_VALUES[::-1] if reverse else _REWRITE_VALUES
    bio_records = [
        {
            **template,
            "record_id": f"bio-{index}",
            "utility_score": value,
            "leakage_mass": value,
            "weighted_leakage_rate": value,
            "needs_human_review": index % 2 == 0,
            "needs_repair": False,
        }
        for index, value in enumerate(values)
    ]
    rows = [row for row in rows if not _bio_record(row)] + bio_records
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _bio_record(row: dict[str, object]) -> bool:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    return row["record_type"] == "record" and tags["workload_id"] == "bio"


def _set_bio_record(**fields: object) -> Mutation:
    def mutate(row: dict[str, object]) -> None:
        if _bio_record(row):
            row.update(fields)

    return mutate


def _drop_bio_true_positives(row: dict[str, object]) -> None:
    if _bio_record(row) and row["entity_true_positive_count"] == 10:
        row["entity_true_positive_count"] = 5
        row["entity_false_negative_count"] = 15


def _improve_bio_rewrite_and_drop_recall(row: dict[str, object]) -> None:
    _set_bio_record(utility_score=0.9, leakage_mass=0.1, needs_human_review=False)(row)
    _drop_bio_true_positives(row)


def _rename_shell_config(row: dict[str, object]) -> None:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    if tags["workload_id"] == "shell":
        tags["config_id"] = "native-local-renamed"


def _rename_every_config(row: dict[str, object]) -> None:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    tags["config_id"] = f"{tags['config_id']}-renamed"


def _add_bio_repetition(benchmark_dir: Path, *, utility_score: float) -> None:
    """Append a second repetition (its own case) to the bio group, with every record at *utility_score*."""
    path = benchmark_dir / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        _set_bio_record(utility_score=0.8)(row)
    second = []
    for row in rows:
        if row["run_tags"]["workload_id"] == "bio":
            tags = {**row["run_tags"], "repetition": 1, "case_id": "bio__default__r001"}
            second.append({**row, "run_tags": tags, **({"utility_score": utility_score} if _bio_record(row) else {})})
    path.write_text("".join(json.dumps(row) + "\n" for row in [*rows, *second]), encoding="utf-8")


def _duplicate_bio_group(row: dict[str, object]) -> None:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    if tags["workload_id"] == "shell":
        tags.update(workload_id="bio", config_id="default", gliner_threshold=0.5)


def _schema_version(version: int) -> Mutation:
    def mutate(row: dict[str, object]) -> None:
        row["schema_version"] = version

    return mutate


def _bio_v2_shell_v1(row: dict[str, object]) -> None:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    row["schema_version"] = 2 if tags["workload_id"] == "bio" else 1


def _drop_records(row: dict[str, object]) -> object:
    return _DROP if row["record_type"] == "record" else None


def _drop_first_bio_record() -> Mutation:
    dropped = False

    def mutate(row: dict[str, object]) -> object:
        nonlocal dropped
        if _bio_record(row) and not dropped:
            dropped = True
            return _DROP
        return None

    return mutate


def _noop(row: dict[str, object]) -> None:
    return None


def _clear_run_tags(row: dict[str, object]) -> None:
    _set_bio_record(utility_score=0.8)(row)
    row["run_tags"] = {}


def _drop_run_tags(row: dict[str, object]) -> None:
    _set_bio_record(utility_score=0.8)(row)
    del row["run_tags"]


def _shell_tags_only_case_id(row: dict[str, object]) -> None:
    tags = row["run_tags"]
    assert isinstance(tags, dict)
    if tags["workload_id"] == "shell":
        if row["record_type"] == "record":
            row["utility_score"] = 0.5
        row["run_tags"] = {"case_id": tags["case_id"]}


@dataclass(frozen=True)
class RejectCase:
    baseline: Mutation = _noop
    candidate: Mutation = _noop
    error: str = ""


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            RejectCase(
                baseline=_schema_version(1),
                candidate=_schema_version(2),
                error=r"baseline uses measurement schema versions \[1\] and candidate uses \[2\]",
            ),
            id="mixed-schema",
        ),
        pytest.param(
            RejectCase(
                candidate=_schema_version(2),
                error=r"baseline uses measurement schema versions \[1\] and candidate uses \[2\]",
            ),
            id="unversioned-vs-versioned-schema",
        ),
        pytest.param(
            RejectCase(
                baseline=_bio_v2_shell_v1,
                candidate=_bio_v2_shell_v1,
                error=r"baseline .* mixes measurement schema versions \[1, 2\]",
            ),
            id="mixed-schema-within-run",
        ),
        pytest.param(
            RejectCase(baseline=_duplicate_bio_group, error=r"has more than one group for bio/default"),
            id="duplicate-group-key",
        ),
        pytest.param(
            RejectCase(candidate=_rename_every_config, error=r"share no workload/config groups"),
            id="no-matched-groups",
        ),
        pytest.param(RejectCase(candidate=_drop_records, error=r"/candidate has no record rows"), id="empty-candidate"),
        pytest.param(RejectCase(baseline=_drop_records, error=r"/baseline has no record rows"), id="empty-baseline"),
    ],
)
def test_compare_benchmark_output_rejects_incompatible_runs(
    case: RejectCase, compare_benchmark_output_tool: ModuleType, tmp_path: Path
) -> None:
    tool = compare_benchmark_output_tool
    baseline_dir = _copy_fixture(tmp_path, "baseline")
    candidate_dir = _copy_fixture(tmp_path, "candidate")
    _mutate_records(baseline_dir, case.baseline)
    _mutate_records(candidate_dir, case.candidate)

    with pytest.raises(ValueError, match=case.error):
        tool.compare_benchmark_output(baseline_dir, candidate_dir)
    with pytest.raises(SystemExit) as raised:
        tool.main(baseline_dir, candidate_dir, json_output=True)
    assert raised.value.code == 125


@dataclass(frozen=True)
class CompareCase:
    baseline: Mutation = _noop
    candidate: Mutation = _noop
    # (workload_id, config_id, metric) -> (verdict, delta_pct)
    expected: dict[tuple[str, str, str], tuple[str, float | None]] = field(default_factory=dict)
    # (workload_id, config_id, metric) -> baseline value
    baseline_values: dict[tuple[str, str, str], float] = field(default_factory=dict)
    rewrite_records: bool = False
    unmatched_baseline: list[str] = field(default_factory=list)
    unmatched_candidate: list[str] = field(default_factory=list)
    all_unchanged: bool = False


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            CompareCase(
                expected={
                    ("bio", "default", "micro_entity_recall"): ("unchanged", 0.0),
                    ("bio", "default", "empty_detection_with_ground_truth_rate"): ("unchanged", None),
                    ("shell", "native-local", "sum_original_value_leak_count"): ("unchanged", 0.0),
                    ("bio", "default", "utility_score_mean"): ("unchanged", 0.0),
                    ("bio", "default", "leakage_mass_mean"): ("unchanged", 0.0),
                    ("bio", "default", "weighted_leakage_rate_mean"): ("unchanged", 0.0),
                    ("bio", "default", "needs_human_review_rate"): ("unchanged", 0.0),
                    ("bio", "default", "needs_repair_rate"): ("unchanged", None),
                },
                baseline_values={
                    ("bio", "default", "needs_human_review_rate"): 0.5,
                    ("bio", "default", "needs_repair_rate"): 0.0,
                },
                rewrite_records=True,
                all_unchanged=True,
            ),
            id="identity",
        ),
        pytest.param(
            CompareCase(
                baseline=_set_bio_record(utility_score=0.8, leakage_mass=0.4, needs_human_review=True),
                candidate=_improve_bio_rewrite_and_drop_recall,
                expected={
                    ("bio", "default", "utility_score_mean"): ("improved", 12.5),
                    ("bio", "default", "leakage_mass_mean"): ("improved", -75.0),
                    ("bio", "default", "micro_entity_recall"): ("regressed", -50.0),
                    ("bio", "default", "needs_human_review_rate"): ("improved", -100.0),
                },
                baseline_values={("bio", "default", "needs_human_review_rate"): 1.0},
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
                baseline=_set_bio_record(leakage_mass=1e-13),
                candidate=_set_bio_record(leakage_mass=0.0),
                expected={("bio", "default", "leakage_mass_mean"): ("unchanged", 0.0)},
            ),
            id="within-tolerance-of-zero-is-not-a-percentage-change",
        ),
        pytest.param(
            CompareCase(
                candidate=_set_bio_record(utility_score=0.9),
                expected={("bio", "default", "utility_score_mean"): ("not_comparable", None)},
            ),
            id="metric-on-one-side",
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
            CompareCase(
                candidate=_schema_version(1),
                expected={("bio", "default", "micro_entity_recall"): ("unchanged", 0.0)},
            ),
            id="unversioned-reads-as-v1",
        ),
        pytest.param(
            CompareCase(
                baseline=_clear_run_tags,
                candidate=_clear_run_tags,
                expected={("bio", "default", "utility_score_mean"): ("unchanged", 0.0)},
                baseline_values={("bio", "default", "utility_score_mean"): 0.8},
            ),
            id="empty-run-tags",
        ),
        pytest.param(
            CompareCase(
                baseline=_drop_run_tags,
                candidate=_drop_run_tags,
                expected={("bio", "default", "utility_score_mean"): ("unchanged", 0.0)},
            ),
            id="no-run-tags",
        ),
        pytest.param(
            CompareCase(
                baseline=_shell_tags_only_case_id,
                candidate=_shell_tags_only_case_id,
                expected={("shell", "native-local", "utility_score_mean"): ("unchanged", 0.0)},
                baseline_values={("shell", "native-local", "utility_score_mean"): 0.5},
            ),
            id="group-resolved-from-artifacts",
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
    if case.rewrite_records:
        _write_bio_rewrite_records(baseline_dir, reverse=False)
        _write_bio_rewrite_records(candidate_dir, reverse=True)

    tool.main(baseline_dir, candidate_dir, json_output=True)
    result = json.loads(capsys.readouterr().out)

    assert result["unmatched_baseline"] == case.unmatched_baseline
    assert result["unmatched_candidate"] == case.unmatched_candidate
    rows = {(row["workload_id"], row["config_id"], row["metric"]): row for row in result["rows"]}
    # Every row belongs to a group the analyzer resolved; none may fall into a None/None bucket.
    assert all(workload is not None and config is not None for workload, config, _ in rows)
    for key, (verdict, delta_pct) in case.expected.items():
        assert rows[key]["verdict"] == verdict, key
        if delta_pct is None:
            assert rows[key]["delta_pct"] is None, key
        else:
            assert rows[key]["delta_pct"] == pytest.approx(delta_pct), key
    for key, value in case.baseline_values.items():
        assert rows[key]["baseline"] == pytest.approx(value), key
    metrics = [row["metric"] for row in result["rows"]]
    for index, row in enumerate(result["rows"]):
        # The rate is omitted for groups without ground truth; when present it must follow utility.
        group = (row["workload_id"], row["config_id"])
        if row["metric"] == "utility_score_mean" and (*group, "empty_detection_with_ground_truth_rate") in rows:
            assert metrics[index + 1] == "empty_detection_with_ground_truth_rate"
    if case.all_unchanged:
        assert {row["verdict"] for row in result["rows"]} == {"unchanged"}


def test_compare_benchmark_output_averages_records_across_repetitions_of_a_group(
    compare_benchmark_output_tool: ModuleType, tmp_path: Path
) -> None:
    tool = compare_benchmark_output_tool
    baseline_dir = _copy_fixture(tmp_path, "baseline")
    candidate_dir = _copy_fixture(tmp_path, "candidate")
    _add_bio_repetition(baseline_dir, utility_score=0.4)
    _add_bio_repetition(candidate_dir, utility_score=0.8)

    result = tool.compare_benchmark_output(baseline_dir, candidate_dir)

    row = next(row for row in result.rows if row.metric == "utility_score_mean" and row.workload_id == "bio")
    assert (row.baseline, row.candidate, row.verdict) == (pytest.approx(0.6), pytest.approx(0.8), "improved")
    assert row.delta_pct == pytest.approx(100 / 3)


def test_compare_benchmark_output_counts_and_renders_matched_groups(
    compare_benchmark_output_tool: ModuleType, tmp_path: Path
) -> None:
    tool = compare_benchmark_output_tool
    baseline_dir = _copy_fixture(tmp_path, "baseline")
    candidate_dir = _copy_fixture(tmp_path, "candidate")

    result = tool.compare_benchmark_output(baseline_dir, candidate_dir)

    assert result.matched_groups == ["bio/default", "shell/native-local"]
    assert tool.render_result(result, json_output=False).startswith("Compared 2 matched group(s)")


def test_compare_benchmark_output_warns_when_runs_cover_different_records(
    compare_benchmark_output_tool: ModuleType, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tool = compare_benchmark_output_tool
    baseline_dir = _copy_fixture(tmp_path, "baseline")
    candidate_dir = _copy_fixture(tmp_path, "candidate")
    _mutate_records(candidate_dir, _drop_first_bio_record())

    with caplog.at_level("WARNING", logger="measurement.benchmark_comparison"):
        tool.compare_benchmark_output(baseline_dir, candidate_dir)

    assert "bio/default" in caplog.text
    assert "shell/native-local" not in caplog.text


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
    lines = table.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 1
    recall = next(line for line in lines[1:] if line.startswith("bio,default,micro_entity_recall,higher,"))
    assert recall.endswith(",unchanged")
    assert (output_dir / "manifest.json").exists()
