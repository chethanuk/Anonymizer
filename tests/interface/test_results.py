# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod, Substitute
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_TEXT,
)
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.errors import AnonymizerIOError
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def test_anonymizer_result_repr_is_compact() -> None:
    result = AnonymizerResult(
        dataframe=pd.DataFrame({"bio": ["a"], "bio_replaced": ["b"]}),
        trace_dataframe=pd.DataFrame(
            {
                "__nemo_anonymizer_text_input__": ["a"],
                "__nemo_anonymizer_text_output__": ["b"],
                COL_DETECTED_ENTITIES: [[]],
            }
        ),
        resolved_text_column="bio",
        failed_records=[],
    )
    rendered = repr(result)
    assert rendered.startswith("AnonymizerResult(")
    assert "rows=1" in rendered
    assert "columns=2" in rendered
    assert "trace_columns=3" in rendered
    assert "failed_records=0" in rendered
    assert "__nemo_anonymizer_text_input__" not in rendered
    assert "bio_replaced" not in rendered


def test_preview_result_repr_is_compact() -> None:
    preview = PreviewResult(
        dataframe=pd.DataFrame({"bio": ["a"], "bio_replaced": ["b"]}),
        trace_dataframe=pd.DataFrame(
            {
                "__nemo_anonymizer_text_input__": ["a"],
                "__nemo_anonymizer_text_output__": ["b"],
                COL_DETECTED_ENTITIES: [[]],
            }
        ),
        resolved_text_column="bio",
        failed_records=[],
        preview_num_records=10,
    )
    rendered = repr(preview)
    assert rendered.startswith("PreviewResult(")
    assert "rows=1" in rendered
    assert "columns=2" in rendered
    assert "trace_columns=3" in rendered
    assert "failed_records=0" in rendered
    assert "preview_num_records=10" in rendered
    assert "__nemo_anonymizer_text_input__" not in rendered
    assert "bio_replaced" not in rendered


def test_anonymizer_result_preserves_positional_data_summary_contract() -> None:
    result = AnonymizerResult(
        pd.DataFrame(),
        pd.DataFrame(),
        "bio",
        [],
        None,
        None,
        ["first_name"],
        "Customer support transcripts.",
    )

    assert result.data_summary == "Customer support transcripts."
    assert result.excluded_entity_labels is None


def test_preview_result_preserves_positional_data_summary_contract() -> None:
    result = PreviewResult(
        pd.DataFrame(),
        pd.DataFrame(),
        "bio",
        [],
        3,
        None,
        None,
        ["first_name"],
        "Customer support transcripts.",
    )

    assert result.data_summary == "Customer support transcripts."
    assert result.excluded_entity_labels is None


def _replacement_application(counts: dict[str, int]) -> dict[str, object]:
    skipped = sum(counts.values())
    return {
        "targeted_span_count": 2,
        "applied_span_count": 2 - skipped,
        "skipped_span_count": skipped,
        "skipped_span_label_counts": counts,
    }


def _artifact_result(mode: str, counts: list[dict[str, int]]) -> AnonymizerResult:
    texts = [f"Synthetic person {i} lives on Example Street" for i in range(len(counts))]
    outputs = [f"[REDACTED] {i}" for i in range(len(counts))]
    return AnonymizerResult(
        dataframe=pd.DataFrame({"bio": texts, "bio_replaced": outputs}),
        trace_dataframe=pd.DataFrame(
            {
                COL_TEXT: texts,
                COL_REPLACED_TEXT: outputs,
                COL_REPLACEMENT_APPLICATION: [_replacement_application(c) for c in counts],
            }
        ),
        resolved_text_column="bio",
        failed_records=[FailedRecord(record_id="r9", step="replace", reason="timeout")],
        replace_method=Redact(format_template="[{label}]") if mode == "replace" else None,
        rewrite_config=(
            PrivacyGoal(protect="direct identifiers and addresses", preserve="the gist of each biography")
            if mode == "rewrite"
            else None
        ),
        entity_labels=["first_name", "street_address"],
        data_summary="Synthetic biographies.",
        excluded_entity_labels=["date"],
    )


@pytest.mark.parametrize("mode", ["replace", "rewrite"])
@pytest.mark.parametrize(
    "counts",
    [
        pytest.param([{}, {}], id="all-empty"),
        pytest.param([{"street_address": 2}, {"first_name": 1}], id="populated"),
        pytest.param([{}, {"street_address": 2}, {}], id="mixed"),
    ],
)
def test_result_round_trips_through_artifact_directory(tmp_path: Path, mode: str, counts: list[dict[str, int]]) -> None:
    result = _artifact_result(mode, counts)
    applications_before = copy.deepcopy(result.trace_dataframe[COL_REPLACEMENT_APPLICATION].tolist())

    result.write_artifacts(tmp_path / "artifacts")
    loaded = AnonymizerResult.read_artifacts(tmp_path / "artifacts")

    assert result.trace_dataframe[COL_REPLACEMENT_APPLICATION].tolist() == applications_before
    pd.testing.assert_frame_equal(loaded.dataframe, result.dataframe)
    pd.testing.assert_frame_equal(loaded.trace_dataframe, result.trace_dataframe)
    assert loaded.trace_dataframe[COL_REPLACEMENT_APPLICATION].tolist() == applications_before
    assert loaded.resolved_text_column == "bio"
    assert loaded.failed_records == result.failed_records
    assert loaded.replace_method == result.replace_method
    assert type(loaded.replace_method) is type(result.replace_method)
    assert loaded.rewrite_config == result.rewrite_config
    assert loaded.entity_labels == result.entity_labels
    assert loaded.data_summary == result.data_summary
    assert loaded.excluded_entity_labels == result.excluded_entity_labels


_MISSING = object()


@pytest.mark.parametrize(
    "version",
    [
        pytest.param(2, id="future"),
        pytest.param(0, id="zero"),
        pytest.param("1", id="string"),
        pytest.param(True, id="bool"),
        pytest.param(_MISSING, id="missing"),
    ],
)
def test_read_artifacts_rejects_unsupported_format_version(tmp_path: Path, version: object) -> None:
    directory = _artifact_result("replace", [{}]).write_artifacts(tmp_path / "artifacts")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if version is _MISSING:
        del metadata["artifact_format_version"]
    else:
        metadata["artifact_format_version"] = version
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(AnonymizerIOError, match="artifact_format_version"):
        AnonymizerResult.read_artifacts(directory)


@pytest.mark.parametrize(
    "remove",
    [
        pytest.param("directory", id="missing-directory"),
        pytest.param("metadata.json", id="missing-metadata"),
        pytest.param("trace.parquet", id="missing-trace"),
    ],
)
def test_read_artifacts_raises_io_error_for_incomplete_directory(tmp_path: Path, remove: str) -> None:
    directory = tmp_path / "artifacts"
    if remove != "directory":
        _artifact_result("replace", [{}]).write_artifacts(directory)
        (directory / remove).unlink()

    with pytest.raises(AnonymizerIOError):
        AnonymizerResult.read_artifacts(directory)


def test_read_artifacts_rejects_non_object_metadata(tmp_path: Path) -> None:
    directory = _artifact_result("replace", [{}]).write_artifacts(tmp_path / "artifacts")
    (directory / "metadata.json").write_text("[1]")

    with pytest.raises(AnonymizerIOError, match="JSON object"):
        AnonymizerResult.read_artifacts(directory)


@pytest.mark.parametrize("overwrite", [pytest.param(False, id="fresh"), pytest.param(True, id="overwrite")])
def test_write_artifacts_wraps_unwritable_dataframe(tmp_path: Path, overwrite: bool) -> None:
    directory = tmp_path / "artifacts"
    if overwrite:
        _artifact_result("replace", [{}]).write_artifacts(directory)
    result = _artifact_result("replace", [{}, {}])
    result.dataframe["extra"] = [{}, {}]

    with pytest.raises(AnonymizerIOError) as exc_info:
        result.write_artifacts(directory)

    assert isinstance(exc_info.value.__cause__, pa.ArrowException)
    assert not (directory / "metadata.json").exists()
    with pytest.raises(AnonymizerIOError):
        AnonymizerResult.read_artifacts(directory)


def test_round_trip_without_replacement_application_column_and_without_strategy(tmp_path: Path) -> None:
    result = AnonymizerResult(
        dataframe=pd.DataFrame({"bio": ["Synthetic text"]}),
        trace_dataframe=pd.DataFrame({COL_TEXT: ["Synthetic text"]}),
        resolved_text_column="bio",
        failed_records=[],
    )

    loaded = AnonymizerResult.read_artifacts(str(result.write_artifacts(str(tmp_path / "nested" / "artifacts"))))

    pd.testing.assert_frame_equal(loaded.dataframe, result.dataframe)
    pd.testing.assert_frame_equal(loaded.trace_dataframe, result.trace_dataframe)
    assert loaded.failed_records == []
    assert loaded.replace_method is None
    assert loaded.rewrite_config is None
    assert loaded.entity_labels is None
    assert loaded.data_summary is None
    assert loaded.excluded_entity_labels is None


@pytest.mark.parametrize(
    "replace_method",
    [
        pytest.param(Annotate(), id="annotate"),
        pytest.param(Redact(format_template="[{label}]"), id="redact"),
        pytest.param(Hash(digest_length=8), id="hash"),
        pytest.param(Substitute(instructions="Keep names short."), id="substitute"),
    ],
)
def test_round_trip_preserves_every_replace_method_variant(tmp_path: Path, replace_method: ReplaceMethod) -> None:
    result = _artifact_result("replace", [{}])
    result.replace_method = replace_method

    loaded = AnonymizerResult.read_artifacts(result.write_artifacts(tmp_path / "artifacts"))

    assert loaded.replace_method == replace_method
    assert type(loaded.replace_method) is type(replace_method)
