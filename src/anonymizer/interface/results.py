# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from pydantic import TypeAdapter

from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.rewrite.rewrite_generation import (
    encode_skipped_span_label_counts,
    restore_empty_skipped_span_label_counts,
)
from anonymizer.interface.display import render_record_html
from anonymizer.interface.errors import AnonymizerIOError

ARTIFACT_FORMAT_VERSION = 1
_ARTIFACT_METADATA_FILE = "metadata.json"
_ARTIFACT_RESULT_FILE = "result.parquet"
_ARTIFACT_TRACE_FILE = "trace.parquet"
_ARTIFACT_FAILED_RECORDS_FILE = "failed_records.json"
_REPLACE_METHOD_ADAPTER: TypeAdapter[Any] = TypeAdapter(ReplaceMethod | None)


class _DisplayMixin:
    """Shared ``display_record`` behavior for result types."""

    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    _display_cycle_index: int

    def display_record(self, index: int | None = None) -> None:
        """Render a record with entity highlights and replacement map in a notebook.

        Args:
            index: Row index to display. If None, cycles through records on repeated calls.
        """
        i = index if index is not None else self._display_cycle_index
        if i < 0 or i >= len(self.trace_dataframe):
            raise IndexError(f"Record index {i} is out of bounds for {len(self.trace_dataframe)} records.")

        row = self.trace_dataframe.iloc[i]
        html_str = render_record_html(row, record_index=i, resolved_text_column=self.resolved_text_column)

        try:
            from IPython.display import HTML, display

            display(HTML(html_str))
        except ImportError:
            print(html_str)

        if index is None:
            self._display_cycle_index = (self._display_cycle_index + 1) % len(self.trace_dataframe)


@dataclass
class AnonymizerResult(_DisplayMixin):
    """Result returned by full anonymization runs.

    Attributes:
        dataframe: User-facing columns only (text, replaced/rewritten text, scores).
        trace_dataframe: Full pipeline trace including all internal columns.
        resolved_text_column: Name of the user-facing text column. Equals the
            user's requested ``text_column`` unless the reader had to rename it
            to avoid colliding with an Anonymizer output column, in which case
            it is the post-rename identifier (e.g. ``"final_entities__input"``).
        failed_records: Records that failed during pipeline processing.
        replace_method: The replace strategy that produced this result. Set by
            ``run()`` / ``preview()``; consumed by ``evaluate()`` to dispatch the
            right judges. ``None`` on results that were constructed by hand or
            loaded from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this result when rewrite
            mode was used. Set by ``run()`` / ``preview()``; consumed by
            ``evaluate()`` to dispatch the rewrite judges. Mutually exclusive
            with ``replace_method``.
        entity_labels: Allowlist of entity labels that were in scope during
            detection. Preserved for ``evaluate()`` so the coverage judge scopes
            its evaluation to the same label set. ``None`` means all default
            labels were in scope.
        data_summary: Optional dataset context supplied with the original input.
            Preserved for ``evaluate()`` so entity-coverage judging uses the
            same context as detection.
        excluded_entity_labels: Labels that were explicitly excluded from
            detection. Preserved for ``evaluate()`` so the coverage judge does
            not penalise the output for not anonymizing excluded labels.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    data_summary: str | None = None
    excluded_entity_labels: list[str] | None = None
    _display_cycle_index: int = field(default=0, init=False, repr=False)

    def __repr__(self) -> str:
        return (
            "AnonymizerResult("
            f"rows={len(self.dataframe)}, "
            f"columns={len(self.dataframe.columns)}, "
            f"trace_columns={len(self.trace_dataframe.columns)}, "
            f"failed_records={len(self.failed_records)}"
            ")"
        )

    def write_artifacts(self, directory: str | Path) -> Path:
        """Save this result as a versioned artifact directory.

        The directory holds ``result.parquet`` (``dataframe``), ``trace.parquet``
        (``trace_dataframe``), ``failed_records.json`` and ``metadata.json`` (format
        version, ``resolved_text_column``, the replace method or privacy goal, and
        the ``evaluate()`` inputs). ``metadata.json`` is removed first and written
        last, so a failed or interrupted write never looks loadable. Other files in
        the directory are left alone. The caller's dataframes are not modified.

        Args:
            directory: Target directory. Created, with parents, if missing.

        Returns:
            The artifact directory path.

        Raises:
            AnonymizerIOError: If any file cannot be written.
        """
        path = Path(directory)
        replace_method = None
        if self.replace_method is not None:
            # ReplaceMethod's discriminator needs "kind" for dict input; its tags are the lowercased class names.
            replace_method = {
                "kind": type(self.replace_method).__name__.lower(),
                **self.replace_method.model_dump(mode="json"),
            }
        metadata = {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "resolved_text_column": self.resolved_text_column,
            "replace_method": replace_method,
            "rewrite_config": None if self.rewrite_config is None else self.rewrite_config.model_dump(mode="json"),
            "entity_labels": self.entity_labels,
            "data_summary": self.data_summary,
            "excluded_entity_labels": self.excluded_entity_labels,
        }
        try:
            path.mkdir(parents=True, exist_ok=True)
            (path / _ARTIFACT_METADATA_FILE).unlink(missing_ok=True)
            self.dataframe.to_parquet(path / _ARTIFACT_RESULT_FILE, index=False)
            encode_skipped_span_label_counts(self.trace_dataframe).to_parquet(path / _ARTIFACT_TRACE_FILE, index=False)
            failed_records = [dataclasses.asdict(record) for record in self.failed_records]
            (path / _ARTIFACT_FAILED_RECORDS_FILE).write_text(json.dumps(failed_records), encoding="utf-8")
            (path / _ARTIFACT_METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
        except (OSError, ValueError, TypeError, pa.ArrowException) as error:
            raise AnonymizerIOError(f"Failed to write result artifacts to {str(path)!r}") from error
        return path

    @classmethod
    def read_artifacts(cls, directory: str | Path) -> AnonymizerResult:
        """Load a result saved by ``write_artifacts``.

        ``metadata.json`` is read and its format version checked before any Parquet
        file is opened. ``skipped_span_label_counts`` comes back as dicts, and the
        replace method, privacy goal and failed records as their original types.
        List cells in trace columns come back as ``numpy.ndarray``, the standard
        pandas Parquet read shape. The result can be passed to ``Anonymizer.evaluate()``.

        Args:
            directory: Artifact directory written by ``write_artifacts``.

        Returns:
            The reconstructed result.

        Raises:
            AnonymizerIOError: If the directory is incomplete or unreadable, or its
                ``artifact_format_version`` is not supported.
        """
        path = Path(directory)
        try:
            metadata = json.loads((path / _ARTIFACT_METADATA_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AnonymizerIOError(f"Failed to read result artifact metadata from {str(path)!r}") from error
        if not isinstance(metadata, dict):
            raise AnonymizerIOError(f"Result artifact metadata in {str(path)!r} must be a JSON object")
        version = metadata.get("artifact_format_version")
        # bool is an int subclass, so ``True == 1`` must not pass as version 1.
        if not (type(version) is int and version == ARTIFACT_FORMAT_VERSION):
            raise AnonymizerIOError(
                f"Unsupported artifact_format_version {version!r} in {str(path)!r}; expected {ARTIFACT_FORMAT_VERSION}"
            )
        try:
            dataframe = pd.read_parquet(path / _ARTIFACT_RESULT_FILE)
            trace_dataframe = pd.read_parquet(path / _ARTIFACT_TRACE_FILE)
            restore_empty_skipped_span_label_counts(trace_dataframe)
            failed_records = json.loads((path / _ARTIFACT_FAILED_RECORDS_FILE).read_text(encoding="utf-8"))
            rewrite_config = metadata["rewrite_config"]
            return cls(
                dataframe=dataframe,
                trace_dataframe=trace_dataframe,
                resolved_text_column=metadata["resolved_text_column"],
                failed_records=[FailedRecord(**record) for record in failed_records],
                replace_method=_REPLACE_METHOD_ADAPTER.validate_python(metadata["replace_method"]),
                rewrite_config=None if rewrite_config is None else PrivacyGoal.model_validate(rewrite_config),
                entity_labels=metadata["entity_labels"],
                data_summary=metadata["data_summary"],
                excluded_entity_labels=metadata["excluded_entity_labels"],
            )
        except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
            raise AnonymizerIOError(f"Failed to read result artifacts from {str(path)!r}") from error


@dataclass
class PreviewResult(_DisplayMixin):
    """Result returned by preview runs.

    Attributes:
        dataframe: User-facing columns only (text, replaced/rewritten text, scores).
        trace_dataframe: Full pipeline trace including all internal columns.
        resolved_text_column: Name of the user-facing text column. Equals the
            user's requested ``text_column`` unless the reader had to rename it
            to avoid colliding with an Anonymizer output column, in which case
            it is the post-rename identifier (e.g. ``"final_entities__input"``).
        failed_records: Records that failed during pipeline processing.
        preview_num_records: Number of records requested for the preview.
        replace_method: The replace strategy that produced this preview. Set by
            ``preview()``; consumed by ``evaluate()`` to dispatch the right
            judges. ``None`` on results that were constructed by hand or loaded
            from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this preview when rewrite
            mode was used. Set by ``preview()``; consumed by ``evaluate()`` to
            dispatch the rewrite judges. Mutually exclusive with ``replace_method``.
        entity_labels: Allowlist of entity labels that were in scope during
            detection. Preserved for ``evaluate()`` so the coverage judge scopes
            its evaluation to the same label set. ``None`` means all default
            labels were in scope.
        data_summary: Optional dataset context supplied with the original input.
            Preserved for ``evaluate()`` so entity-coverage judging uses the
            same context as detection.
        excluded_entity_labels: Labels that were explicitly excluded from
            detection. Preserved for ``evaluate()`` so the coverage judge does
            not penalise the output for not anonymizing excluded labels.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    preview_num_records: int
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    data_summary: str | None = None
    excluded_entity_labels: list[str] | None = None
    _display_cycle_index: int = field(default=0, init=False, repr=False)

    def __repr__(self) -> str:
        return (
            "PreviewResult("
            f"rows={len(self.dataframe)}, "
            f"columns={len(self.dataframe.columns)}, "
            f"trace_columns={len(self.trace_dataframe.columns)}, "
            f"failed_records={len(self.failed_records)}, "
            f"preview_num_records={self.preview_num_records}"
            ")"
        )
