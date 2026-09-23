<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Result Artifact Directory

## Goal

Give integrators a supported way to save a full `AnonymizerResult` and load it back without importing from `anonymizer.engine`.

Today a caller has to write `result.dataframe` and `result.trace_dataframe` to Parquet on its own and store the rest of the result however it likes. The trace column `_replacement_application` holds a `skipped_span_label_counts` map that breaks a plain Parquet round trip:

- When every row's map is `{}`, `to_parquet` fails because Arrow cannot write a struct with no fields.
- When `{}` and populated maps are mixed, the write succeeds but `{}` comes back as a struct of `None` values and the counts come back as floats.

## API

```python
from anonymizer.interface.results import AnonymizerResult

result.write_artifacts("run_dir")
loaded = AnonymizerResult.read_artifacts("run_dir")
```

`write_artifacts` writes this layout:

| file | content |
|---|---|
| `result.parquet` | `dataframe` |
| `trace.parquet` | `trace_dataframe`, with `skipped_span_label_counts` stored as a JSON string |
| `failed_records.json` | the `FailedRecord` list |
| `metadata.json` | `artifact_format_version` (1), `resolved_text_column`, the replace method (with its `kind` tag) or privacy goal, `entity_labels`, `data_summary`, `excluded_entity_labels` |

`read_artifacts` reads `metadata.json` first and rejects any `artifact_format_version` other than the integer 1 before it opens a Parquet file. It then decodes the count maps back to dicts and rebuilds the replace method, privacy goal and failed records as their original types. The loaded result can be passed to `Anonymizer.evaluate()`.

All I/O, Arrow, JSON and validation failures raise `AnonymizerIOError` with the original exception as the cause.

## Affected Subsystems

- `interface`: `AnonymizerResult.write_artifacts` and `AnonymizerResult.read_artifacts` in `interface/results.py`.
- `engine/rewrite`: `encode_skipped_span_label_counts` sits beside `restore_empty_skipped_span_label_counts` in `rewrite_generation.py`, so the encoded format has one owner. It returns a copy with new dicts and never mutates the caller's trace.

## Trade-offs

- Only the variable-key `skipped_span_label_counts` map is JSON-encoded. Other nested trace columns have fixed keys and round-trip through Parquet. Their list cells come back as `numpy.ndarray`, which is the standard pandas read shape and is already handled by the display code.
- `metadata.json` is deleted at the start of a write and written last, so a failed or interrupted write never leaves a directory that looks complete.
- Pickle was rejected: it is not language-neutral, it is unsafe to load, and it breaks across pandas versions.
- Dropping `_replacement_application` before writing was rejected because it loses diagnostics.
- Exposing `restore_empty_skipped_span_label_counts` publicly was rejected because every integrator would still need to know the encoding.
- `PreviewResult` persistence and CLI flags are out of scope.

## Validation

- `tests/interface/test_results.py` covers a lossless round trip for replace and rewrite results with all-empty, populated and mixed count maps, and checks that the caller's dicts are unchanged.
- Table-driven tests cover unsupported format versions (`2`, `0`, `"1"`, `True`, missing), incomplete directories, non-object metadata, Arrow write failures on fresh and overwritten directories, results without a strategy or application column, and every replace method variant.

## Rollout

The change is additive and ships format version 1. A future format change bumps `ARTIFACT_FORMAT_VERSION` and adds a reader for the older version at that point.
