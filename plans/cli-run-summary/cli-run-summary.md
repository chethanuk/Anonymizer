<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CLI Run Summary and Preview Output

## Goal

Make `anonymizer run` and `anonymizer preview` readable in a terminal.

- `run` still prints `Output written to: <path>` as its first line, so scripts that parse it keep working. Below it comes a short context block (Mode, Source, Output rows and size, Trace, Failures, Elapsed) and, when records failed, a block that groups them by step and reason with a count, followed by a one-line Tip.
- `preview` prints one block per record with one `column : value` line per output column, each value cut to the terminal width. Entity columns show as a label list.
- Two new `run` flags: `--trace PATH` writes the full trace dataset, `--failed-output PATH` writes the failed records (`record_id, step, reason`).
- Text taken from records never carries C0/C1 control bytes to the terminal.

## Affected Subsystem

`anonymizer.interface.cli` only: `_output.py` (rendering and side-file writers) and `main.py` (flags and wiring). The engine, the Python API and the notebook display are unchanged.

## Answers to the Earlier CLI-Display Review

**Trace dataset wording.** The new flag, its help text and the summary row say "trace dataset" / "Trace". No user-facing CLI string says "dataframe".

**Context header.** The summary shows `Mode / Source / Output (rows, size) / Trace / Failures / Elapsed`. Mode comes from the result's own `replace_method` / `rewrite_config` fields, not from sniffing columns. Trace and Failures rows appear only when those flags were given. The output path is printed once, on the first line.

**Grouped failures, `--failed-output` and the Tip.** Failures are grouped by `(step, reason)` with a count, sorted by count and then by step and reason, so a run with hundreds of identical timeouts prints one line. `--failed-output` sits next to `--trace` and writes a header-only file when nothing failed. The Tip points at `--debug` (which logs each failed record) and, when no export was requested, at `--failed-output`. `preview` only mentions `--debug`.

**TTY-aware styling.** Bold section headers, a green output path, a red failure header and dim keys. Styling is on only when stdout is a TTY, `NO_COLOR` is unset or empty, and `--no-color` was not passed. It uses plain ANSI codes: no new dependency.

## Trade-offs

Rejected alternatives:

- **A rich-text rendering library** (available transitively): not a declared dependency, and its control-code stripping keeps ESC.
- **Guessing the mode from column names**: the result fields are authoritative, and guessing is wrong for hand-built results.
- **A fixed 120-column truncation width**: still wraps on an 80-column terminal. The width comes from `shutil.get_terminal_size()`.
- **A terminal branch inside `interface/display.py`**: that module renders HTML for notebooks and is out of proportion for this change.
- **Putting `trace` / `failed_output` on the shared options**: that would expose them on `validate` and `preview`. Preview is a quick inspection tool.

Known limitations: truncation counts characters, not display cells, so wide CJK text may wrap; bidi override characters are not stripped (they change how text looks, not what the terminal executes).

## Validation

```bash
uv run --group dev pytest tests/interface/cli/ -q
make format-check
make typecheck
make copyright-check
uv run --group dev pytest -q
```

The CLI tests drive `app([...])` with the pipeline patched out and cover: the context rows per mode, failure grouping and order, the Tip variants, value truncation and control-byte stripping, entity label lists, side-file round trips (csv, parquet, header-only), bad side-file paths rejected before the pipeline runs, and the colour gate over TTY, `NO_COLOR` and `--no-color`.

## Rollout

Additive. The new flags are optional and default to off. The only change to existing stdout keeps `Output written to: <path>` as the first line. Nothing is written to stderr by the new code. No configuration, API or dependency changes.
