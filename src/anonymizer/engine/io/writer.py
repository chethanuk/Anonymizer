# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.interface.errors import AnonymizerIOError, InvalidInputError


def write_output(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Write dataframe to csv/parquet/json/jsonl based on output suffix."""
    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IO_FORMATS:
        raise InvalidInputError(f"Unsupported output format for path: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".csv":
            dataframe.to_csv(path, index=False)
        elif suffix in (".json", ".jsonl"):
            _write_json_records(dataframe, path, lines=suffix == ".jsonl")
        else:
            dataframe.to_parquet(path, index=False)
        return path
    except (OSError, TypeError, ValueError) as error:
        raise AnonymizerIOError(f"Failed to write output data to path: {path}") from error


def _write_json_records(dataframe: pd.DataFrame, path: Path, *, lines: bool) -> None:
    # DataFrame.to_json rounds floats to at most 15 decimal places (10 by default), so
    # scores like 0.8333333333333334 or 1.5e-12 would not survive; json.dumps writes repr.
    if not dataframe.columns.is_unique:
        # to_dict(orient="records") would silently keep only one of two same-named columns.
        raise AnonymizerIOError(f"Cannot write DataFrame with non-unique column names as JSON to path: {path}")
    records = dataframe.astype(object).where(dataframe.notna(), None).to_dict(orient="records")
    try:
        dumps = [
            json.dumps(record, default=_json_default, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
            for record in records
        ]
    except ValueError as error:
        # allow_nan=False rejects inf and NaN nested in list/dict cells; top-level NaN/NA became null above.
        raise AnonymizerIOError(f"Cannot write non-finite values (NaN/inf) as JSON to path: {path}") from error
    if lines:
        path.write_text("".join(f"{line}\n" for line in dumps), encoding="utf-8")
    else:
        path.write_text(f"[{','.join(dumps)}]", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
