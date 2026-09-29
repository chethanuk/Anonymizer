import tempfile
from pathlib import Path

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerInput
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.io.writer import write_output

d = Path(tempfile.mkdtemp())
(d / "in.jsonl").write_text('{"id": 1, "text": "Alice lives in Paris"}\n{"id": 2, "text": "Call Bob on 555-0100"}\n')
(d / "in.json").write_text('[{"id": 1, "text": "Alice lives in Paris"}]')
for name in ("in.jsonl", "in.json"):
    try:
        df = read_input(AnonymizerInput(source=str(d / name))).dataframe
        print(f"read  {name:9} OK  {len(df)} rows, columns={list(df.columns)}")
    except Exception as e:
        print(f"read  {name:9} ERR {type(e).__name__}: {e}")
frame = pd.DataFrame({"id": [1], "text_replaced": ["<PERSON> lives in <CITY>"]})
for name in ("out.jsonl", "out.json"):
    try:
        path = write_output(frame, d / name)
        print(f"write {name:9} OK  {path.read_text().strip()}")
    except Exception as e:
        print(f"write {name:9} ERR {type(e).__name__}: {e}")
