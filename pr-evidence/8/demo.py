import tempfile
from pathlib import Path

import pandas as pd

from anonymizer.config.replace_strategies import Redact
from anonymizer.interface.results import AnonymizerResult

COL = "_replacement_application"


def app(counts):
    return {"targeted_span_count": 2, "skipped_span_count": sum(counts.values()), "skipped_span_label_counts": counts}


def result(counts):
    return AnonymizerResult(
        dataframe=pd.DataFrame({"bio": ["t"] * len(counts)}),
        trace_dataframe=pd.DataFrame({COL: [app(c) for c in counts]}),
        resolved_text_column="bio",
        failed_records=[],
        replace_method=Redact(),
    )


for name, counts in [("all-empty", [{}, {}]), ("mixed", [{}, {"street_address": 2}])]:
    d = Path(tempfile.mkdtemp())
    r = result(counts)
    print(f"# {name}: skipped_span_label_counts = {counts}")
    try:
        r.trace_dataframe.to_parquet(d / "trace.parquet")
        back = pd.read_parquet(d / "trace.parquet")[COL].map(lambda a: a["skipped_span_label_counts"]).tolist()
        print(f"  trace_dataframe.to_parquet -> read back {back}")
    except Exception as e:
        print(f"  trace_dataframe.to_parquet -> {type(e).__name__}: {str(e)[:70]}")
    try:
        r.write_artifacts(d / "run")
        loaded = AnonymizerResult.read_artifacts(d / "run")
        back = loaded.trace_dataframe[COL].map(lambda a: a["skipped_span_label_counts"]).tolist()
        print(f"  write_artifacts/read_artifacts -> {back}  equal={back == counts}")
        print(f"  files: {sorted(p.name for p in (d / 'run').iterdir())}")
    except Exception as e:
        print(f"  write_artifacts -> {type(e).__name__}: {e}")
