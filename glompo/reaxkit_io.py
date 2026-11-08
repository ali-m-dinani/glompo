"""File-only history export for consumers which do not import GloMPO."""

import hashlib
import json
import math
from pathlib import Path

import tables


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_native(directory):
    directory = Path(directory)
    source = directory / "glompo_log.h5"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    records = []
    with tables.open_file(str(source), "r") as archive:
        for table in archive.walk_nodes("/", "Table"):
            if table.name != "iter_hist" or not {"call_id", "fx", "x"}.issubset(table.colnames):
                continue
            optimizer = table._v_parent._v_name
            for index, row in enumerate(table.iterrows()):
                score = float(row["fx"])
                parameters = [float(value) for value in row["x"]]
                records.append(dict(
                    schema_version=1, optimizer_id=optimizer, call_id=int(row["call_id"]),
                    score=score if math.isfinite(score) else None,
                    raw_score=None if math.isfinite(score) else str(score),
                    parameters=[value if math.isfinite(value) else None for value in parameters],
                    source="glompo_log.h5", source_dataset=table._v_pathname, source_row=index,
                ))
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("HDF5 changed during export; retry after the manager stops")
    records.sort(key=lambda record: (record["call_id"], record["optimizer_id"], record["source_row"]))
    output = directory / "glompo_history.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(output)
    atomic_json(directory / "glompo_history.meta.json", dict(
        schema_version=1, source="glompo_log.h5", source_sha256=after,
        history_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), count=len(records)))
    return len(records)
