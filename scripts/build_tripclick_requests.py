"""
Build the TripClick request stream used by exp44.

TripClick (Rekabsaz et al., SIGIR 2021; Hugging Face Navid/tripclick) is the
click log of the Trip Database health search engine, January 2013 to October
2020, with a millisecond timestamp per click. Terms: non-commercial research
only; the data and models trained on it must not be shared. Run this on a
machine that holds the licensed copy, and keep every output out of git.

One request is one search: the pair (session, normalised query). Its time is
the first click, and its equivalence class is the document clicked first. Two
requests are equivalent when they led to the same first-clicked document. The
label is used only to score hits, never to decide them or to tune anything.

Usage:  python scripts/build_tripclick_requests.py /content/tripclick/logs /content/work/tripclick_requests.parquet
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd


def main(logs_dir: str, out: str) -> None:
    rows = []
    for fn in sorted(glob.glob(str(Path(logs_dir) / "20*.json"))):
        with open(fn, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    rows.append((int(re.search(r"\d+", d["DateCreated"]).group()),
                                 d["SessionId"], d["Keywords"] or "", d["DocumentId"]))
    df = pd.DataFrame(rows, columns=["ts_ms", "session", "query_raw", "doc_id"])
    df["prompt"] = df.query_raw.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.prompt != ""].sort_values("ts_ms", kind="stable")
    g = df.groupby(["session", "prompt"], sort=False)
    req = g.agg(ts_ms=("ts_ms", "min"), id_set=("doc_id", "first"), n_clicks=("doc_id", "size")).reset_index()
    req = req.sort_values("ts_ms", kind="stable").reset_index(drop=True)
    req["created"] = pd.to_datetime(req.ts_ms, unit="ms")
    req["id_set"] = req.id_set.astype("int64")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    req[["prompt", "id_set", "created", "session", "n_clicks"]].to_parquet(out)
    print(f"{len(df):,} clicks -> {len(req):,} requests; {req.prompt.nunique():,} distinct queries; "
          f"{req.id_set.nunique():,} classes; {req.created.min()} .. {req.created.max()}; "
          f"requests with >1 click: {100 * (req.n_clicks > 1).mean():.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
