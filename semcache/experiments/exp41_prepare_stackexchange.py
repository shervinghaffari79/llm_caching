"""
Experiment 41 -- prepare chronological Stack Exchange question streams (remote only).

For each site, the container downloads the public data dump from archive.org
(CC BY-SA), extracts Posts.xml and PostLinks.xml, and builds a stream of
question titles in creation order. Two questions are equivalent when a human
closed one as a duplicate of the other (LinkTypeId 3); classes are connected
components of those links. A diagnostic then asks how retrievable the labelled
duplicates are for a bi-encoder cache: on the oldest DIAG_N questions, the
cosine similarity of same-class pairs and the pair precision/recall at the
serving thresholds. Streams are written to data/external/se_<site>.parquet,
and with --encode SITE the titles of the chosen rows are encoded once and
stored as data/external/se_<site>_vectors.npz. Nothing is written to the
user's machine; only the volume and results/raw/revision/exp41_*.json.

Run: python -m semcache.experiments.exp41_prepare_stackexchange --sites unix superuser
     python -m semcache.experiments.exp41_prepare_stackexchange --encode unix --rows 0 100000
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C

URL = "https://archive.org/download/stackexchange/{site}.7z"
ARCHIVE = {"askubuntu": "askubuntu.com", "unix": "unix.stackexchange.com", "superuser": "superuser.com",
           "serverfault": "serverfault.com", "gaming": "gaming.stackexchange.com",
           "stats": "stats.stackexchange.com", "english": "english.stackexchange.com"}
EXT = C.DATA_DIR / "external"
DIAG_N = 20_000
THRESHOLDS = (0.7, 0.8, 0.85, 0.895, 0.93, 0.95)


def _fetch(site: str, work: Path) -> Path:
    dest = work / f"{site}.7z"
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(URL.format(site=ARCHIVE[site]), dest)
            break
        except Exception as exc:                      # retry a transient network failure
            last = exc
            time.sleep(5)
    else:
        raise RuntimeError(f"download failed for {site}: {last}")
    print(f"  {site}: downloaded {dest.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)
    return dest


def _extract(archive: Path, out: Path) -> None:
    if not shutil.which("7z"):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "p7zip-full"], check=True)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["7z", "x", "-y", f"-o{out}", str(archive), "Posts.xml", "PostLinks.xml"],
                   check=True, stdout=subprocess.DEVNULL)


def _stream(raw: Path) -> pd.DataFrame:
    rows = []
    for _, el in ET.iterparse(raw / "Posts.xml", events=("end",)):
        if el.tag == "row" and el.get("PostTypeId") == "1":
            rows.append((int(el.get("Id")), el.get("CreationDate"), el.get("Title") or ""))
        el.clear()
    q = pd.DataFrame(rows, columns=["post_id", "created", "prompt"])
    q["created"] = pd.to_datetime(q["created"])
    ids = set(q.post_id)
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, el in ET.iterparse(raw / "PostLinks.xml", events=("end",)):
        if el.tag == "row" and el.get("LinkTypeId") == "3":
            a, b = int(el.get("PostId")), int(el.get("RelatedPostId"))
            if a in ids and b in ids and a != b:
                parent[find(a)] = find(b)
        el.clear()
    q["id_set"] = [find(i) for i in q.post_id]
    return q[q.prompt.str.len() > 0].sort_values(["created", "post_id"]).reset_index(drop=True)


def _diagnostic(q: pd.DataFrame) -> dict:
    from ..encoders import get_bi
    b = q.iloc[:DIAG_N]
    V = get_bi().model.encode(b.prompt.tolist(), batch_size=256, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    g = b.id_set.to_numpy()
    iu = np.triu_indices(len(b), 1)
    S = V @ V.T
    s, m = S[iu], (g[:, None] == g[None, :])[iu]
    out = {"pairs_same_class": int(m.sum()),
           "dup_cosine_quantiles_10_25_50_75_90": np.round(np.quantile(s[m], [.1, .25, .5, .75, .9]), 3).tolist(),
           "at_threshold": {}}
    for t in THRESHOLDS:
        hit = s >= t
        out["at_threshold"][str(t)] = {"pairs": int(hit.sum()),
                                      "pair_precision": float(m[hit].mean()) if hit.any() else None,
                                      "recall_of_duplicates": float(hit[m].mean()) if m.any() else None}
    return out


def prepare(site: str) -> dict:
    work = Path("/tmp/se") / site
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    arch = _fetch(site, work)
    _extract(arch, work / "x")
    q = _stream(work / "x")
    sizes = q.groupby("id_set").size()
    info = {"site": site, "questions": int(len(q)), "classes": int(len(sizes)),
            "classes_gt1": int((sizes > 1).sum()), "questions_in_gt1": int(sizes[sizes > 1].sum()),
            "repeat_rate_pct": 100 * (1 - len(sizes) / len(q)),
            "first": str(q.created.iloc[0]), "last": str(q.created.iloc[-1]),
            "mean_title_chars": float(q.prompt.str.len().mean())}
    EXT.mkdir(parents=True, exist_ok=True)
    q.to_parquet(EXT / f"se_{site}.parquet", index=False)
    info["diagnostic_oldest_%d" % DIAG_N] = _diagnostic(q)
    shutil.rmtree(work, ignore_errors=True)
    print(f"  {site}: {json.dumps(info)}", flush=True)
    return info


def encode(site: str, lo: int, hi: int) -> None:
    from ..encoders import get_bi
    q = pd.read_parquet(EXT / f"se_{site}.parquet")
    hi = hi or len(q)
    titles = sorted(set(q.prompt.iloc[lo:hi].astype(str)))
    t0 = time.time()
    V = get_bi().model.encode(titles, batch_size=256, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.savez(EXT / f"se_{site}_vectors.npz", queries=np.array(titles, dtype=object), vectors=V)
    print(f"  encoded {len(titles)} titles of rows {lo}-{hi} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=[])
    ap.add_argument("--encode", default=None)
    ap.add_argument("--rows", nargs=2, type=int, default=[0, 0])
    a = ap.parse_args()
    from ._common import banner, hardware_manifest
    banner("EXP 41 | Stack Exchange stream preparation")
    if a.sites:
        out = {"manifest": hardware_manifest({"sites": a.sites}), "sites": {}}
        for s in a.sites:
            out["sites"][s] = prepare(s)
            p = C.RAW_DIR / "revision" / "exp41_se_candidates.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=1), encoding="utf8")
    if a.encode:
        encode(a.encode, *a.rows)
