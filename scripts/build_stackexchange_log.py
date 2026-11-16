"""Build a chronological query stream from a Stack Exchange data dump.

Each question title is one request, ordered by creation time. Two questions are
equivalent when a human closed one as a duplicate of the other (PostLinks,
LinkTypeId 3); equivalence classes are the connected components of those links.
Questions without a duplicate link are singleton classes.

Usage: python scripts/build_stackexchange_log.py DUMP_DIR OUT.parquet
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for _, el in ET.iterparse(src / "Posts.xml", events=("end",)):
    if el.tag == "row" and el.get("PostTypeId") == "1":
        rows.append((int(el.get("Id")), el.get("CreationDate"), el.get("Title") or ""))
    el.clear()
q = pd.DataFrame(rows, columns=["post_id", "created", "prompt"])
q["created"] = pd.to_datetime(q["created"])
ids = set(q.post_id)

parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
n_links = 0
for _, el in ET.iterparse(src / "PostLinks.xml", events=("end",)):
    if el.tag == "row" and el.get("LinkTypeId") == "3":
        a, b = int(el.get("PostId")), int(el.get("RelatedPostId"))
        if a in ids and b in ids and a != b:
            parent[find(a)] = find(b)
            n_links += 1
    el.clear()
q["id_set"] = [find(i) for i in q.post_id]
q = q[q.prompt.str.len() > 0].sort_values(["created", "post_id"]).reset_index(drop=True)
sizes = q.groupby("id_set").size()
print(f"questions {len(q)}, duplicate links used {n_links}, classes {len(sizes)}, "
      f"classes with >1 question {(sizes > 1).sum()}, questions in such classes {int(sizes[sizes > 1].sum())}")
print("span", q.created.min(), "->", q.created.max(), "| repeat rate %.1f%%" % (100 * (1 - len(sizes) / len(q))))
print("exact duplicate titles", int(q.prompt.duplicated().sum()), "| mean title chars %.0f" % q.prompt.str.len().mean())
q.to_parquet(out, index=False)
