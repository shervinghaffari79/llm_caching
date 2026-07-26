"""
QQP / STS loading with an explicit, auditable answer to the duplication and
leakage questions raised in review.

Three distinct problems are handled here and kept separate, because they have
different fixes and different effects on the reported F1:

1. **Duplicate rows.** The same (q1, q2) pair, or its mirror (q2, q1), occurs
   more than once in QQP. Left in place these inflate scores whenever one copy
   lands in train and another in validation.
2. **Question overlap across folds.** QQP is a *graph*: a popular question
   appears in dozens of pairs. Even after row de-duplication, plain
   `StratifiedKFold` puts pairs sharing a question on both sides of the split.
   The model can then memorise "this question is usually a duplicate" instead
   of learning a similarity function. The fix is to split by connected
   component of the question co-occurrence graph (`make_group_folds`).
3. **Overlap between the 30k meta-classifier set and the 3k ablation set.**
   The manuscript describes the ablation subset as held out. Held out at the
   *pair* level is not enough: it must be held out at the *component* level,
   otherwise questions seen during meta-classifier training reappear in the
   ablation sample. `build_splits` enforces component-level disjointness.

Every one of these is quantified in `audit_report()` so the paper can state
what was found rather than asserting that nothing was wrong.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


# --------------------------------------------------------------------------
# Normalisation and identity
# --------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Casefold, strip punctuation and collapse whitespace.

    Used only for *identity* decisions (dedup / graph nodes), never for
    feature extraction -- features always see the original strings.
    """
    s = (s or "").strip().lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def qid_of(s: str) -> str:
    return hashlib.blake2b(normalize_text(s).encode("utf8"), digest_size=8).hexdigest()


def pair_key(a: str, b: str) -> str:
    """Order-insensitive key so (q1,q2) and (q2,q1) collide."""
    ka, kb = qid_of(a), qid_of(b)
    lo, hi = (ka, kb) if ka <= kb else (kb, ka)
    return f"{lo}:{hi}"


# --------------------------------------------------------------------------
# Union-find over the question co-occurrence graph
# --------------------------------------------------------------------------
class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------
def load_qqp_raw(force: bool = False) -> pd.DataFrame:
    """Load the labelled QQP training split (363,846 pairs), cached to parquet."""
    cache = C.DATA_DIR / "qqp_train.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
    else:
        from datasets import load_dataset
        df = load_dataset("glue", "qqp", split="train").to_pandas()
        df.to_parquet(cache, index=False)
    df = df.rename(columns={"label": "is_duplicate"})
    df["question1"] = df["question1"].astype(str)
    df["question2"] = df["question2"].astype(str)
    return df[["question1", "question2", "is_duplicate", "idx"]].copy()


def load_stsb() -> dict[str, pd.DataFrame]:
    """STS-Benchmark train/dev/test with scores in [0,5]."""
    from datasets import load_dataset
    out = {}
    for hf_split, name in (("train", "train"), ("validation", "dev"), ("test", "test")):
        d = load_dataset("sentence-transformers/stsb", split=hf_split).to_pandas()
        d = d.rename(columns={"sentence1": "question1", "sentence2": "question2"})
        # This mirror publishes `score` already scaled to [0,1]; restore 0-5.
        s = d["score"].astype(float)
        d["sts_score"] = s * 5.0 if s.max() <= 1.0 + 1e-9 else s
        d["is_duplicate"] = (d["sts_score"] >= 2.5).astype(int)
        out[name] = d[["question1", "question2", "sts_score", "is_duplicate"]].copy()
    return out


# --------------------------------------------------------------------------
# Duplicate / leakage audit
# --------------------------------------------------------------------------
@dataclass
class AuditReport:
    n_rows_raw: int
    n_exact_duplicate_rows: int
    n_mirrored_duplicate_rows: int
    n_self_pairs: int
    n_label_conflicts: int
    n_rows_clean: int
    n_unique_questions: int
    n_components: int
    largest_component: int
    pct_rows_in_top1pct_components: float

    def as_dict(self) -> dict:
        return asdict(self)


def audit_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, AuditReport]:
    """Remove duplicated rows, mirrors, self-pairs and label conflicts.

    Returns the cleaned frame (with `pkey` and `component` columns) plus a
    report giving the exact counts, so the manuscript can cite them.
    """
    n_raw = len(df)
    df = df.copy()
    df["q1n"] = df["question1"].map(normalize_text)
    df["q2n"] = df["question2"].map(normalize_text)
    df["pkey"] = [pair_key(a, b) for a, b in zip(df["question1"], df["question2"])]

    # Self-pairs: identical text on both sides -- trivially duplicate, and a
    # free win for any similarity model. Removed.
    self_mask = df["q1n"] == df["q2n"]
    n_self = int(self_mask.sum())
    df = df[~self_mask]

    # Label conflicts: same pair annotated both 1 and 0.
    grp = df.groupby("pkey")["is_duplicate"].nunique()
    conflict_keys = set(grp[grp > 1].index)
    n_conflicts = int(df["pkey"].isin(conflict_keys).sum())
    df = df[~df["pkey"].isin(conflict_keys)]

    # Exact (same order) vs mirrored (reversed order) repeats.
    ordered_key = df["q1n"] + " ||| " + df["q2n"]
    n_exact = int(ordered_key.duplicated().sum())
    df = df.assign(_ok=ordered_key)
    before = len(df)
    df = df.drop_duplicates(subset="pkey", keep="first")
    n_total_dupe = before - len(df)
    n_mirror = max(0, n_total_dupe - n_exact)
    df = df.drop(columns=["_ok"])

    # Component structure of the question co-occurrence graph.
    uf = UnionFind()
    q1id = df["question1"].map(qid_of)
    q2id = df["question2"].map(qid_of)
    for a, b in zip(q1id, q2id):
        uf.union(a, b)
    comp = np.array([uf.find(a) for a in q1id])
    df = df.assign(q1id=q1id.values, q2id=q2id.values, component=comp)

    sizes = pd.Series(comp).value_counts()
    n_top = max(1, int(np.ceil(0.01 * len(sizes))))
    pct_top = float(sizes.iloc[:n_top].sum() / len(df) * 100)

    rep = AuditReport(
        n_rows_raw=n_raw,
        n_exact_duplicate_rows=n_exact,
        n_mirrored_duplicate_rows=n_mirror,
        n_self_pairs=n_self,
        n_label_conflicts=n_conflicts,
        n_rows_clean=len(df),
        n_unique_questions=int(pd.unique(np.concatenate([q1id.values, q2id.values])).size),
        n_components=int(sizes.size),
        largest_component=int(sizes.iloc[0]),
        pct_rows_in_top1pct_components=pct_top,
    )
    return df.reset_index(drop=True), rep


# --------------------------------------------------------------------------
# Split construction
# --------------------------------------------------------------------------
@dataclass
class Splits:
    meta: pd.DataFrame           # 30k balanced  -> Table 4
    ablation: pd.DataFrame       # 3k, 60/40     -> Table 7 (component-disjoint from meta)
    ablation_bal: pd.DataFrame   # 3k, 50/50     -> stability re-check
    stream_pool: pd.DataFrame    # for the dynamic workload simulator
    audit: AuditReport
    provenance: dict


def _balanced_sample(df: pd.DataFrame, n_pos: int, n_neg: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    pos = df[df.is_duplicate == 1]
    neg = df[df.is_duplicate == 0]
    if len(pos) < n_pos or len(neg) < n_neg:
        raise ValueError(f"pool too small: have {len(pos)}/{len(neg)}, "
                         f"need {n_pos}/{n_neg}")
    ip = rng.choice(len(pos), n_pos, replace=False)
    ineg = rng.choice(len(neg), n_neg, replace=False)
    out = pd.concat([pos.iloc[ip], neg.iloc[ineg]])
    return out.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


def build_splits(seed: int = C.SEED, force: bool = False) -> Splits:
    """Build component-disjoint meta / ablation / stream pools.

    The ablation pool is carved out of components that contribute **no** pair
    to the meta-classifier set, so no question text is shared. This is
    stricter than the pair-level hold-out described in the manuscript and is
    what makes the Table 7 numbers independent of Table 4.
    """
    cache = C.DATA_DIR / f"splits_seed{seed}.pkl"
    if cache.exists() and not force:
        import pickle
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    rng = np.random.default_rng(seed)
    clean, audit = audit_and_clean(load_qqp_raw())

    # Partition components into three disjoint buckets, sized by pair count so
    # each bucket can satisfy its class quotas.
    comps = clean.groupby("component").agg(
        n=("is_duplicate", "size"), n_pos=("is_duplicate", "sum")).reset_index()
    comps = comps.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Headroom multipliers: QQP is ~37% positive, so a bucket that must yield
    # a *balanced* sample needs roughly 3x the pairs it will finally use.
    need = {"meta": 3 * C.N_META, "abl": 6 * C.N_ABLATION, "stream": 60_000}
    bucket_of: dict[str, str] = {}
    got = {k: 0 for k in need}
    order = ["meta", "abl", "stream"]
    comp_ids = comps["component"].to_numpy()
    comp_n = comps["n"].to_numpy()
    for cid, cn in zip(comp_ids, comp_n):
        for b in order:
            if got[b] < need[b]:
                bucket_of[cid] = b
                got[b] += cn
                break
        else:
            bucket_of[cid] = "stream"
            got["stream"] += cn

    clean = clean.assign(bucket=clean["component"].map(bucket_of))
    pool = {b: clean[clean.bucket == b].reset_index(drop=True) for b in order}

    meta = _balanced_sample(pool["meta"], C.N_META // 2, C.N_META // 2, rng)
    n_pos_abl = int(round(0.60 * C.N_ABLATION))
    ablation = _balanced_sample(pool["abl"], n_pos_abl, C.N_ABLATION - n_pos_abl, rng)
    # Balanced re-check drawn from the same bucket but disjoint pairs.
    remaining = pool["abl"][~pool["abl"].pkey.isin(set(ablation.pkey))]
    ablation_bal = _balanced_sample(remaining, C.N_ABLATION_BALANCED // 2,
                                    C.N_ABLATION_BALANCED // 2, rng)

    splits = Splits(
        meta=meta, ablation=ablation, ablation_bal=ablation_bal,
        stream_pool=pool["stream"], audit=audit,
        provenance={
            "seed": seed,
            "meta_components": int(meta.component.nunique()),
            "ablation_components": int(ablation.component.nunique()),
            "component_overlap_meta_ablation": len(
                set(meta.component) & set(ablation.component)),
            "question_overlap_meta_ablation": len(
                (set(meta.q1id) | set(meta.q2id)) &
                (set(ablation.q1id) | set(ablation.q2id))),
            "pair_overlap_meta_ablation": len(set(meta.pkey) & set(ablation.pkey)),
            "ablation_duplicate_ratio": float(ablation.is_duplicate.mean()),
            "meta_duplicate_ratio": float(meta.is_duplicate.mean()),
        },
    )
    import pickle
    with open(cache, "wb") as fh:
        pickle.dump(splits, fh)
    return splits


# --------------------------------------------------------------------------
# Fold construction
# --------------------------------------------------------------------------
def make_stratified_folds(y: np.ndarray, n_splits: int, seed: int = C.SEED):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed).split(np.zeros(len(y)), y))


def make_group_folds(y: np.ndarray, groups: np.ndarray, n_splits: int,
                     seed: int = C.SEED):
    """Stratified *and* group-disjoint folds.

    `StratifiedGroupKFold` keeps every question component wholly inside one
    fold while holding the duplicate ratio roughly constant, so validation
    scores measure generalisation to unseen questions rather than recall of
    memorised ones.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed
                                     ).split(np.zeros(len(y)), y, groups))


def fold_overlap_stats(df: pd.DataFrame, folds) -> list[dict]:
    """How many questions/components a fold shares with its own training part."""
    out = []
    qids = (df["q1id"].values, df["q2id"].values)
    comps = df["component"].values
    for i, (tr, va) in enumerate(folds):
        q_tr = set(qids[0][tr]) | set(qids[1][tr])
        q_va = set(qids[0][va]) | set(qids[1][va])
        c_tr, c_va = set(comps[tr]), set(comps[va])
        out.append({
            "fold": i,
            "n_train": len(tr), "n_val": len(va),
            "shared_questions": len(q_tr & q_va),
            "pct_val_questions_seen": 100.0 * len(q_tr & q_va) / max(1, len(q_va)),
            "shared_components": len(c_tr & c_va),
            "val_duplicate_ratio": float(df["is_duplicate"].values[va].mean()),
        })
    return out


def save_audit(splits: Splits, path: Path | None = None) -> Path:
    path = path or (C.RAW_DIR / "data_audit.json")
    payload = {"audit": splits.audit.as_dict(), "provenance": splits.provenance,
               "sizes": {"meta": len(splits.meta), "ablation": len(splits.ablation),
                         "ablation_balanced": len(splits.ablation_bal),
                         "stream_pool": len(splits.stream_pool)}}
    path.write_text(json.dumps(payload, indent=2))
    return path
