"""
Experiment 49 -- slice-based evaluation of bi-encoder failure modes (docs/PREREGISTRATION_round2b.md, E5).

Systems: C1 (cosine), C2 (QQP-trained, current features), C2-diff (C2 + number-mismatch F2, negation F3 and
capitalised-token-mismatch F5 families, fixed a priori).

Part A (QQP): five perturbation slices and a control are built from held-out *duplicate* pairs by deterministic
rules; one side is perturbed, so the pair becomes non-equivalent (the control keeps the meaning). Models are
trained on the other folds (group-disjoint 5-fold CV). Matched coverage: each system's threshold accepts the same
fraction R (0.80, 0.90) of the held-out unperturbed duplicate pairs; the metric is the false-acceptance rate on a
slice. 95% bootstrap CI over question components, resampling base pairs and slice items together.
Part B (external, zero-shot from models trained on all QQP pairs): HEROS (dcml0714/Heros) and PAWS test, with the
meaning-preserving pairs (HEROS Synonym and Typo; PAWS label 1) as the matched-coverage base.

Run: python -m semcache.experiments.exp49_slices
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..data import build_splits, make_group_folds
from ..encoders import get_bi
from ..features import build_bundle
from ..metaclf import make_pipeline
from ._common import banner, encode_frame, hardware_manifest, save_result
from .exp48_targeted_features import FAMILIES, extra_features

C2_GROUPS = ("bi_cosine", "distance", "elementwise", "text")
DIFF_COLS = FAMILIES["F2_numbers"] + FAMILIES["F3_negation"] + FAMILIES["F5_caps"]
SYSTEMS = ("C1", "C2", "C2-diff")
RECALLS = (0.90, 0.80)
B = 300

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
REL = ["today", "tomorrow", "yesterday", "tonight"]
PERIOD = [("this year", "last year"), ("next year", "this year"), ("this week", "next week"),
          ("last week", "this week"), ("this month", "last month"), ("next month", "this month")]
UNIT_PAIRS = [("km", "miles"), ("kilometers", "miles"), ("kilometres", "miles"), ("kg", "pounds"), ("kilograms", "pounds"),
              ("cm", "inches"), ("meters", "feet"), ("metres", "feet"), ("liters", "gallons"), ("litres", "gallons"),
              ("celsius", "fahrenheit"), ("dollars", "euros"), ("usd", "inr"), ("rupees", "dollars"),
              ("hours", "minutes"), ("days", "weeks"), ("years", "months"), ("mb", "gb"), ("grams", "ounces")]
UNIT_SWAP = {}
for a, b in UNIT_PAIRS:
    UNIT_SWAP.setdefault(a, b)
    UNIT_SWAP.setdefault(b, a)
AUX = {"is", "are", "was", "were", "can", "could", "do", "does", "did", "will", "would", "should", "has", "have", "had"}
CAP = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


def _swap_word(text, table, rng):
    toks = text.split(" ")
    hits = [i for i, t in enumerate(toks) if re.sub(r"\W", "", t).lower() in table]
    if not hits:
        return None
    i = hits[int(rng.integers(len(hits)))]
    w = re.sub(r"\W", "", toks[i])
    new = table[w.lower()]
    toks[i] = toks[i].replace(w, new.capitalize() if w[0].isupper() else new)
    return " ".join(toks)


def perturb(kind, text, rng, pool):
    if kind == "number":
        m = [x for x in NUM.finditer(text) if not YEAR.fullmatch(x.group())]
        if not m:
            return None
        x = m[int(rng.integers(len(m)))]
        n = float(x.group())
        cand = [v for v in ({1, 2, 3, 5, 10, 12, 20, 100} | {int(n * 2) + 1, int(n) + 3}) if v != n and v > 0]
        return text[:x.start()] + str(int(rng.choice(cand))) + text[x.end():]
    if kind == "unit":
        return _swap_word(text, UNIT_SWAP, rng)
    if kind == "entity":
        toks = text.split(" ")
        hits = [i for i, t in enumerate(toks) if i > 0 and CAP.fullmatch(re.sub(r"\W", "", t) or "x")]
        if not hits:
            return None
        i = hits[int(rng.integers(len(hits)))]
        old = re.sub(r"\W", "", toks[i])
        new = old
        for _ in range(20):
            new = pool[int(rng.integers(len(pool)))]
            if new.lower() != old.lower():
                break
        toks[i] = toks[i].replace(old, new)
        return " ".join(toks)
    if kind == "time":
        y = YEAR.search(text)
        if y:
            ny = str(int(y.group()) + int(rng.choice([-5, -3, -2, -1, 1, 2, 3, 5])))
            return text[:y.start()] + ny + text[y.end():]
        lower = text.lower()
        for lst in (MONTHS, DAYS, REL):
            present = [w for w in lst if re.search(r"\b" + w + r"\b", lower)]
            if present:
                w = present[int(rng.integers(len(present)))]
                other = [v for v in lst if v != w][int(rng.integers(len(lst) - 1))]
                return re.sub(r"\b" + w + r"\b", other, text, count=1, flags=re.I)
        for a, b in PERIOD:
            if a in lower:
                return re.sub(a, b, text, count=1, flags=re.I)
        return None
    if kind == "negation":
        if re.search(r"\b(not|never)\b|n't", text, re.I):
            return re.sub(r"\s*\b(not|never)\b|n't", "", text, count=1, flags=re.I).replace("  ", " ")
        toks = text.split(" ")
        for i, t in enumerate(toks):
            if re.sub(r"\W", "", t).lower() in AUX:
                toks.insert(i + 1, "not")
                return " ".join(toks)
        return None
    if kind == "control":
        out = re.sub(r"[?.!,]", "", text.lower())
        out = re.sub(r"\s+", " ", out).strip()
        return out if out != text else None
    raise ValueError(kind)


KINDS = ("number", "unit", "entity", "time", "negation", "control")


def feats(qa, qb, E1, E2):
    X2 = build_bundle(list(qa), list(qb), E1, E2, None, groups=C2_GROUPS).matrix(C2_GROUPS)
    ex = extra_features(list(qa), list(qb), {}, 1.0)[DIFF_COLS].to_numpy(np.float32)
    return X2, np.hstack([X2, ex])


def encode(texts):
    return get_bi().model.encode(list(texts), batch_size=256, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def thr_for_recall(base_scores, r):
    return float(np.quantile(base_scores, 1.0 - r))


def far_table(base, slices, groups_base, groups_slice, seed):
    """base[s]: scores of unperturbed duplicates; slices[name][s]: scores of perturbed pairs. FAR + group-bootstrap CI."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups_base)
    rows_b = {g: a.to_numpy() for g, a in pd.Series(np.arange(len(groups_base))).groupby(groups_base)}
    rows_s = {k: {g: a.to_numpy() for g, a in pd.Series(np.arange(len(gr))).groupby(gr)} for k, gr in groups_slice.items()}

    def far_all(idx_b, idx_s):
        res = {}
        for r in RECALLS:
            t = {s: thr_for_recall(base[s][idx_b], r) for s in SYSTEMS}
            for name in slices:
                ii = idx_s[name]
                res[(name, r)] = {s: (float(np.mean(slices[name][s][ii] >= t[s])) if len(ii) else float("nan"))
                                  for s in SYSTEMS}
        return res

    full_idx = {name: np.arange(len(groups_slice[name])) for name in slices}
    pt = far_all(np.arange(len(groups_base)), full_idx)
    reps = {key: {s: [] for s in SYSTEMS} for key in pt}
    for _ in range(B):
        g = rng.choice(ug, len(ug))
        ib = np.concatenate([rows_b[x] for x in g])
        idx_s = {}
        for name in slices:
            parts = [rows_s[name][x] for x in g if x in rows_s[name]]
            idx_s[name] = np.concatenate(parts) if parts else np.array([], int)
        rr = far_all(ib, idx_s)
        for key in rr:
            for s in SYSTEMS:
                reps[key][s].append(rr[key][s])
    out = {name: {} for name in slices}
    for (name, r), p in pt.items():
        ci = {}
        for s in ("C2", "C2-diff"):
            d = np.array(reps[(name, r)][s]) - np.array(reps[(name, r)]["C1"])
            ci[s] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        out[name][r] = {"n": int(len(groups_slice[name])), "far": p,
                        "delta": {s: p[s] - p["C1"] for s in ("C2", "C2-diff")}, "ci": ci}
    return out


def main(seed: int = C.SEED, tag: str = "") -> dict:
    banner("EXP 49 | slice-based evaluation of bi-encoder failure modes")
    t0 = time.time()
    splits = build_splits()
    meta = splits.meta
    enc = encode_frame(meta, "meta30000", with_cross=False)
    y, groups = enc.y, np.asarray(enc.groups)
    qa, qb = meta.question1.tolist(), meta.question2.tolist()
    X2, X2d = feats(qa, qb, enc.e1, enc.e2)
    pool = sorted({w for q in qa + qb for w in CAP.findall(q)})
    rng = np.random.default_rng(seed)

    # ---- perturbed pairs from duplicate rows ------------------------------------------------------
    pos = np.flatnonzero(y == 1)
    gen = {k: {"src": [], "text": []} for k in KINDS}
    examples = {}
    for kind in KINDS:
        for i in pos:
            new = perturb(kind, qb[i], rng, pool)
            if new and new != qb[i]:
                gen[kind]["src"].append(int(i))
                gen[kind]["text"].append(new)
        examples[kind] = [(qb[s], t) for s, t in zip(gen[kind]["src"][:3], gen[kind]["text"][:3])]
        print(f"  slice {kind:9s}: {len(gen[kind]['src'])} perturbed pairs [{time.time() - t0:.0f}s]", flush=True)
    E = {k: encode(v["text"]) if v["text"] else np.zeros((0, enc.e2.shape[1]), np.float32) for k, v in gen.items()}

    # ---- out-of-fold scoring ----------------------------------------------------------------------
    folds = make_group_folds(y, groups, C.CV_FOLDS_META)
    base = {s: np.full(len(y), np.nan) for s in SYSTEMS}
    sl = {k: {s: np.full(len(v["src"]), np.nan) for s in SYSTEMS} for k, v in gen.items()}
    src_idx = {k: np.array(v["src"], int) for k, v in gen.items()}
    for f, (tr, va) in enumerate(folds):
        m2, md = make_pipeline("LightGBM", seed=seed + f), make_pipeline("LightGBM", seed=seed + f)
        m2.fit(X2[tr], y[tr])
        md.fit(X2d[tr], y[tr])
        base["C1"][va] = X2[va, 0]
        base["C2"][va] = m2.predict_proba(X2[va])[:, 1]
        base["C2-diff"][va] = md.predict_proba(X2d[va])[:, 1]
        vaset = set(va.tolist())
        for k in KINDS:
            sel = np.flatnonzero([s in vaset for s in src_idx[k]]) if len(src_idx[k]) else np.array([], int)
            if not len(sel):
                continue
            i = src_idx[k][sel]
            a, b = [qa[j] for j in i], [gen[k]["text"][j] for j in sel]
            Xs, Xsd = feats(a, b, enc.e1[i], E[k][sel])
            sl[k]["C1"][sel] = Xs[:, 0]
            sl[k]["C2"][sel] = m2.predict_proba(Xs)[:, 1]
            sl[k]["C2-diff"][sel] = md.predict_proba(Xsd)[:, 1]
        print(f"  fold {f} scored [{time.time() - t0:.0f}s]", flush=True)
    posb = {s: base[s][pos] for s in SYSTEMS}
    gb = groups[pos]
    slg = {k: groups[src_idx[k]] for k in KINDS}
    partA = far_table(posb, sl, gb, slg, seed)
    for k in ("number", "negation"):
        r = partA[k][0.90]
        print(f"  A {k:9s} R=0.90 FAR C1 {r['far']['C1']:.3f} C2 {r['far']['C2']:.3f} C2-diff {r['far']['C2-diff']:.3f}", flush=True)

    # ---- Part B: external sets, models trained on all QQP pairs -----------------------------------
    full2, fulld = make_pipeline("LightGBM", seed=seed), make_pipeline("LightGBM", seed=seed)
    full2.fit(X2, y)
    fulld.fit(X2d, y)

    def score(a, b):
        Xs, Xsd = feats(a, b, encode(a), encode(b))
        return {"C1": Xs[:, 0], "C2": full2.predict_proba(Xs)[:, 1], "C2-diff": fulld.predict_proba(Xsd)[:, 1]}

    from datasets import load_dataset
    ext = {}
    h = load_dataset("dcml0714/Heros")["train"].to_pandas()
    sc = {v: score(h["Original"].tolist(), h[v].tolist()) for v in ("Synonym", "Typo", "Antonym", "Negation", "Random",
                                                                      "Negation (antonym)")}
    hb = {s: np.concatenate([sc["Synonym"][s], sc["Typo"][s]]) for s in SYSTEMS}
    ext["HEROS"] = {"base": "Synonym+Typo (meaning-preserving)", "slices": {}}
    for v in ("Antonym", "Negation", "Random", "Negation (antonym)", "Synonym", "Typo"):
        ext["HEROS"]["slices"][v] = {str(r): {s: float(np.mean(sc[v][s] >= thr_for_recall(hb[s], r))) for s in SYSTEMS}
                                     for r in RECALLS}
    p = load_dataset("google-research-datasets/paws", "labeled_final")["test"].to_pandas()
    sp = score(p.sentence1.tolist(), p.sentence2.tolist())
    pl = p.label.to_numpy()
    from sklearn.metrics import roc_auc_score
    ext["PAWS"] = {"n": int(len(p)), "positive_rate": float(pl.mean()),
                   "auc": {s: float(roc_auc_score(pl, sp[s])) for s in SYSTEMS},
                   "far_at_recall": {str(r): {s: float(np.mean(sp[s][pl == 0] >= thr_for_recall(sp[s][pl == 1], r)))
                                               for s in SYSTEMS} for r in RECALLS}}
    print("  B PAWS AUC", ext["PAWS"]["auc"], flush=True)

    out = {"preregistration": "docs/PREREGISTRATION_round2b.md section E5", "partA_qqp": partA, "partB_external": ext,
           "slice_sizes": {k: len(v["src"]) for k, v in gen.items()}, "examples": examples, "diff_columns": DIFF_COLS,
           "manifest": hardware_manifest({"bootstrap": B, "recalls": list(RECALLS)}, seed=seed)}
    save_result(out, "exp49_slices", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    main(tag=ap.parse_args().tag)
