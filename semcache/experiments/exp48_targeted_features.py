"""
Experiment 48 -- cheap features that target what the cosine misses (docs/PREREGISTRATION_round2b.md, E4).

Development (feature-family selection) uses only the QQP ablation split. Confirmation runs once on (Q) the QQP meta
set under group-disjoint 5-fold CV and (S) transfer to the search log (models trained on QQP only, test pairs are the
window-local nearest earlier neighbours of rows 60,000-79,999). No label from the search log is used.

Run: python -m semcache.experiments.exp48_targeted_features
"""
from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import config as C
from ..data import build_splits, make_group_folds
from ..features import build_bundle
from ..metaclf import make_pipeline
from ._common import banner, encode_frame, hardware_manifest, save_result
from .exp45_cheap_feature_regimes import compare, metrics

C2_GROUPS = ("bi_cosine", "distance", "elementwise", "text")
TOK = re.compile(r"[a-z0-9']+")
NUM = re.compile(r"\d+(?:\.\d+)?")
CAP = re.compile(r"\b[A-Z][A-Za-z]+")
NEG = {"not", "no", "never", "cannot", "without", "none", "nothing", "neither", "nor", "nobody", "nowhere"}
WH = {"what": 0, "how": 1, "why": 2, "when": 3, "where": 4, "who": 5, "which": 6, "can": 7, "is": 8, "are": 9,
      "do": 10, "does": 11, "should": 12, "will": 13, "would": 14, "could": 15}
FAMILIES = {"F1_idf": ["idf_jac", "idf_diff_max", "idf_diff_mass"], "F2_numbers": ["n_a", "n_b", "n_eq", "n_diff"],
            "F3_negation": ["neg_a", "neg_b", "neg_xor"], "F4_qtype": ["wh_a", "wh_b", "wh_eq"],
            "F5_caps": ["cap_jac", "cap_diff"], "F6_char3": ["c3_jac"]}
ALL_COLS = [c for cols in FAMILIES.values() for c in cols]
B_DEV = 300


def idf_table(texts):
    df = Counter()
    for t in texts:
        df.update(set(TOK.findall(t.lower())))
    n = len(texts)
    return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}, math.log(n + 1) + 1.0


def _c3(s):
    s = " " + s.lower() + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def extra_features(qa, qb, idf, default):
    rows = []
    for a, b in zip(qa, qb):
        ta, tb = set(TOK.findall(a.lower())), set(TOK.findall(b.lower()))
        w = lambda t: idf.get(t, default)
        un, it, sd = ta | tb, ta & tb, ta ^ tb
        mass_u = sum(map(w, un)) or 1.0
        na, nb = set(NUM.findall(a)), set(NUM.findall(b))
        ga, gb = bool(ta & NEG) or "n't" in a.lower(), bool(tb & NEG) or "n't" in b.lower()
        fa, fb = a.lower().split()[:1], b.lower().split()[:1]
        wa, wb = WH.get(fa[0], 99) if fa else 99, WH.get(fb[0], 99) if fb else 99
        ca, cb = set(CAP.findall(" ".join(a.split()[1:]))), set(CAP.findall(" ".join(b.split()[1:])))
        xa, xb = _c3(a), _c3(b)
        rows.append([sum(map(w, it)) / mass_u, max(map(w, sd)) if sd else 0.0, sum(map(w, sd)) / mass_u,
                     len(na), len(nb), float(na == nb), len(na ^ nb),
                     float(ga), float(gb), float(ga != gb), wa, wb, float(wa == wb),
                     len(ca & cb) / max(1, len(ca | cb)), len(ca ^ cb), len(xa & xb) / max(1, len(xa | xb))])
    return pd.DataFrame(rows, columns=ALL_COLS)


def band_auc(y, s, band):
    return roc_auc_score(y[band], s[band]) if len(np.unique(y[band])) == 2 else float("nan")


def cv_scores(X2, y, folds, qa, qb, cols, seed):
    """Out-of-fold scores of LightGBM on C2 features plus the extra columns `cols` (IDF fitted on training rows only)."""
    oof = np.full(len(y), np.nan)
    for i, (tr, va) in enumerate(folds):
        parts_tr, parts_va = [X2[tr]], [X2[va]]
        if cols:
            idf, dflt = idf_table([qa[k] for k in tr] + [qb[k] for k in tr])
            parts_tr.append(extra_features([qa[k] for k in tr], [qb[k] for k in tr], idf, dflt)[cols].to_numpy(np.float32))
            parts_va.append(extra_features([qa[k] for k in va], [qb[k] for k in va], idf, dflt)[cols].to_numpy(np.float32))
        pipe = make_pipeline("LightGBM", seed=seed + i)
        pipe.fit(np.hstack(parts_tr), y[tr])
        oof[va] = pipe.predict_proba(np.hstack(parts_va))[:, 1]
    return oof


def boot_delta_band(y, groups, cos, s_new, s_old, seed):
    band = cos >= np.quantile(cos, 0.8)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rows_of = pd.Series(np.arange(len(y))).groupby(groups).apply(lambda s: s.to_numpy())
    d = []
    for _ in range(B_DEV):
        idx = np.concatenate([rows_of[g] for g in rng.choice(ug, len(ug))])
        d.append(band_auc(y[idx], s_new[idx], band[idx]) - band_auc(y[idx], s_old[idx], band[idx]))
    point = band_auc(y, s_new, band) - band_auc(y, s_old, band)
    return point, [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def search_log_pairs(start=60_000, n=20_000, audit=40_000, floor=0.6):
    from .exp46_audit_budget import DATA, VECS
    df = pd.read_parquet(DATA)
    prompts, ids = df.prompt.astype(str).to_numpy(), df.id_set.to_numpy()
    z = np.load(VECS, allow_pickle=True)
    vecs = dict(zip(z["queries"].tolist(), z["vectors"]))
    V = np.stack([vecs[q] for q in prompts[start:start + n]]).astype(np.float32)
    nn, sim = np.full(n, -1), np.full(n, -1.0, dtype=np.float32)
    for a in range(0, n, 2048):
        b = min(n, a + 2048)
        S = V[a:b] @ V[:b].T
        S[np.arange(b)[None, :] >= np.arange(a, b)[:, None]] = -1.0
        j = S.argmax(axis=1)
        nn[a:b], sim[a:b] = j, S[np.arange(b - a), j]
    keep = np.flatnonzero((nn >= 0) & (sim >= floor))
    i, j = keep, nn[keep]
    ref = [prompts[k] for k in range(audit)]                        # unlabelled reference text, rows 0-39,999
    return {"qa": [prompts[start + k] for k in i], "qb": [prompts[start + k] for k in j], "e1": V[i], "e2": V[j],
            "y": (ids[start + i] == ids[start + j]).astype(int), "groups": ids[start + i], "ref": ref}


def main(seed: int = C.SEED, tag: str = "") -> dict:
    banner("EXP 48 | cheap features that target the confusable band")
    t0 = time.time()
    splits = build_splits()
    out = {"preregistration": "docs/PREREGISTRATION_round2b.md section E4"}

    # ---- development: QQP ablation split only -----------------------------------------------------
    abl = splits.ablation
    enc = encode_frame(abl, "ablation3000", with_cross=False)
    y, groups = enc.y, np.asarray(enc.groups)
    X2 = enc.bundle.matrix(C2_GROUPS)
    cos = X2[:, 0]
    qa, qb = abl.question1.tolist(), abl.question2.tolist()
    folds = make_group_folds(y, groups, 5)
    base = cv_scores(X2, y, folds, qa, qb, [], seed)
    dev, kept = {}, []
    for fam, cols in FAMILIES.items():
        s = cv_scores(X2, y, folds, qa, qb, cols, seed)
        d, ci = boot_delta_band(y, groups, cos, s, base, seed)
        keep = d >= 0.01 and ci[0] > 0
        dev[fam] = {"delta_top_quintile_auc": d, "ci": ci, "kept": bool(keep)}
        kept += cols if keep else []
        print(f"  dev {fam:12s} top-quintile AUC {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {'KEEP' if keep else 'drop'} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    out["development"] = {"pairs": int(len(y)), "families": dev, "kept_columns": kept}
    if not kept:
        out["decision"] = "no family kept on the development split; confirmation not run"
        save_result(out, "exp48_targeted_features", tag=tag)
        return out

    # ---- confirmation Q: QQP meta set, group-disjoint 5-fold CV -------------------------------------
    meta = splits.meta
    enc = encode_frame(meta, "meta30000", with_cross=False)
    y, groups = enc.y, np.asarray(enc.groups)
    X2m = enc.bundle.matrix(C2_GROUPS)
    cosm = X2m[:, 0]
    qam, qbm = meta.question1.tolist(), meta.question2.tolist()
    foldsm = make_group_folds(y, groups, C.CV_FOLDS_META)
    s_c2 = cv_scores(X2m, y, foldsm, qam, qbm, [], seed)
    s_c2p = cv_scores(X2m, y, foldsm, qam, qbm, kept, seed)
    out["Q"] = {"C1_vs_C2": compare(y, groups, cosm, s_c2, seed), "C1_vs_C2plus": compare(y, groups, cosm, s_c2p, seed),
                "C2_vs_C2plus": compare(y, groups, s_c2, s_c2p, seed)}
    r = out["Q"]["C1_vs_C2plus"]["p_at_r90"]
    print(f"  Q  P@R90 C1 {r['C1']:.3f} -> C2+ {r['C2']:.3f} ({r['delta']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]) "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # ---- confirmation S: transfer to the search log ----------------------------------------------
    idf_q, dq = idf_table(qam + qbm)
    Xtr = np.hstack([X2m, extra_features(qam, qbm, idf_q, dq)[kept].to_numpy(np.float32)])
    pipe_c2, pipe_c2p = make_pipeline("LightGBM", seed=seed), make_pipeline("LightGBM", seed=seed)
    pipe_c2.fit(X2m, y)
    pipe_c2p.fit(Xtr, y)
    S = search_log_pairs()
    idf_s, ds = idf_table(S["ref"])
    Xs2 = build_bundle(S["qa"], S["qb"], S["e1"], S["e2"], None, groups=C2_GROUPS).matrix(C2_GROUPS)
    Xsp = np.hstack([Xs2, extra_features(S["qa"], S["qb"], idf_s, ds)[kept].to_numpy(np.float32)])
    ys, gs, coss = S["y"], np.asarray(S["groups"]), Xs2[:, 0]
    sc2, sc2p = pipe_c2.predict_proba(Xs2)[:, 1], pipe_c2p.predict_proba(Xsp)[:, 1]
    out["S"] = {"pairs": int(len(ys)), "positive_rate": float(ys.mean()),
                "C1_vs_C2": compare(ys, gs, coss, sc2, seed), "C1_vs_C2plus": compare(ys, gs, coss, sc2p, seed),
                "C2_vs_C2plus": compare(ys, gs, sc2, sc2p, seed)}
    r = out["S"]["C1_vs_C2plus"]["p_at_r90"]
    print(f"  S  P@R90 C1 {r['C1']:.3f} -> C2+ {r['C2']:.3f} ({r['delta']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]) "
          f"[{time.time() - t0:.0f}s]", flush=True)

    q, s = out["Q"]["C1_vs_C2plus"]["p_at_r90"], out["S"]["C1_vs_C2plus"]["p_at_r90"]
    out["decision"] = {"Q_gain_ge_5pts_ci_above_0": bool(q["delta"] >= 0.05 and q["ci"][0] > 0),
                       "S_gain_ge_3pts": bool(s["delta"] >= 0.03), "substantive_leap": bool(
                           q["delta"] >= 0.05 and q["ci"][0] > 0 and s["delta"] >= 0.03)}
    print("  decision:", out["decision"], flush=True)
    out["manifest"] = hardware_manifest({"bootstrap_dev": B_DEV, "kept": kept}, seed=seed)
    save_result(out, "exp48_targeted_features", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    main(tag=ap.parse_args().tag)
