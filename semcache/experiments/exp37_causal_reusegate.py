"""
Experiment 37 -- ReuseGate (label-free Learned[match]) under a strictly causal
layout of the real query log, and with a smaller or empty reference corpus.

Why: the log carries no timestamps (it is a clustered sample of ORCAS queries,
and the row order is arbitrary), but experiment 36 placed its reference and
tuning data at row indices *after* the evaluation windows. A deployment can
only learn from the past, so this experiment re-runs the evaluation with every
data source preceding the data it informs, in row order:

  reference corpus   rows      0 -  49,999   (term frequencies, neighbours, PCA)
  tuning windows     rows 50,000 -  99,999   (five windows of 10,000: classifier
                                              training, threshold selection)
  evaluation windows rows 100,000 - 149,999  (five windows of 10,000)

Nothing is changed in the method: same features, same label-free target, same
LightGBM settings, same half-life (4 x capacity), same threshold grid, same
capacities, same selection rule (per false-hit price lambda, the setting with
the highest tuning score for every family). The evaluation rows were used in
earlier design experiments (33-35, as tuning/test windows), so they are not
fresh data; the point of this run is the causal layout and its sensitivity to
the reference size, not a new held-out estimate.

Reference-size sweep: R in {0, 1,000, 10,000, 50,000} queries taken from the
start of the reference block. R = 0 is a cold start: every reference-derived
feature is constant, and the classifier sees text features and the history
within the window only. The classifier is re-fitted for every R.

Pre-set tests (paired t-tests over the five evaluation windows, each averaged
over capacities 250-4,000; one Holm family, R = 50,000):
  H1c  Learned[match] - W-TinyLFU at lambda = 0, 1, 2
  H2c  Learned[match] - LRU and - LFU at lambda = 1
Secondary (second Holm family): Learned[match] at R < 50,000 against
W-TinyLFU at lambda = 1, and at R = 50,000 for lambda = 5, 10.
Results are reported whatever their sign.

Run: python -m semcache.experiments.exp37_causal_reusegate --workers 30
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ..stats import holm, paired_seed_test, seed_ci
from ._common import REVISION_DIR, banner, hardware_manifest, save_result
from . import exp34_recurrence_predictability as E34
from . import exp36_learned_admission_final as E36
from .exp30_sd_redesign import LOG, _vectors, ensure_log_vectors, log_window

REF = (0, 50_000)
TUNE = tuple((s, 10_000) for s in range(50_000, 100_000, 10_000))
EVAL = tuple((s, 10_000) for s in range(100_000, 150_000, 10_000))
CAPS = E36.LOG_CAPS
TAUS = E36.TAUS
LAMBDAS = E36.LAMBDAS
REF_SIZES = (0, 1_000, 10_000, 50_000)
BASE = ("LRU", "LFU", "SphereLFU", "WTinyLFU")
LEARNED = "Learned[match]"
WITH_GROUP = False         # also evaluate ReuseGate-R (group-label variant, labeled tuning traffic) at the largest reference size
VARIANTS = {"SOLAR-E": (0, 1), "SOLAR-A": (0, 1, 2, 3), "SOLAR": (0, 1, 2, 3)}   # hyper-parameter grid indices (policies_extra.SOLAR_VARIANTS)
EXTRA_BASE = ("LeCaR", "SemanticDecay_fixed", "FIFO", "ARC", "S3FIFO", "SIEVE", "GDSF", "DynamicAdaptiveClimb", "SOLAR-E", "SOLAR-A", "SOLAR", "SOLARo-E", "SOLARo-A", "SOLARo")
BASELINE_ONLY = False      # run only the baseline select/eval tasks (writes the checkpoint) and stop; lets families run on several machines
BASE_SFX = None            # reuse the baseline checkpoint of an earlier run (baseline runs do not depend on the learned model)
CAUSAL_TUNING = False      # score each tuning window with a model fitted on earlier tuning windows only


class NullRef:
    """Cold start: no reference traffic, so every reference-derived feature is constant."""

    def __init__(self, dim: int):
        import torch
        self.V = torch.zeros((1, dim), dtype=torch.float32)
        self.df = Counter()
        self.mu = np.zeros(dim, dtype=np.float32)
        self.pc = np.zeros((E36.N_PC, dim), dtype=np.float32)


def _score(rows, lam):
    return float(np.mean([r["correct_hit_ratio_pct"] - lam * r["false_hit_rate_pct"] for r in rows]))


def main(workers: int, tag: str = "") -> dict:
    from sklearn.metrics import roc_auc_score
    banner("EXP 37 | ReuseGate under a strictly causal log layout and reduced reference traffic")
    import os
    import torch
    from ..encoders import get_bi
    get_bi()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ensure_log_vectors([REF, *TUNE, *EVAL])
    SEL = TUNE[1:] if CAUSAL_TUNING else TUNE   # with causal tuning the first window only trains
    vecs = _vectors()
    sfx = f"_{tag}" if tag else ""
    ck_base = REVISION_DIR / f"exp37_checkpoint_base{sfx if BASE_SFX is None else BASE_SFX}.jsonl"
    queries_ref = pd.read_parquet(LOG).iloc[REF[0]:REF[0] + REF[1]]["prompt"].astype(str).tolist()
    dim = len(next(iter(vecs.values())))

    # ---- baselines: selection on the tuning windows, evaluation on the evaluation windows
    cfgs_b = [{"family": f, "kappa": None, "tau": tau, **({"h": v} if v is not None else {})}
              for f in BASE for tau in TAUS for v in (VARIANTS.get(f) or (None,))]
    sel = [{"stage": "select", "data": ["log", *w], "cap": c, **cfg} for cfg in cfgs_b for w in SEL for c in CAPS]
    got = E36.run_tasks(sel, workers, ck_base)
    by_cfg = defaultdict(list)
    for t, r in zip(sel, got):
        by_cfg[E36._cfg(t)].append(r)
    chosen_b = {lam: {f: max((c for c in cfgs_b if c["family"] == f), key=lambda c: _score(by_cfg[E36._cfg(c)], lam))
                      for f in BASE} for lam in LAMBDAS}
    ev_cfgs = {E36._cfg(c): c for lam in LAMBDAS for c in chosen_b[lam].values()}
    ev = [{"stage": "eval", "data": ["log", *w], "cap": c, **cfg} for cfg in ev_cfgs.values() for w in EVAL for c in CAPS]
    got = E36.run_tasks(ev, workers, ck_base)
    res_b = defaultdict(lambda: defaultdict(dict))
    for t, r in zip(ev, got):
        res_b[E36._cfg(t)][json.dumps(t["data"])][str(t["cap"])] = r

    if BASELINE_ONLY:
        print("  baseline-only run finished (checkpoint written)", flush=True)
        return {}

    def per_window(res, cfg, fn):
        return [float(np.mean([fn(r) for r in caps.values()])) for _, caps in sorted(res[cfg].items())]

    lam_fn = lambda lam: (lambda r: r["correct_hit_ratio_pct"] - lam * r["false_hit_rate_pct"])
    metrics = ("correct_hit_ratio_pct", "false_hit_rate_pct", "net_hit_ratio_pct", "precision_of_hits_pct")

    out_R, tests, PW = {}, {}, {}
    for R in REF_SIZES:
        print(f"\n  --- reference size R = {R}", flush=True)
        ref = NullRef(dim) if R == 0 else E36.Ref(queries_ref[:R], vecs)
        LX, LY = {}, {}
        for w in TUNE + EVAL:
            LX[w], LY[w] = E36.features(log_window(*w), vecs, ref, "log")
        pred = {}
        for w in SEL:
            others = TUNE[:TUNE.index(w)] if CAUSAL_TUNING else [o for o in TUNE if o != w]
            m = E34._fit([LX[o] for o in others], [LY[o]["match"] for o in others], E36.COLS)
            pred[w] = m.predict_proba(LX[w])[:, 1]
        m = E34._fit([LX[o] for o in TUNE], [LY[o]["match"] for o in TUNE], E36.COLS)
        for w in EVAL:
            pred[w] = m.predict_proba(LX[w])[:, 1]
        g = m.booster_.feature_importance("gain")
        imp = dict(zip(E36.COLS, (g / g.sum()).round(4).tolist()))
        auc = [float(roc_auc_score(LY[w]["match"], pred[w])) for w in EVAL]
        print(f"    AUC (label-free target) {np.mean(auc):.3f}", flush=True)

        ck = REVISION_DIR / f"exp37_checkpoint_R{R}{sfx}.jsonl"
        cfgs_l = [{"family": LEARNED, "kappa": 0.0, "tau": tau} for tau in TAUS]
        sel = [{"stage": "select", "data": ["log", *w], "cap": c, "v": pred[w].tolist(), **cfg}
               for cfg in cfgs_l for w in SEL for c in CAPS]
        got = E36.run_tasks(sel, workers, ck)
        by = defaultdict(list)
        for t, r in zip(sel, got):
            by[E36._cfg(t)].append(r)
        chosen_l = {lam: max(cfgs_l, key=lambda c: _score(by[E36._cfg(c)], lam)) for lam in LAMBDAS}
        ev_l = {E36._cfg(c): c for c in chosen_l.values()}
        ev = [{"stage": "eval", "data": ["log", *w], "cap": c, "v": pred[w].tolist(), **cfg}
              for cfg in ev_l.values() for w in EVAL for c in CAPS]
        got = E36.run_tasks(ev, workers, ck)
        res_l = defaultdict(lambda: defaultdict(dict))
        for t, r in zip(ev, got):
            res_l[E36._cfg(t)][json.dumps(t["data"])][str(t["cap"])] = r

        grp = None
        if WITH_GROUP and R == max(REF_SIZES):
            pg, pb = {}, {}
            for w in SEL:
                others = TUNE[:TUNE.index(w)] if CAUSAL_TUNING else [o for o in TUNE if o != w]
                mg = E34._fit([LX[o] for o in others], [LY[o]["group_good"] for o in others], E36.COLS)
                mb = E34._fit([LX[o] for o in others], [LY[o]["group_bad"] for o in others], E36.COLS)
                pg[w], pb[w] = mg.predict_proba(LX[w])[:, 1], mb.predict_proba(LX[w])[:, 1]
            mg = E34._fit([LX[o] for o in TUNE], [LY[o]["group_good"] for o in TUNE], E36.COLS)
            mb = E34._fit([LX[o] for o in TUNE], [LY[o]["group_bad"] for o in TUNE], E36.COLS)
            for w in EVAL:
                pg[w], pb[w] = mg.predict_proba(LX[w])[:, 1], mb.predict_proba(LX[w])[:, 1]
            vg = lambda w, k: (pg[w] - k * pb[w]).tolist()
            cfgs_g = [{"family": "Learned[group]", "kappa": k, "tau": tau} for k in E36.KAPPAS for tau in TAUS]
            sel_g = [{"stage": "select", "data": ["log", *w], "cap": c, "v": vg(w, cfg["kappa"]), **cfg}
                     for cfg in cfgs_g for w in SEL for c in CAPS]
            got = E36.run_tasks(sel_g, workers, ck)
            by_g = defaultdict(list)
            for t, r in zip(sel_g, got):
                by_g[E36._cfg(t)].append(r)
            chosen_g = {lam: max(cfgs_g, key=lambda c: _score(by_g[E36._cfg(c)], lam)) for lam in LAMBDAS}
            ev_g = {E36._cfg(c): c for c in chosen_g.values()}
            evg = [{"stage": "eval", "data": ["log", *w], "cap": c, "v": vg(w, cfg["kappa"]), **cfg}
                   for cfg in ev_g.values() for w in EVAL for c in CAPS]
            got = E36.run_tasks(evg, workers, ck)
            res_g = defaultdict(lambda: defaultdict(dict))
            for t, r in zip(evg, got):
                res_g[E36._cfg(t)][json.dumps(t["data"])][str(t["cap"])] = r
            grp = (chosen_g, res_g)

        table = {}
        for lam in LAMBDAS:
            cl = E36._cfg(chosen_l[lam])
            row = {LEARNED: {"tau": chosen_l[lam]["tau"],
                             "score": seed_ci(per_window(res_l, cl, lam_fn(lam))),
                             **{m_: seed_ci(per_window(res_l, cl, lambda r, m_=m_: r[m_])) for m_ in metrics}}}
            for f in BASE:
                cb = E36._cfg(chosen_b[lam][f])
                row[f] = {"tau": chosen_b[lam][f]["tau"],
                          "score": seed_ci(per_window(res_b, cb, lam_fn(lam))),
                          **{m_: seed_ci(per_window(res_b, cb, lambda r, m_=m_: r[m_])) for m_ in metrics}}
            if grp is not None:
                cg = E36._cfg(grp[0][lam])
                row["Learned[group]"] = {"tau": grp[0][lam]["tau"], "kappa": grp[0][lam]["kappa"],
                                         "score": seed_ci(per_window(grp[1], cg, lam_fn(lam))),
                                         **{m_: seed_ci(per_window(grp[1], cg, lambda r, m_=m_: r[m_])) for m_ in metrics}}
                PW.setdefault(str(R), {}).setdefault(str(lam), {})["Learned[group]"] = per_window(grp[1], cg, lam_fn(lam))
                for b in BASE:
                    tests[f"R{R}|lambda{lam}|Learned[group]-{b}"] = paired_seed_test(
                        per_window(grp[1], cg, lam_fn(lam)), per_window(res_b, E36._cfg(chosen_b[lam][b]), lam_fn(lam)))
            table[str(lam)] = row
            print(f"    lambda {lam}: " + "  ".join(f"{f} {row[f]['score']['mean']:.2f}" for f in (LEARNED, *BASE)),
                  flush=True)
            a = per_window(res_l, cl, lam_fn(lam))
            PW.setdefault(str(R), {}).setdefault(str(lam), {})[LEARNED] = a
            for b in BASE:
                PW[str(R)][str(lam)][b] = per_window(res_b, E36._cfg(chosen_b[lam][b]), lam_fn(lam))
                tests[f"R{R}|lambda{lam}|{LEARNED}-{b}"] = paired_seed_test(
                    a, per_window(res_b, E36._cfg(chosen_b[lam][b]), lam_fn(lam)))
        out_R[str(R)] = {"auc": auc, "feature_importance": imp, "table": table,
                         "selection": {str(lam): chosen_l[lam] for lam in LAMBDAS}}

    Rm = max(REF_SIZES)
    pl = [l for l in (0, 1, 2) if l in LAMBDAS]
    primary_keys = ([f"R{Rm}|lambda{l}|{LEARNED}-WTinyLFU" for l in pl]
                    + [f"R{Rm}|lambda1|{LEARNED}-{b}" for b in ("LRU", "LFU") if b in BASE and 1 in LAMBDAS])
    secondary_keys = ([f"R{R}|lambda1|{LEARNED}-WTinyLFU" for R in REF_SIZES if R != Rm and 1 in LAMBDAS]
                      + [f"R{Rm}|lambda{l}|{LEARNED}-WTinyLFU" for l in (5, 10) if l in LAMBDAS])
    for keys in (primary_keys, secondary_keys):
        adj = holm({k: tests[k].get("t_p") for k in keys})
        for k in keys:
            tests[k]["holm_p"] = adj.get(k)
    for k in primary_keys + secondary_keys:
        v = tests[k]
        print(f"  {k:48s} {v['delta_mean']:+6.2f}  holm p {v.get('holm_p', float('nan')):.4f}", flush=True)
    group_keys = [f"R{Rm}|lambda{l}|Learned[group]-WTinyLFU" for l in LAMBDAS] if WITH_GROUP else []
    if group_keys:
        adj = holm({k: tests[k].get("t_p") for k in group_keys})
        for k in group_keys:
            tests[k]["holm_p"] = adj.get(k)
            print(f"  {k:48s} {tests[k]['delta_mean']:+6.2f}  holm p {tests[k]['holm_p']:.4f}", flush=True)
    decision = {"causal_layout_confirms_H1c_H2c": all(
        tests[k].get("holm_p") is not None and tests[k]["holm_p"] < 0.05 and tests[k]["delta_mean"] > 0
        for k in primary_keys)}
    print("  decision:", decision, flush=True)
    out = {"manifest": hardware_manifest({"layout": {"reference": list(REF), "tune": [list(w) for w in TUNE], "causal_tuning": CAUSAL_TUNING,
                                                     "selection_windows": [list(w) for w in SEL],
                                                     "eval": [list(w) for w in EVAL]},
                                          "caps": list(CAPS), "taus": list(TAUS), "lambdas": list(LAMBDAS),
                                          "reference_sizes": list(REF_SIZES), "lightgbm": E34.LGB,
                                          "half_life_x_cap": E36.HALF_LIFE}),
           "by_reference_size": out_R, "per_window_scores": PW, "tests": tests, "primary_keys": primary_keys,
           "secondary_keys": secondary_keys, "group_keys": group_keys, "decision": decision}
    save_result(out, "exp37_causal_reusegate", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--causal-tuning", action="store_true")
    ap.add_argument("--base-suffix", default=None)
    ap.add_argument("--with-group", action="store_true")
    ap.add_argument("--lecar", action="store_true")
    ap.add_argument("--extra-baselines", action="store_true")
    ap.add_argument("--only-families", default="", help="comma list: restrict BASE to these families (use with --baseline-only)")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--ref-max-only", action="store_true", help="evaluate ReuseGate at the largest reference size only")
    a = ap.parse_args()
    CAUSAL_TUNING = a.causal_tuning
    WITH_GROUP = a.with_group
    if a.lecar:
        BASE = BASE + ("LeCaR",)
    if a.extra_baselines:
        BASE = BASE + EXTRA_BASE
    if a.only_families:
        BASE = tuple(x for x in a.only_families.split(",") if x)
    BASELINE_ONLY = a.baseline_only
    if a.ref_max_only:
        REF_SIZES = (max(REF_SIZES),)
    BASE_SFX = None if a.base_suffix is None else ("" if a.base_suffix == "none" else a.base_suffix)
    if a.smoke:                 # code check on a few thousand queries
        REF = (0, 3000)
        TUNE = ((3000, 1500), (4500, 1500))
        EVAL = ((6000, 1500), (7500, 1500))
        CAPS, TAUS, LAMBDAS, REF_SIZES = (200,), (0.895, 0.94), (0, 1), (0, 1000, 3000)
        BASE = ("LRU", "WTinyLFU") + (EXTRA_BASE if a.extra_baselines else ())
        a.tag = "smoke"
    main(a.workers, a.tag)
