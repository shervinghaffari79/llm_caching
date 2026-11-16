"""
Experiment 35 -- learned admission without group labels, and priced for
false hits.

Experiment 34 trained the recurrence predictor on the benchmark's paraphrase
groups and found that the policy gains correct hits but also false hits
(break-even false-hit price about 2). This experiment removes the group
labels from training and adds a false-hit term.

Training labels, all computed from the window's later requests at the serving
threshold 0.895 (labels are known only offline, after the fact):
  match         a later request lands within the threshold (no judge at all);
  judged        the same, split by the cross-encoder of C3 (score >= 0.557)
                into a correct-looking match (good) and a wrong-looking one
                (bad); at most the first eight later matches of a request are
                judged;
  group         split by the benchmark's groups (reference only; not
                available in deployment).
Each label set trains LightGBM models on the features of experiment 34. The
policy values an entry at v = p_good - kappa * p_bad, keeps the entries with
the highest v (positive values halve every 4 x capacity requests since last
use, as selected in experiment 34) and never keeps an entry with v < 0.

Protocol, fixed before any run of this experiment:
Data as in experiment 34: reference corpus 50,000-99,999, training windows
100,000-129,999, test windows 130,000-149,999; capacities 500, 1,000, 2,000.
Families and grids: LRU and W-TinyLFU (experiment 32 setting); Learned[match]
(kappa 0); Learned[judged] and Learned[group] (kappa in {0, 0.5, 1, 2, 4});
every family also chooses its serving threshold from {0.895, 0.91, 0.925,
0.94, 0.955}. For each false-hit price lambda in {0, 1, 2, 5, 10} (the grid of
the manuscript) and each family, the setting with the highest mean of
correct-hit ratio - lambda x false-hit rate on the training windows (out-of-
fold predictions) is selected and then run on the test windows.
Criteria (means over the two test windows, each averaged over capacities):
  step 1: at lambda = 1, Learned[judged] exceeds W-TinyLFU by >= 1.0 point;
  step 2: Learned[judged] exceeds W-TinyLFU at every lambda in {1, 2, 5, 10}.
Learned[match] is reported as the fully label-free variant.

Run: python -m semcache.experiments.exp35_selfsupervised_risk --workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from .. import config as C
from ..cache import EvictionPolicy
from ..encoders import warmup
from ..stats import seed_ci
from ._common import REVISION_DIR, banner, hardware_manifest, memoized_encoders, save_result
from .exp30_sd_redesign import _vectors, ensure_log_vectors, log_window
from .exp31_semantic_admission import SERVE, run_one
from .exp32_window_admission import build
from .exp33_log_ceiling import SELECTED, ServableOracle, future_pairs
from . import exp34_recurrence_predictability as E34

TAU0 = SERVE["log"][2]
TAUS = (0.895, 0.91, 0.925, 0.94, 0.955)
KAPPAS = (0.0, 0.5, 1.0, 2.0, 4.0)
LAMBDAS = (0, 1, 2, 5, 10)
HALF_LIFE = 4
MAX_JUDGED = 8
FAMILIES = {"LRU": None, "WTinyLFU": None, "Learned[match]": ("match", None),
            "Learned[judged]": ("judged_good", "judged_bad"), "Learned[group]": ("group_good", "group_bad")}
TARGETS = ("match", "judged_good", "judged_bad", "group_good", "group_bad")


def labels(wl, vecs: dict, cross=None):
    """Offline labels from the window's later requests (see module docstring)."""
    import torch
    qs = [e.query for e in wl.events]
    n = len(qs)
    T = torch.from_numpy(np.stack([vecs[q] for q in qs]).astype(np.float32))
    M = (T @ T.T >= TAU0) & torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    top = torch.tensor([e.topic_id for e in wl.events])
    same = top[:, None] == top[None, :]
    y = {"match": M.any(1), "group_good": (M & same).any(1), "group_bad": (M & ~same).any(1)}
    y = {k: v.numpy().astype(int) for k, v in y.items()}
    stats = {}
    if cross is not None:
        rows, cols = (t.tolist() for t in torch.nonzero(M, as_tuple=True))
        kept, per = [], defaultdict(int)
        for i, j in zip(rows, cols):                    # row-major: earliest matches first
            if per[i] < MAX_JUDGED:
                per[i] += 1
                kept.append((i, j))
        t0 = time.time()
        sc = cross.score([(qs[i], qs[j]) for i, j in kept]) if kept else np.zeros(0)
        good, bad = np.zeros(n, int), np.zeros(n, int)
        for (i, j), s in zip(kept, sc):
            (good if s >= C.CROSS_THRESHOLD else bad)[i] = 1
        y["judged_good"], y["judged_bad"] = good, bad
        same_np = same.numpy()
        is_same = np.array([same_np[i, j] for i, j in kept], bool)
        says = sc >= C.CROSS_THRESHOLD
        stats = {"pairs_above_threshold": len(rows), "pairs_judged": len(kept),
                 "judge_seconds": time.time() - t0,
                 "judge_precision_vs_groups": float(is_same[says].mean()) if says.any() else None,
                 "judge_recall_vs_groups": float(says[is_same].mean()) if is_same.any() else None}
    return y, stats


class LearnedValue(EvictionPolicy):
    """Keep the entries with the highest expected value v = p_good - kappa *
    p_bad; positive values halve every `half_life` requests since last use,
    and an entry with v < 0 is dropped at once (never kept)."""
    name = "LearnedValue"

    def __init__(self, v, half_life: float):
        self.v, self.h = v, half_life
        self.step = -1
        self.val: dict[int, float] = {}
        self.last: dict[int, int] = {}
        self.drop: list[int] = []

    def on_query(self, vec):
        self.step += 1

    def on_insert(self, entry, vec):
        x = self.v[self.step]
        self.val[entry.eid], self.last[entry.eid] = x, self.step
        if x < 0:
            self.drop.append(entry.eid)

    def on_hit(self, entry, vec):
        self.last[entry.eid] = self.step

    def _score(self, e):
        x = self.val[e]
        return x * 0.5 ** ((self.step - self.last[e]) / self.h) if x > 0 else x

    def choose_victim(self, cache, vb):
        return min(self.val, key=lambda e: (self._score(e), self.last[e]))

    def purge(self):
        out, self.drop = self.drop, []
        return [e for e in out if e in self.val]

    def on_evict(self, eid):
        self.val.pop(eid, None)
        self.last.pop(eid, None)


def _policy(t: dict, cap: int, wl, vecs):
    f = t["family"]
    if f.startswith("Learned"):
        return LearnedValue(t["v"], HALF_LIFE * cap)
    if f == "OracleServable":
        V = np.stack([vecs[e.query] for e in wl.events]).astype(np.float32)
        topics = [e.topic_id for e in wl.events]
        return ServableOracle(topics, *future_pairs(V, topics, TAU0))
    return build({"LRU": {"kind": "LRU"}, **SELECTED}[f], cap, "log", C.SEED)


def _worker(tasks: list[dict]) -> list[dict]:
    warmup()
    vecs = _vectors()
    key = SERVE["log"][0]
    out = []
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for t in tasks:
            wl = log_window(*t["window"])
            out.append(run_one(wl, None, t["cap"], _policy(t, t["cap"], wl, vecs), key, C.SEED,
                               bi_threshold=t["tau"]))
    return out


def _cfg(t: dict) -> tuple:
    return (t["family"], t.get("kappa"), t["tau"])


def _key(t: dict) -> str:
    return json.dumps([t["stage"], t["window"], t["cap"], *_cfg(t)])


def run_tasks(tasks, workers, checkpoint):
    return E34.run_tasks(tasks, workers, checkpoint, worker=_worker, key=_key)


def configs() -> list[dict]:
    out = []
    for f, tg in FAMILIES.items():
        kappas = KAPPAS if (tg and tg[1]) else (None,) if tg is None else (0.0,)
        out += [{"family": f, "kappa": k, "tau": tau} for k in kappas for tau in TAUS]
    return out


def main(workers: int, tag: str = "") -> dict:
    from sklearn.metrics import roc_auc_score
    banner("EXP 35 | Learned admission: self-supervised labels and false-hit pricing")
    import torch
    from ..encoders import get_cross
    ensure_log_vectors([E34.REF, *E34.TRAIN, *E34.TEST])
    vecs = _vectors()
    cross = get_cross()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ref = E34.Reference(vecs)
    X, Y, judge = {}, {}, {}
    for w in E34.TRAIN + E34.TEST:
        wl = log_window(*w)
        X[w] = E34.window_data(wl, vecs, ref)[0]
        Y[w], judge[str(w[0])] = labels(wl, vecs, cross if w in E34.TRAIN else None)
        print(f"  window {w[0]}: labels done {judge[str(w[0])]}", flush=True)
    cols = E34._cols("all")

    # models: out-of-fold on the training windows, final model for the test windows
    pred = defaultdict(dict)                      # window -> target -> p
    for tg in TARGETS:
        for w in E34.TRAIN:
            others = [o for o in E34.TRAIN if o != w]
            m = E34._fit([X[o] for o in others], [Y[o][tg] for o in others], cols)
            pred[w][tg] = m.predict_proba(X[w][cols])[:, 1]
        m = E34._fit([X[o] for o in E34.TRAIN], [Y[o][tg] for o in E34.TRAIN], cols)
        for w in E34.TEST:
            pred[w][tg] = m.predict_proba(X[w][cols])[:, 1]
    auc = {tg: {str(w[0]): float(roc_auc_score(Y[w]["group_bad" if tg.endswith("bad") else "group_good"],
                                               pred[w][tg])) for w in E34.TEST} for tg in TARGETS}
    print("  test AUC against the group-defined outcome: "
          + ", ".join("%s %.3f" % (tg, np.mean(list(v.values()))) for tg, v in auc.items()), flush=True)

    def value(w, cfg):
        good, bad = FAMILIES[cfg["family"]]
        v = pred[w][good]
        return (v - cfg["kappa"] * pred[w][bad] if bad and cfg["kappa"] else v).tolist()

    def tasks_for(stage, windows, cfgs):
        out = []
        for cfg in cfgs:
            for w in windows:
                for c in E34.CAPS:
                    t = {"stage": stage, "window": list(w), "cap": c, **cfg}
                    if cfg["family"].startswith("Learned"):
                        t["v"] = value(w, cfg)
                    out.append(t)
        return out

    ck = REVISION_DIR / f"exp35_checkpoint{('_' + tag) if tag else ''}.jsonl"
    cfgs = configs()
    sel_tasks = tasks_for("select", E34.TRAIN, cfgs)
    got = run_tasks(sel_tasks, workers, ck)

    def score(rows, lam):
        return float(np.mean([r["correct_hit_ratio_pct"] - lam * r["false_hit_rate_pct"] for r in rows]))

    by_cfg = defaultdict(list)
    for t, r in zip(sel_tasks, got):
        by_cfg[_cfg(t)].append(r)
    chosen = {lam: {f: max((c for c in cfgs if c["family"] == f), key=lambda c: score(by_cfg[_cfg(c)], lam))
                    for f in FAMILIES} for lam in LAMBDAS}
    test_cfgs = {_cfg(c): c for lam in LAMBDAS for c in chosen[lam].values()}
    test_cfgs[("OracleServable", None, TAU0)] = {"family": "OracleServable", "kappa": None, "tau": TAU0}
    test_tasks = tasks_for("test", E34.TEST, list(test_cfgs.values()))
    got_t = run_tasks(test_tasks, workers, ck)
    res = defaultdict(lambda: defaultdict(list))           # cfg -> window -> rows
    for t, r in zip(test_tasks, got_t):
        res[_cfg(t)][str(t["window"][0])].append(r)

    def per_window(cfg, lam):
        return [score(rows, lam) for _, rows in sorted(res[cfg].items())]

    table, gains = {}, {}
    for lam in LAMBDAS:
        table[str(lam)] = {f: {"setting": {"kappa": c["kappa"], "tau": c["tau"]},
                               "train_score": score(by_cfg[_cfg(c)], lam),
                               "test": seed_ci(per_window(_cfg(c), lam)),
                               "test_correct": float(np.mean([r["correct_hit_ratio_pct"] for rows in res[_cfg(c)].values() for r in rows])),
                               "test_false": float(np.mean([r["false_hit_rate_pct"] for rows in res[_cfg(c)].values() for r in rows]))}
                           for f, c in chosen[lam].items()}
        base = per_window(_cfg(chosen[lam]["WTinyLFU"]), lam)
        gains[str(lam)] = {f: [a - b for a, b in zip(per_window(_cfg(chosen[lam][f]), lam), base)]
                           for f in FAMILIES if f.startswith("Learned")}
    oracle = {str(lam): seed_ci(per_window(("OracleServable", None, TAU0), lam)) for lam in LAMBDAS}
    g = lambda lam, f="Learned[judged]": float(np.mean(gains[str(lam)][f]))
    criteria = {"step1_gain_at_lambda1": g(1), "step1_pass": bool(g(1) >= 1.0),
                "step2_gains": {str(l): g(l) for l in (1, 2, 5, 10)},
                "step2_pass": bool(all(g(l) > 0 for l in (1, 2, 5, 10))),
                "label_free_match_gains": {str(l): g(l, "Learned[match]") for l in LAMBDAS},
                "group_label_gains": {str(l): g(l, "Learned[group]") for l in LAMBDAS}}
    criteria["pursue"] = criteria["step1_pass"] and criteria["step2_pass"]

    for lam in LAMBDAS:
        print(f"  lambda {lam}:", flush=True)
        for f, row in table[str(lam)].items():
            gg = f"  gain {np.mean(gains[str(lam)][f]):+5.2f}" if f in gains[str(lam)] else ""
            print(f"    {f:16s} kappa {str(row['setting']['kappa']):4s} tau {row['setting']['tau']:.3f}  "
                  f"test {row['test']['mean']:6.2f}  (correct {row['test_correct']:4.2f}, false {row['test_false']:4.2f}){gg}",
                  flush=True)
    print(f"  criteria: {criteria}", flush=True)
    out = {"manifest": hardware_manifest({"tau0": TAU0, "taus": list(TAUS), "kappas": list(KAPPAS),
                                          "lambdas": list(LAMBDAS), "half_life_x_cap": HALF_LIFE,
                                          "max_judged": MAX_JUDGED, "cross_threshold": C.CROSS_THRESHOLD,
                                          "splits": {"ref": list(E34.REF), "train": [list(w) for w in E34.TRAIN],
                                                     "test": [list(w) for w in E34.TEST]}}),
           "judge": judge, "auc_vs_group_outcome": auc,
           "positive_rates": {str(w[0]): {k: float(v.mean()) for k, v in Y[w].items()} for w in Y},
           "table": table, "gains_vs_wtinylfu": gains, "oracle": oracle, "criteria": criteria,
           "test_results": {json.dumps(k): v for k, v in res.items()}}
    save_result(out, "exp35_selfsupervised_risk", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:                 # code check only, on locally cached vectors
        E34.REF = (107_000, 3000)
        E34.TRAIN = ((100_000, 1500), (101_500, 1500), (103_000, 1500))
        E34.TEST = ((104_500, 1500), (106_000, 1500))
        E34.CAPS = (200,)
        TAUS, KAPPAS = (0.895, 0.94), (0.0, 1.0)
        a.tag = "smoke"
    main(a.workers, a.tag)
