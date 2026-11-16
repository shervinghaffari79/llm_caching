"""
Experiment 34 -- can a cache tell, when a query arrives, whether it will be
asked again? (predictability check for learned admission on the real log)

Experiment 33 found that clairvoyant policies reach about 8% correct hits on
the log at every capacity, against 1-4% for online policies: the room lies in
keeping the queries that will be paraphrased later. This experiment asks
whether that is predictable from cheap, label-free features available at
arrival, and whether a cache that admits and evicts by the prediction gains
on W-TinyLFU.

Label: a request is positive if a later request of its group in the same
window has cosine >= 0.895 with it (it would serve a correct hit if kept).
Features (no group labels; the window's past only):
  text        length in characters and words, digits, web tokens
  terms       document frequency of its words in a reference corpus (log
              queries 50,000-99,999, never used for evaluation), share of
              unseen words, counts of its words earlier in the window
  reference   number of reference queries within cosine 0.7 / 0.8 / 0.895,
              highest cosine to the reference corpus
  window      the same counts and highest cosine over earlier requests
  embedding   16 principal components of the bi-encoder vector (fitted on
              the reference corpus)
Model: LightGBM with fixed settings (no tuning).

Protocol, fixed before any run of this experiment:
Training windows: log queries 100,000-129,999 (three 10,000-query windows);
test windows: 130,000-149,999 (two windows). The evaluation windows 0-49,999
are not used. Serving: C1 at 0.895; capacities 500, 1,000, 2,000.
Policies driven by the predicted probability p:
  PredLRU(t)     a missed query enters a full cache only if p >= t; LRU
                 eviction; t in {0.02, 0.05, 0.1, 0.2, 0.3}
  PredValue(h)   evict the entry (newcomer included) with the lowest p,
                 discounted by half every h x capacity requests since its last
                 use; h in {none, 0.5, 1, 2, 4}
Selection: the variant with the highest mean net hit ratio on the training
windows, using out-of-fold predictions (each training window predicted by a
model fitted on the other two). The final model is fitted on all three
training windows and applied to the test windows.
References on the test windows: LRU, W-TinyLFU and WSemDemand (experiment 32
settings) and the servable oracle of experiment 33.
Criteria: (1) mean test-window AUC of the full-feature model >= 0.70;
(2) the selected policy's net hit ratio exceeds W-TinyLFU's by >= 1.0
percentage point (mean over the test windows, each averaged over the
capacities). Learned admission is developed further only if both hold.

Run: python -m semcache.experiments.exp34_recurrence_predictability --workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import pandas as pd

from .. import config as C
from ..cache import EvictionPolicy
from ..encoders import warmup
from ..stats import seed_ci
from ._common import REVISION_DIR, banner, hardware_manifest, memoized_encoders, save_result
from .exp30_sd_redesign import LOG, _vectors, ensure_log_vectors, log_window
from .exp31_semantic_admission import SERVE, run_one
from .exp32_window_admission import build
from .exp33_log_ceiling import SELECTED, ServableOracle, future_pairs

REF = (50_000, 50_000)
TRAIN = tuple((s, 10_000) for s in (100_000, 110_000, 120_000))
TEST = tuple((s, 10_000) for s in (130_000, 140_000))
CAPS = (500, 1000, 2000)
TAU = SERVE["log"][2]
RADII = (0.7, 0.8, TAU)
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3)
HALF_LIVES = (None, 0.5, 1, 2, 4)
N_PC = 16
LGB = dict(objective="binary", learning_rate=0.05, n_estimators=300, num_leaves=31,
           min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
           random_state=42, verbose=-1, n_jobs=16,
           deterministic=True, force_row_wise=True)     # identical models across restarts
WORD = re.compile(r"[a-z0-9]+")
WEB = (".com", "www", "http", ".org", ".net")
GROUPS = {"text": ["n_chars", "n_words", "n_digits", "web"],
          "terms": ["ref_df_mean", "ref_df_min", "ref_df_max", "ref_unseen", "win_term_max", "win_term_mean"],
          "reference": [f"ref_n{r}" for r in RADII] + ["ref_max"],
          "window": [f"win_n{r}" for r in RADII] + ["win_max"],
          "embedding": [f"pc{k}" for k in range(N_PC)]}
ABLATIONS = {"text+terms": ("text", "terms"),
             "+reference": ("text", "terms", "reference"),
             "+window": ("text", "terms", "reference", "window"),
             "all": tuple(GROUPS)}


def _words(q: str) -> list[str]:
    return WORD.findall(q.lower())


class Reference:
    """Unlabelled reference corpus: term document frequencies, the matrix of
    its vectors, and principal axes of the embedding space."""

    def __init__(self, vecs: dict):
        import torch
        qs = pd.read_parquet(LOG).iloc[REF[0]:REF[0] + REF[1]]["prompt"].astype(str).tolist()
        X = np.stack([vecs[q] for q in qs]).astype(np.float32)
        self.V = torch.from_numpy(X)
        self.df = Counter(w for q in qs for w in set(_words(q)))
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X[::5] - self.mu, full_matrices=False)
        self.pc = vt[:N_PC]


def window_data(wl, vecs: dict, ref: Reference):
    import torch
    qs = [e.query for e in wl.events]
    n = len(qs)
    V = np.stack([vecs[q] for q in qs]).astype(np.float32)
    T = torch.from_numpy(V)
    f: dict[str, list] = defaultdict(list)
    for a in range(0, n, 2000):
        S = T[a:a + 2000] @ ref.V.T
        for r in RADII:
            f[f"ref_n{r}"] += (S >= r).sum(1).tolist()
        f["ref_max"] += S.max(1).values.tolist()
    S = T @ T.T
    past = torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)       # j < i
    for r in RADII:
        f[f"win_n{r}"] = ((S >= r) & past).sum(1).tolist()
    f["win_max"] = S.masked_fill(~past, -1.0).max(1).values.tolist()
    top = torch.tensor([e.topic_id for e in wl.events])
    later_same = past.T & (top[:, None] == top[None, :])                    # j > i, same group
    y = ((S >= TAU) & later_same).any(1).numpy().astype(int)
    y_group = later_same.any(1).numpy().astype(int)
    del S, past, later_same
    seen: Counter = Counter()
    for q in qs:
        w = _words(q)
        lg = [float(np.log1p(ref.df.get(t, 0))) for t in w] or [0.0]
        wc = [seen.get(t, 0) for t in w] or [0]
        f["n_chars"].append(len(q))
        f["n_words"].append(len(w))
        f["n_digits"].append(sum(c.isdigit() for c in q))
        f["web"].append(int(any(t in q.lower() for t in WEB)))
        f["ref_df_mean"].append(float(np.mean(lg)))
        f["ref_df_min"].append(min(lg))
        f["ref_df_max"].append(max(lg))
        f["ref_unseen"].append(float(np.mean([ref.df.get(t, 0) == 0 for t in w])) if w else 1.0)
        f["win_term_max"].append(max(wc))
        f["win_term_mean"].append(float(np.mean(wc)))
        seen.update(set(w))
    P = (V - ref.mu) @ ref.pc.T
    for k in range(N_PC):
        f[f"pc{k}"] = P[:, k].tolist()
    return pd.DataFrame(f), y, y_group


def _cols(ablation: str) -> list[str]:
    return [c for g in ABLATIONS[ablation] for c in GROUPS[g]]


def _fit(frames, labels, cols):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**LGB)
    m.fit(pd.concat([fr[cols] for fr in frames], ignore_index=True), np.concatenate(labels))
    return m


# ------------------------------------------------------------------ policies
class PredictedAdmission(EvictionPolicy):
    """A missed query enters a full cache only if its predicted probability of
    being asked again reaches t; residents leave in LRU order."""
    name = "PredLRU"

    def __init__(self, p, t: float):
        self.p, self.t = p, t
        self.step, self.new, self.p_new = -1, None, 1.0
        self.order: OrderedDict[int, None] = OrderedDict()

    def on_query(self, vec):
        self.step += 1

    def on_insert(self, entry, vec):
        self.order[entry.eid] = None
        self.new, self.p_new = entry.eid, self.p[self.step]

    def on_hit(self, entry, vec):
        self.order.move_to_end(entry.eid)

    def choose_victim(self, cache, vb):
        if self.new in self.order and self.p_new < self.t:
            return self.new
        return next(iter(self.order))

    def on_evict(self, eid):
        self.order.pop(eid, None)


class PredictedValue(EvictionPolicy):
    """Evict the entry, newcomer included, with the lowest predicted
    probability of being asked again, halved every `half_life` requests since
    its last use (no discount when half_life is None)."""
    name = "PredValue"

    def __init__(self, p, half_life: float | None):
        self.p, self.h = p, half_life
        self.step = -1
        self.val: dict[int, float] = {}
        self.last: dict[int, int] = {}

    def on_query(self, vec):
        self.step += 1

    def on_insert(self, entry, vec):
        self.val[entry.eid] = self.p[self.step]
        self.last[entry.eid] = self.step

    def on_hit(self, entry, vec):
        self.last[entry.eid] = self.step

    def choose_victim(self, cache, vb):
        if self.h is None:
            return min(self.val, key=lambda e: (self.val[e], self.last[e]))
        return min(self.val, key=lambda e: (self.val[e] * 0.5 ** ((self.step - self.last[e]) / self.h),
                                            self.last[e]))

    def on_evict(self, eid):
        self.val.pop(eid, None)
        self.last.pop(eid, None)


def variants() -> list[dict]:
    return ([{"policy": "PredLRU", "param": t} for t in THRESHOLDS]
            + [{"policy": "PredValue", "param": h} for h in HALF_LIVES])


def _vname(v: dict) -> str:
    return f"{v['policy']}({v['param']})"


def _policy(t: dict, cap: int, wl, vecs):
    k = t["policy"]
    if k == "PredLRU":
        return PredictedAdmission(t["p"], t["param"])
    if k == "PredValue":
        return PredictedValue(t["p"], None if t["param"] is None else t["param"] * cap)
    if k == "OracleServable":
        V = np.stack([vecs[e.query] for e in wl.events]).astype(np.float32)
        topics = [e.topic_id for e in wl.events]
        return ServableOracle(topics, *future_pairs(V, topics, TAU))
    return build({"LRU": {"kind": "LRU"}, **SELECTED}[k], cap, "log", C.SEED)


def _worker(tasks: list[dict]) -> list[dict]:
    warmup()
    vecs = _vectors()
    key, kw, _ = SERVE["log"]
    out = []
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for t in tasks:
            wl = log_window(*t["window"])
            out.append(run_one(wl, None, t["cap"], _policy(t, t["cap"], wl, vecs), key, C.SEED, **kw))
    return out


def _key(t: dict) -> str:
    return json.dumps([t["stage"], t["window"], t["cap"], t["policy"], t.get("param")])


def run_tasks(tasks: list[dict], workers: int, checkpoint, worker=None, key=None) -> list[dict]:
    """Checkpointed runner; `worker` and `key` default to this experiment's."""
    worker, _key = worker or _worker, key or globals()["_key"]
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r["result"]
    out = [done.get(_key(t)) for t in tasks]
    todo = [i for i, r in enumerate(out) if r is None]
    print(f"    {len(tasks) - len(todo)} runs restored, {len(todo)} to run", flush=True)

    def record(i, r):
        out[i] = r
        with open(checkpoint, "a", encoding="utf8") as fh:
            fh.write(json.dumps({"key": _key(tasks[i]), "result": r}) + "\n")

    if workers <= 1:
        for i in todo:
            record(i, worker([tasks[i]])[0])
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(worker, [tasks[i]]): i for i in todo}
        n = len(tasks) - len(todo)
        for f in as_completed(futs):
            record(futs[f], f.result()[0])
            n += 1
            if n % 10 == 0 or n == len(tasks):
                print(f"    {n}/{len(tasks)} runs", flush=True)
    return out


# ------------------------------------------------------------------ main
def main(workers: int, tag: str = "") -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    banner("EXP 34 | Recurrence predictability and learned admission on the real log")
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:
        pass
    ensure_log_vectors([REF, *TRAIN, *TEST])
    vecs = _vectors()
    ref = Reference(vecs)
    data = {w: window_data(log_window(*w), vecs, ref) for w in TRAIN + TEST}
    rates = {str(w[0]): {"servable_recurrence_pct": 100.0 * float(data[w][1].mean()),
                         "group_recurrence_pct": 100.0 * float(data[w][2].mean())} for w in TRAIN + TEST}
    shown = ", ".join("%s %.1f%%" % (k, v["servable_recurrence_pct"]) for k, v in rates.items())
    print(f"  positive rate (servable recurrence): {shown}", flush=True)

    # 1. predictability: fit on the training windows, score the test windows
    auc = {}
    for ab in ABLATIONS:
        cols = _cols(ab)
        m = _fit([data[w][0] for w in TRAIN], [data[w][1] for w in TRAIN], cols)
        auc[ab] = {}
        for w in TEST:
            X, y, yg = data[w]
            p = m.predict_proba(X[cols])[:, 1]
            auc[ab][str(w[0])] = {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
                                  "auc_group_label": float(roc_auc_score(yg, p))}
        print(f"  {ab:12s} test AUC " + ", ".join(f"{v['auc']:.3f}" for v in auc[ab].values())
              + "  AP " + ", ".join(f"{v['ap']:.3f}" for v in auc[ab].values()), flush=True)
    cols = _cols("all")
    final = _fit([data[w][0] for w in TRAIN], [data[w][1] for w in TRAIN], cols)
    gain = final.booster_.feature_importance("gain")
    importance = {g: float(sum(gain[cols.index(c)] for c in GROUPS[g]) / gain.sum()) for g in GROUPS}
    p_test = {w: final.predict_proba(data[w][0][cols])[:, 1] for w in TEST}
    p_oof = {}
    for w in TRAIN:
        others = [o for o in TRAIN if o != w]
        m = _fit([data[o][0] for o in others], [data[o][1] for o in others], cols)
        p_oof[w] = m.predict_proba(data[w][0][cols])[:, 1]
        print(f"  out-of-fold AUC window {w[0]}: {roc_auc_score(data[w][1], p_oof[w]):.3f}", flush=True)

    # 2. policies: select on the training windows (out-of-fold), then test
    var = variants()
    tasks = [{"stage": "select", "window": list(w), "cap": c, **v, "p": p_oof[w].tolist()}
             for v in var for w in TRAIN for c in CAPS]
    tasks += [{"stage": "test", "window": list(w), "cap": c, **v, "p": p_test[w].tolist()}
              for v in var for w in TEST for c in CAPS]
    tasks += [{"stage": "test", "window": list(w), "cap": c, "policy": r}
              for r in ("LRU", "WTinyLFU", "WSemDemand", "OracleServable") for w in TEST for c in CAPS]
    got = run_tasks(tasks, workers, REVISION_DIR / f"exp34_checkpoint{('_' + tag) if tag else ''}.jsonl")
    res: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))   # stage -> name -> window -> cap
    for t, r in zip(tasks, got):
        name = _vname(t) if t["policy"] in ("PredLRU", "PredValue") else t["policy"]
        res[t["stage"]][name][str(t["window"][0])][str(t["cap"])] = r

    def per_window(stage, name, metric):
        return [float(np.mean(list(v[metric] for v in caps.values())))
                for _, caps in sorted(res[stage][name].items())]

    selection = {_vname(v): float(np.mean(per_window("select", _vname(v), "net_hit_ratio_pct"))) for v in var}
    chosen = max(selection, key=selection.get)
    metrics = ("correct_hit_ratio_pct", "false_hit_rate_pct", "net_hit_ratio_pct", "precision_of_hits_pct")
    test = {n: {m: seed_ci(per_window("test", n, m)) for m in metrics} for n in res["test"]}
    gain_w = [a - b for a, b in zip(per_window("test", chosen, "net_hit_ratio_pct"),
                                    per_window("test", "WTinyLFU", "net_hit_ratio_pct"))]
    room_w = [a - b for a, b in zip(per_window("test", "OracleServable", "net_hit_ratio_pct"),
                                    per_window("test", "WTinyLFU", "net_hit_ratio_pct"))]
    mean_auc = float(np.mean([v["auc"] for v in auc["all"].values()]))
    criteria = {"auc_all_features": mean_auc, "c1_auc_at_least_0.70": bool(mean_auc >= 0.70),
                "selected_policy": chosen, "net_gain_over_wtinylfu_pts": float(np.mean(gain_w)),
                "net_gain_per_test_window": gain_w,
                "c2_gain_at_least_1pt": bool(np.mean(gain_w) >= 1.0),
                "share_of_oracle_headroom": float(np.mean(gain_w) / np.mean(room_w)) if np.mean(room_w) else None}
    criteria["pursue"] = criteria["c1_auc_at_least_0.70"] and criteria["c2_gain_at_least_1pt"]

    print(f"  feature-group importance (gain share): " + ", ".join(f"{g} {v:.2f}" for g, v in importance.items()))
    for n, s in sorted(selection.items(), key=lambda kv: -kv[1]):
        print(f"  select {n:18s} net {s:6.2f}", flush=True)
    for n in [chosen, "LRU", "WTinyLFU", "WSemDemand", "OracleServable"]:
        s = test[n]
        print(f"  test {n:18s} correct {s['correct_hit_ratio_pct']['mean']:5.2f}  false {s['false_hit_rate_pct']['mean']:5.2f}  "
              f"net {s['net_hit_ratio_pct']['mean']:5.2f}  precision {s['precision_of_hits_pct']['mean']:5.1f}", flush=True)
    print(f"  criteria: {criteria}", flush=True)
    out = {"manifest": hardware_manifest({"reference": list(REF), "train": [list(w) for w in TRAIN],
                                          "test": [list(w) for w in TEST], "caps": list(CAPS), "tau": TAU,
                                          "lightgbm": LGB, "thresholds": list(THRESHOLDS),
                                          "half_lives": list(HALF_LIVES), "wtinylfu_wsemdemand": SELECTED}),
           "positive_rates": rates, "auc": auc, "feature_group_importance": importance,
           "selection_net_hit_ratio": selection, "test_summary": test,
           "test_results": json.loads(json.dumps(res["test"])), "criteria": criteria}
    save_result(out, "exp34_recurrence_predictability", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:                 # code check only, on locally cached vectors
        REF = (107_000, 3000)
        TRAIN = ((100_000, 1500), (101_500, 1500), (103_000, 1500))
        TEST = ((104_500, 1500), (106_000, 1500))
        CAPS, THRESHOLDS, HALF_LIVES = (200,), (0.1,), (None, 1)
        a.tag = "smoke"
    main(a.workers, a.tag)
