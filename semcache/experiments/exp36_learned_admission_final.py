"""
Experiment 36 -- final evaluation of learned admission (LearnedValue).

The policy (experiments 33-35) values each cached entry by the predicted
probability that a later request will be served by it, from cheap features
available at arrival (text, term popularity, neighbours in a reference corpus
and earlier in the stream, 16 embedding components), and keeps the entries of
highest value: v = p_good - kappa * p_bad, positive values halve every
4 x capacity requests since last use, entries with v < 0 are never kept.
Learned[match] is trained on label-free targets (a later request lands within
the serving threshold), with kappa = 0. Learned[group] is trained on group
labels from a disjoint earlier part of the log (the assumption of the in-domain
classifier of experiment 25) and prices false hits through kappa.

Protocol, fixed before any run of this experiment:

Part 1 -- real log (SemBenchmarkSearchQueries, served by C1).
  Reference corpus: queries 50,000-99,999. Tuning windows: 100,000-149,999
  (five windows of 10,000). Evaluation windows: 0-49,999 (five windows), not
  used by any learned-admission experiment before. Capacities 250, 500, 1,000,
  2,000, 4,000. Families: LRU, LFU, SphereLFU, LeCaR, W-TinyLFU (experiment 32
  setting), Semantic Decay (original weights), Learned[match],
  Learned[group] (kappa in {0, 0.5, 1, 2, 4}). Every family chooses its
  serving threshold from {0.895, 0.91, 0.925, 0.94, 0.955, 0.97}.
  Selection: for each false-hit price lambda in {0, 1, 2, 5, 10} and each
  family, the setting with the highest mean of correct-hit ratio - lambda x
  false-hit rate over the tuning windows and capacities; learned values on the
  tuning windows come from leave-one-window-out models. Evaluation models are
  fitted on all five tuning windows.
Part 2 -- QQP streams (served by C2 at its default operating point), seeds
  42-46, capacities 200-1,000: all families except Learned[group]. The QQP
  model is trained on label-free targets (a later request within cosine
  0.726) from the tuning streams 1042-1044 and the shifted tuning streams
  1042->1043, 1043->1044, 1044->1042; its reference corpus is the streams of
  seeds 1045-1049.
Part 3 -- topic shift: the five shifted streams 42->43, ..., 46->42,
  capacities 200, 400, 600, 800, same QQP model.

Primary tests (paired t-tests over the five log windows or shifted streams,
each averaged over its capacities; one Holm family):
  H1  log, Learned[match] - W-TinyLFU at lambda = 0, 1, 2;
  H2  log, Learned[match] - LRU and - LFU at lambda = 1;
  H3  log, Learned[group] - W-TinyLFU at lambda = 0, 1, 2, 5, 10;
  H4  shift, Learned[match] - LRU, second-half correct-hit ratio.
Decision: Learned admission replaces Semantic Decay as the paper's capacity
policy if every H1 and H2 test is significantly positive and H4 is not
significantly negative; otherwise Semantic Decay is reported as an
exploratory policy (no replacement). The false-hit-priced variant is claimed
only at the lambdas where H3 is significantly positive.
QQP stationary results, per-capacity results and the clairvoyant reference
are secondary.

Amendment, made before the full run: a smoke run (a code check with a model
fitted on two streams, which touched two of the shifted evaluation streams)
showed that the half-life fixed on the log, where no shift occurs, had never
been selected for QQP. The QQP half-life is therefore selected from
{0.5, 1, 2, 4, none} x capacity on the six QQP tuning streams (out-of-fold
predictions, highest mean net hit ratio over capacities 200, 400, 600, 800),
as experiments 31-32 selected their settings. No evaluation stream enters the
selection; the tests and the decision rule are unchanged.

Run: python -m semcache.experiments.exp36_learned_admission_final --workers 26
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .. import config as C
from ..encoders import warmup
from ..stats import holm, paired_seed_test, seed_ci
from ._common import REVISION_DIR, banner, get_stream, hardware_manifest, memoized_encoders, save_result, train_metas
from . import exp34_recurrence_predictability as E34
from .exp28_shift_lecar import shifted_stream
from .exp30_sd_redesign import LOG, _vectors, ensure_log_vectors, ensure_qqp_vectors, log_window
from .exp31_semantic_admission import SERVE, run_one
from .exp32_window_admission import build
from .exp33_log_ceiling import SELECTED, ServableOracle, future_pairs
from .exp35_selfsupervised_risk import LearnedValue

LOG_REF = (50_000, 50_000)
LOG_TUNE = tuple((s, 10_000) for s in range(100_000, 150_000, 10_000))
LOG_EVAL = tuple((s, 10_000) for s in range(0, 50_000, 10_000))
LOG_CAPS = (250, 500, 1000, 2000, 4000)
TAUS = (0.895, 0.91, 0.925, 0.94, 0.955, 0.97)
KAPPAS = (0.0, 0.5, 1.0, 2.0, 4.0)
LAMBDAS = (0, 1, 2, 5, 10)
QQP_REF_SEEDS = (1045, 1046, 1047, 1048, 1049)
QQP_TRAIN = (("qqp", 1042), ("qqp", 1043), ("qqp", 1044),
             ("shift", 1042, 1043), ("shift", 1043, 1044), ("shift", 1044, 1042))
QQP_SEEDS = (42, 43, 44, 45, 46)
QQP_CAPS = tuple(range(200, 1001, 100))
SHIFT_PAIRS = ((42, 43), (43, 44), (44, 45), (45, 46), (46, 42))
SHIFT_CAPS = (200, 400, 600, 800)
QQP_SEL_CAPS = (200, 400, 600, 800)
QQP_HALF_LIVES = (0.5, 1, 2, 4, "inf")
BASELINES = ("LRU", "LFU", "SphereLFU", "LeCaR", "WTinyLFU", "SemanticDecay_fixed")
DOMAIN = {"log": {"radii": (0.7, 0.8, 0.895), "match": 0.895},
          "qqp": {"radii": (0.6, 0.726, 0.85), "match": SERVE["qqp"][2]}}
HALF_LIFE, N_PC = 4, 16
COLS = (["n_chars", "n_words", "n_digits", "web", "ref_df_mean", "ref_df_min", "ref_df_max", "ref_unseen",
         "win_term_max", "win_term_mean", "ref_n0", "ref_n1", "ref_n2", "ref_max", "win_n0", "win_n1", "win_n2",
         "win_max"] + [f"pc{k}" for k in range(N_PC)])


# ------------------------------------------------------------------ features
class Ref:
    """Unlabelled reference corpus: term document frequencies, vectors, and
    principal axes of the embedding space."""

    def __init__(self, queries: list[str], vecs: dict):
        import torch
        X = np.stack([vecs[q] for q in queries]).astype(np.float32)
        self.V = torch.from_numpy(X)
        self.df = Counter(w for q in queries for w in set(E34._words(q)))
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X[::max(1, len(X) // 10_000)] - self.mu, full_matrices=False)
        self.pc = vt[:N_PC]


def features(wl, vecs: dict, ref: Ref, domain: str):
    """Features at arrival (past only) and offline labels (future) of every
    request of a stream; same features as experiment 34 with domain radii."""
    import torch
    radii, match = DOMAIN[domain]["radii"], DOMAIN[domain]["match"]
    qs = [e.query for e in wl.events]
    n = len(qs)
    V = np.stack([vecs[q] for q in qs]).astype(np.float32)
    T = torch.from_numpy(V)
    f: dict[str, list] = defaultdict(list)
    for a in range(0, n, 2000):
        S = T[a:a + 2000] @ ref.V.T
        for k, r in enumerate(radii):
            f[f"ref_n{k}"] += (S >= r).sum(1).tolist()
        f["ref_max"] += S.max(1).values.tolist()
    S = T @ T.T
    past = torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)
    for k, r in enumerate(radii):
        f[f"win_n{k}"] = ((S >= r) & past).sum(1).tolist()
    f["win_max"] = S.masked_fill(~past, -1.0).max(1).values.tolist()
    later = (S >= match) & past.T
    top = torch.tensor([e.topic_id for e in wl.events])
    same = top[:, None] == top[None, :]
    y = {"match": later.any(1), "group_good": (later & same).any(1), "group_bad": (later & ~same).any(1)}
    y = {k: v.numpy().astype(int) for k, v in y.items()}
    del S, past, later, same
    seen: Counter = Counter()
    for q in qs:
        w = E34._words(q)
        lg = [float(np.log1p(ref.df.get(t, 0))) for t in w] or [0.0]
        wc = [seen.get(t, 0) for t in w] or [0]
        f["n_chars"].append(len(q))
        f["n_words"].append(len(w))
        f["n_digits"].append(sum(c.isdigit() for c in q))
        f["web"].append(int(any(t in q.lower() for t in E34.WEB)))
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
    return pd.DataFrame(f)[COLS], y


def workload(data):
    kind = data[0]
    if kind == "log":
        return log_window(data[1], data[2])
    if kind == "qqp":
        return get_stream(data[1], 2000)
    return shifted_stream(data[1], data[2], 2000)


# ------------------------------------------------------------------ runs
def _policy(t: dict, wl, vecs):
    fam, cap, kind = t["family"], t["cap"], t["data"][0]
    if fam.startswith("Learned"):
        h = t.get("h", HALF_LIFE)
        return LearnedValue(t["v"], float("inf") if h == "inf" else h * cap)
    if fam == "OracleServable":
        V = np.stack([vecs[e.query] for e in wl.events]).astype(np.float32)
        topics = [e.topic_id for e in wl.events]
        return ServableOracle(topics, *future_pairs(V, topics, DOMAIN["log"]["match"]))
    spec = SELECTED["WTinyLFU"] if fam == "WTinyLFU" else {"kind": fam}
    return build(spec, cap, kind, C.SEED if kind == "log" else t["data"][1])


def _worker(tasks: list[dict]) -> list[dict]:
    warmup()
    vecs = _vectors()
    out, wls = [], {}
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for t in tasks:
            d = tuple(t["data"])
            if d not in wls:
                wls = {d: workload(d)}                    # tasks arrive grouped by stream
            wl = wls[d]
            key, kw, _ = SERVE[d[0]]
            if d[0] == "log":
                seed, meta, kw = C.SEED, None, {"bi_threshold": t["tau"]}
            else:
                seed = d[1]
                meta = train_metas(seed=seed, keys=(key,)).get(key)
            out.append(run_one(wl, meta, t["cap"], _policy(t, wl, vecs), key, seed, **kw))
    return out


def _cfg(t: dict) -> tuple:
    return (t["family"], t.get("kappa"), t.get("tau"), t.get("h"))


def _key(t: dict) -> str:
    return json.dumps([t["stage"], t["data"], t["cap"], *_cfg(t)])


def run_tasks(tasks: list[dict], workers: int, checkpoint) -> list[dict]:
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r["result"]
    out = [done.get(_key(t)) for t in tasks]
    todo = [i for i, r in enumerate(out) if r is None]
    print(f"    {len(tasks) - len(todo)} runs restored, {len(todo)} to run", flush=True)

    def record(idx, res):
        with open(checkpoint, "a", encoding="utf8") as fh:
            for i, r in zip(idx, res):
                out[i] = r
                fh.write(json.dumps({"key": _key(tasks[i]), "result": r}) + "\n")

    if not todo:
        return out
    if workers <= 1:
        for i in todo:
            record([i], _worker([tasks[i]]))
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    by_data: dict[str, list[int]] = defaultdict(list)
    for i in todo:
        by_data[json.dumps(tasks[i]["data"])].append(i)
    size = max(1, min(12, len(todo) // (workers * 4)))
    chunks = [idx[j:j + size] for idx in by_data.values() for j in range(0, len(idx), size)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_worker, [tasks[i] for i in ch]): ch for ch in chunks}
        n = len(tasks) - len(todo)
        for f in as_completed(futs):
            record(futs[f], f.result())
            before, n = n, n + len(futs[f])
            if n // 100 > before // 100 or n == len(tasks):
                print(f"    {n}/{len(tasks)} runs", flush=True)
    return out


# ------------------------------------------------------------------ main
def main(workers: int, tag: str = "") -> dict:
    from sklearn.metrics import roc_auc_score
    from .exp32_window_admission import _prepare
    banner("EXP 36 | Learned admission: final evaluation (log, QQP, shift)")
    import torch
    from ..encoders import get_bi
    get_bi()                                              # load first: it resets the thread count
    torch.set_num_threads(max(1, os.cpu_count() or 1))    # encoding only; workers stay single-threaded
    ensure_log_vectors([LOG_REF, *LOG_TUNE, *LOG_EVAL])
    ensure_qqp_vectors(sorted({*QQP_REF_SEEDS, 1042, 1043, 1044, *QQP_SEEDS}))
    _prepare({*QQP_SEEDS, *(s for d in QQP_TRAIN for s in d[1:])})
    vecs = _vectors()
    ck = REVISION_DIR / f"exp36_checkpoint{('_' + tag) if tag else ''}.jsonl"

    # ---- log: features, labels, models
    ref_log = Ref(pd.read_parquet(LOG).iloc[LOG_REF[0]:LOG_REF[0] + LOG_REF[1]]["prompt"].astype(str).tolist(), vecs)
    LX, LY = {}, {}
    for w in LOG_TUNE + LOG_EVAL:
        LX[w], LY[w] = features(log_window(*w), vecs, ref_log, "log")
    print("  log features done", flush=True)
    pred = defaultdict(dict)
    targets = ("match", "group_good", "group_bad")
    importance = {}
    for tg in targets:
        for w in LOG_TUNE:
            others = [o for o in LOG_TUNE if o != w]
            m = E34._fit([LX[o] for o in others], [LY[o][tg] for o in others], COLS)
            pred[w][tg] = m.predict_proba(LX[w])[:, 1]
        m = E34._fit([LX[o] for o in LOG_TUNE], [LY[o][tg] for o in LOG_TUNE], COLS)
        for w in LOG_EVAL:
            pred[w][tg] = m.predict_proba(LX[w])[:, 1]
        g = m.booster_.feature_importance("gain")
        importance[f"log_{tg}"] = dict(zip(COLS, (g / g.sum()).round(4).tolist()))
    auc = {f"log_{tg}": [float(roc_auc_score(LY[w]["group_bad" if tg == "group_bad" else "group_good"],
                                             pred[w][tg])) for w in LOG_EVAL] for tg in targets}

    # ---- QQP: features, labels, one label-free model
    ref_q = Ref([e.query for s in QQP_REF_SEEDS for e in get_stream(s, 2000).events], vecs)
    QX, QY = {}, {}
    q_eval = [("qqp", s) for s in QQP_SEEDS] + [("shift", a, b) for a, b in SHIFT_PAIRS]
    for d in QQP_TRAIN + tuple(q_eval):
        QX[d], QY[d] = features(workload(d), vecs, ref_q, "qqp")
    for d in QQP_TRAIN:
        others = [o for o in QQP_TRAIN if o != d]
        m = E34._fit([QX[o] for o in others], [QY[o]["match"] for o in others], COLS)
        pred[d]["match"] = m.predict_proba(QX[d])[:, 1]
    mq = E34._fit([QX[d] for d in QQP_TRAIN], [QY[d]["match"] for d in QQP_TRAIN], COLS)
    for d in q_eval:
        pred[d]["match"] = mq.predict_proba(QX[d])[:, 1]
    g = mq.booster_.feature_importance("gain")
    importance["qqp_match"] = dict(zip(COLS, (g / g.sum()).round(4).tolist()))
    auc["qqp_match_vs_topic_recurrence"] = [float(roc_auc_score(QY[d]["group_good"], pred[d]["match"]))
                                            for d in q_eval if d[0] == "qqp"]
    print("  AUC: " + ", ".join("%s %.3f" % (k, np.mean(v)) for k, v in auc.items()), flush=True)

    def v_of(w, fam, kappa):
        if fam == "Learned[match]":
            return pred[w]["match"].tolist()
        v = pred[w]["group_good"] - (kappa or 0.0) * pred[w]["group_bad"]
        return v.tolist()

    # ---- QQP half-life selection on the tuning streams (out-of-fold values)
    q_sel = [{"stage": "select", "data": list(d), "cap": c, "family": "Learned[match]", "kappa": None,
              "tau": None, "h": h, "v": pred[d]["match"].tolist()}
             for h in QQP_HALF_LIVES for d in QQP_TRAIN for c in QQP_SEL_CAPS]
    got = run_tasks(q_sel, workers, ck)
    q_obj = defaultdict(list)
    for t, r in zip(q_sel, got):
        q_obj[t["h"]].append(r["net_hit_ratio_pct"])
    q_obj = {str(h): float(np.mean(v)) for h, v in q_obj.items()}
    q_h = max(QQP_HALF_LIVES, key=lambda h: q_obj[str(h)])
    print(f"  QQP half-life selection (net hit ratio): {q_obj} -> {q_h}", flush=True)

    # ---- log selection on the tuning windows
    cfgs = ([{"family": f, "kappa": None, "tau": tau} for f in BASELINES for tau in TAUS]
            + [{"family": "Learned[match]", "kappa": 0.0, "tau": tau} for tau in TAUS]
            + [{"family": "Learned[group]", "kappa": k, "tau": tau} for k in KAPPAS for tau in TAUS])
    sel_tasks = []
    for cfg in cfgs:
        for w in LOG_TUNE:
            for c in LOG_CAPS:
                t = {"stage": "select", "data": ["log", *w], "cap": c, **cfg}
                if cfg["family"].startswith("Learned"):
                    t["v"] = v_of(w, cfg["family"], cfg["kappa"])
                sel_tasks.append(t)
    got = run_tasks(sel_tasks, workers, ck)
    by_cfg = defaultdict(list)
    for t, r in zip(sel_tasks, got):
        by_cfg[_cfg(t)].append(r)

    def score(rows, lam):
        return float(np.mean([r["correct_hit_ratio_pct"] - lam * r["false_hit_rate_pct"] for r in rows]))

    families = BASELINES + ("Learned[match]", "Learned[group]")
    chosen = {lam: {f: max((c for c in cfgs if c["family"] == f), key=lambda c: score(by_cfg[_cfg(c)], lam))
                    for f in families} for lam in LAMBDAS}

    # ---- evaluation runs (log, QQP, shift)
    test_cfgs = {_cfg(c): c for lam in LAMBDAS for c in chosen[lam].values()}
    test_cfgs[("OracleServable", None, DOMAIN["log"]["match"], None)] = {"family": "OracleServable", "kappa": None,
                                                                         "tau": DOMAIN["log"]["match"]}
    ev_tasks = []
    for cfg in test_cfgs.values():
        for w in LOG_EVAL:
            for c in LOG_CAPS:
                t = {"stage": "eval", "data": ["log", *w], "cap": c, **cfg}
                if cfg["family"].startswith("Learned"):
                    t["v"] = v_of(w, cfg["family"], cfg["kappa"])
                ev_tasks.append(t)
    for fam in BASELINES + ("Learned[match]",):
        for d, caps in [(("qqp", s), QQP_CAPS) for s in QQP_SEEDS] + [(("shift", a, b), SHIFT_CAPS)
                                                                     for a, b in SHIFT_PAIRS]:
            for c in caps:
                t = {"stage": "eval", "data": list(d), "cap": c, "family": fam, "kappa": None, "tau": None}
                if fam == "Learned[match]":
                    t["v"], t["h"] = pred[d]["match"].tolist(), q_h
                ev_tasks.append(t)
    got = run_tasks(ev_tasks, workers, ck)
    res = defaultdict(lambda: defaultdict(dict))          # (kind, cfg) -> stream -> cap -> row
    for t, r in zip(ev_tasks, got):
        res[(t["data"][0], _cfg(t))][json.dumps(t["data"])][str(t["cap"])] = r

    def per_stream(kind, cfg, fn):
        return [float(np.mean([fn(r) for r in caps.values()])) for _, caps in sorted(res[(kind, cfg)].items())]

    lam_fn = lambda lam: (lambda r: r["correct_hit_ratio_pct"] - lam * r["false_hit_rate_pct"])
    log_cfg = lambda lam, f: _cfg(chosen[lam][f])
    qcfg = lambda f: (f, None, None, q_h if f == "Learned[match]" else None)

    primary = {}
    for lam in (0, 1, 2):
        primary[f"H1|log|lambda{lam}|Learned[match]-WTinyLFU"] = paired_seed_test(
            per_stream("log", log_cfg(lam, "Learned[match]"), lam_fn(lam)),
            per_stream("log", log_cfg(lam, "WTinyLFU"), lam_fn(lam)))
    for b in ("LRU", "LFU"):
        primary[f"H2|log|lambda1|Learned[match]-{b}"] = paired_seed_test(
            per_stream("log", log_cfg(1, "Learned[match]"), lam_fn(1)), per_stream("log", log_cfg(1, b), lam_fn(1)))
    for lam in LAMBDAS:
        primary[f"H3|log|lambda{lam}|Learned[group]-WTinyLFU"] = paired_seed_test(
            per_stream("log", log_cfg(lam, "Learned[group]"), lam_fn(lam)),
            per_stream("log", log_cfg(lam, "WTinyLFU"), lam_fn(lam)))
    second = lambda r: r["second_half_correct_pct"]
    primary["H4|shift|second_half|Learned[match]-LRU"] = paired_seed_test(
        per_stream("shift", qcfg("Learned[match]"), second), per_stream("shift", qcfg("LRU"), second))
    adj = holm({k: v.get("t_p") for k, v in primary.items()})
    for k, v in primary.items():
        v["holm_p"] = adj.get(k)
    sig = lambda v: v.get("holm_p") is not None and v["holm_p"] < 0.05
    h1h2 = [v for k, v in primary.items() if k.startswith(("H1", "H2"))]
    h4 = primary["H4|shift|second_half|Learned[match]-LRU"]
    decision = {"H1_H2_all_significantly_positive": all(sig(v) and v["delta_mean"] > 0 for v in h1h2),
                "H4_not_significantly_negative": not (sig(h4) and h4["delta_mean"] < 0)}
    decision["replace_semantic_decay"] = all(decision.values())
    decision["false_hit_priced_variant_claimed_at_lambdas"] = [
        lam for lam in LAMBDAS if sig(primary[f"H3|log|lambda{lam}|Learned[group]-WTinyLFU"])
        and primary[f"H3|log|lambda{lam}|Learned[group]-WTinyLFU"]["delta_mean"] > 0]

    # ---- summaries
    metrics = ("correct_hit_ratio_pct", "false_hit_rate_pct", "net_hit_ratio_pct", "precision_of_hits_pct")
    log_table = {str(lam): {f: {"setting": {"kappa": chosen[lam][f]["kappa"], "tau": chosen[lam][f]["tau"]},
                                "score": seed_ci(per_stream("log", log_cfg(lam, f), lam_fn(lam))),
                                **{m: seed_ci(per_stream("log", log_cfg(lam, f), lambda r, m=m: r[m]))
                                   for m in metrics}} for f in families} for lam in LAMBDAS}
    oracle_cfg = ("OracleServable", None, DOMAIN["log"]["match"], None)
    log_table["oracle_tau0.895"] = {m: seed_ci(per_stream("log", oracle_cfg, lambda r, m=m: r[m])) for m in metrics}
    per_cap = {str(lam): {f: {str(c): seed_ci([res[("log", log_cfg(lam, f))][s][str(c)]["correct_hit_ratio_pct"]
                                                - lam * res[("log", log_cfg(lam, f))][s][str(c)]["false_hit_rate_pct"]
                                                for s in sorted(res[("log", log_cfg(lam, f))])]) for c in LOG_CAPS}
                          for f in families} for lam in (0, 1, 2)}
    qqp_fams = BASELINES + ("Learned[match]",)
    qqp_table = {f: {m: seed_ci(per_stream("qqp", qcfg(f), lambda r, m=m: r[m])) for m in metrics} for f in qqp_fams}
    shift_table = {f: {m: seed_ci(per_stream("shift", qcfg(f), lambda r, m=m: r[m]))
                       for m in metrics + ("second_half_correct_pct", "recovery_correct_pct", "second_half_false_pct")}
                   for f in qqp_fams}
    secondary = {}
    for b in BASELINES:
        for m in ("correct_hit_ratio_pct", "net_hit_ratio_pct"):
            secondary[f"qqp|{m}|Learned[match]-{b}"] = paired_seed_test(
                per_stream("qqp", qcfg("Learned[match]"), lambda r, m=m: r[m]),
                per_stream("qqp", qcfg(b), lambda r, m=m: r[m]))
        secondary[f"shift|second_half|Learned[match]-{b}"] = paired_seed_test(
            per_stream("shift", qcfg("Learned[match]"), second), per_stream("shift", qcfg(b), second))
    adj2 = holm({k: v.get("t_p") for k, v in secondary.items()})
    for k, v in secondary.items():
        v["holm_p"] = adj2.get(k)

    for lam in LAMBDAS:
        print(f"  log lambda {lam}: " + "  ".join(
            "%s %.2f (tau %.3f%s)" % (f, log_table[str(lam)][f]["score"]["mean"], chosen[lam][f]["tau"],
                                     "" if chosen[lam][f]["kappa"] in (None, 0.0) else f", k {chosen[lam][f]['kappa']}")
            for f in families), flush=True)
    for f in qqp_fams:
        print(f"  qqp {f:20s} correct {qqp_table[f]['correct_hit_ratio_pct']['mean']:5.2f} "
              f"false {qqp_table[f]['false_hit_rate_pct']['mean']:4.2f} | shift 2nd-half "
              f"{shift_table[f]['second_half_correct_pct']['mean']:5.2f} recovery "
              f"{shift_table[f]['recovery_correct_pct']['mean']:5.2f}", flush=True)
    for k, v in primary.items():
        print(f"  primary {k:48s} {v['delta_mean']:+6.2f}  holm p {v['holm_p'] if v['holm_p'] is not None else float('nan'):.4f}",
              flush=True)
    print(f"  decision: {decision}", flush=True)
    out = {"manifest": hardware_manifest({"log": {"ref": list(LOG_REF), "tune": [list(w) for w in LOG_TUNE],
                                                  "eval": [list(w) for w in LOG_EVAL], "caps": list(LOG_CAPS)},
                                          "taus": list(TAUS), "kappas": list(KAPPAS), "lambdas": list(LAMBDAS),
                                          "qqp": {"ref_seeds": list(QQP_REF_SEEDS), "train": [list(d) for d in QQP_TRAIN],
                                                  "seeds": list(QQP_SEEDS), "caps": list(QQP_CAPS),
                                                  "shift_pairs": [list(p) for p in SHIFT_PAIRS],
                                                  "shift_caps": list(SHIFT_CAPS)},
                                          "domain": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                                         for kk, vv in d.items()} for k, d in DOMAIN.items()},
                                          "half_life_x_cap": HALF_LIFE, "lightgbm": E34.LGB,
                                          "wtinylfu": SELECTED["WTinyLFU"]}),
           "auc": auc, "feature_importance": importance,
           "qqp_half_life_selection": {"objective_net_hit_ratio": q_obj, "selected": q_h},
           "selection": {str(lam): {f: {"setting": chosen[lam][f], "tune_score": score(by_cfg[_cfg(chosen[lam][f])], lam)}
                                    for f in families} for lam in LAMBDAS},
           "log": log_table, "log_per_capacity": per_cap, "qqp": qqp_table, "shift": shift_table,
           "primary": primary, "decision": decision, "secondary": secondary,
           "results": {f"{k[0]}|{json.dumps(k[1])}": v for k, v in res.items()}}
    save_result(out, "exp36_learned_admission_final", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:                 # code check only, on locally cached data
        LOG_REF = (107_000, 3000)
        LOG_TUNE = ((100_000, 1500), (101_500, 1500))
        LOG_EVAL = ((0, 1500), (1500, 1500))
        LOG_CAPS, TAUS, KAPPAS = (200,), (0.895, 0.94), (0.0, 1.0)
        QQP_REF_SEEDS, QQP_TRAIN = (47,), (("qqp", 1043), ("shift", 1042, 1043))
        QQP_SEEDS, QQP_CAPS, QQP_SEL_CAPS, QQP_HALF_LIVES = (42, 43), (300,), (300,), (1, "inf")
        SHIFT_PAIRS, SHIFT_CAPS = ((42, 43), (43, 44)), (300,)
        BASELINES = ("LRU", "LFU", "WTinyLFU")
        a.tag = "smoke"
    main(a.workers, a.tag)
