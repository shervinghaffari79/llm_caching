"""
Experiment 28 -- workload shift and a learning-augmented eviction baseline.

(a) Shift. Each shifted stream is the first half of one evaluation stream
    followed by the first half of another (seeds 42->43, 43->44, 44->45,
    45->46, 46->42): the popular topics change abruptly at the midpoint, while
    labels stay comparable because equivalence is defined per question.
    Policies: LRU, LFU, SphereLFU, Semantic Decay (default and tuned weights)
    and LeCaR. Reported: correct-hit ratio and false-hit rate overall, before
    and after the shift, and in the first 200 requests after it.

(b) LeCaR on the standard protocol of experiment 18 (seeds 42-46,
    capacities 200-1000, C2 serving), compared with the policies of
    experiment 18 on the same streams and embeddings.

Seed-level paired tests with Holm adjustment within each part. Encoders are
memoised from recorded embeddings (`--vectors`); latency is not reported.

Run: python -m semcache.experiments.exp28_shift_lecar --workers 4 --vectors FILE
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config as C
from ..cache import (LeCaRPolicy, SemanticAdmissionPolicy, SemanticDecayPolicy, SphereLFUPolicy,
                     make_policy)
from ..encoders import warmup
from ..stats import holm, paired_seed_test, seed_ci
from ..workload import StreamEvent, Workload
from ._common import (REVISION_DIR, add_run_args, banner, get_stream, hardware_manifest,
                      make_backend, memoized_encoders, save_result, serve, train_metas)
from .exp18_eviction import CAPACITIES, EVAL_SEEDS, KEY, _metrics

PAIRS = ((42, 43), (43, 44), (44, 45), (45, 46), (46, 42))
SHIFT_CAPS = (200, 400, 600, 800)
HALF = 1000
RECOVERY = 200
SD_ADMIT = {"beta": 0.0, "cos_radius": 0.65, "period_mult": 16, "admission": True}  # exp24 frozen
SD_TUNED = (0.0, 0.0, 1.0)                                                          # exp18 tuned
SHIFT_POLICIES = ("LRU", "LFU", "SphereLFU", "SemanticDecay_fixed", "SemanticDecay_tuned", "LeCaR")
STD_POLICIES = ("LeCaR",)
SHIFT_SUBJECTS = ("SemanticDecay_fixed", "SemanticDecay_tuned", "LeCaR")
SHIFT_REFS = ("LRU", "LFU", "SphereLFU", "LeCaR")


def build(name: str, cap: int, seed: int):
    if name in ("LRU", "LFU"):
        return make_policy(name)
    if name == "SphereLFU":
        return SphereLFUPolicy(capacity=cap)
    if name == "SemanticDecay_fixed":
        return SemanticDecayPolicy()
    if name == "SemanticDecay_tuned":
        return SemanticDecayPolicy(*SD_TUNED)
    if name.startswith("Admission"):
        mult = int(name.split("aging")[1]) if "aging" in name else SD_ADMIT["period_mult"]
        return SemanticAdmissionPolicy(beta=SD_ADMIT["beta"], cos_radius=SD_ADMIT["cos_radius"],
                                       admission=True, period=mult * cap)
    if name == "LeCaR":
        return LeCaRPolicy(capacity=cap, seed=seed)
    raise ValueError(name)


_ROOT = None


def global_label(query: str) -> str:
    """Seed-independent equivalence label, defined exactly as the stream
    generator defines topics: the root of the question's QQP positive
    component when that component has 2-12 questions, and the question's own
    id otherwise (components above the size cap are discarded as topics, so
    their questions only occur as distractors with their own label). Stream
    topic ids are positions in a per-seed shuffled topic list, so they cannot
    be compared across streams."""
    global _ROOT
    if _ROOT is None:
        from ..data import UnionFind, build_splits, qid_of
        from ..workload import STREAM_DEFAULTS
        pool = build_splits().stream_pool
        uf = UnionFind()
        pos = pool[pool.is_duplicate == 1]
        qids = set()
        for q1, q2 in zip(pos.question1, pos.question2):
            a, b = qid_of(q1), qid_of(q2)
            qids.update((a, b))
            uf.union(a, b)
        size: dict[str, int] = {}
        for q in qids:
            r = uf.find(q)
            size[r] = size.get(r, 0) + 1
        lo, hi = STREAM_DEFAULTS["min_topic_size"], STREAM_DEFAULTS["max_topic_size"]
        _ROOT = (uf, qid_of, size, lo, hi, qids)
    uf, qid_of, size, lo, hi, qids = _ROOT
    q = qid_of(query)
    if q in qids:
        r = uf.find(q)
        if lo <= size[r] <= hi:
            return "c:" + r
    return "q:" + q


def _relabel_check(w: Workload) -> None:
    """The global labels must induce exactly the stream's own partition."""
    by_local, by_global = {}, {}
    for e in w.events[:HALF]:
        g = global_label(e.query)
        if by_local.setdefault(e.topic_id, g) != g or by_global.setdefault(g, e.topic_id) != e.topic_id:
            raise AssertionError(f"global labels disagree with stream labels for {e.query!r}")


def shifted_stream(a: int, b: int, n: int = 2000) -> Workload:
    wa, wb = get_stream(a, n), get_stream(b, n)
    _relabel_check(wa)
    _relabel_check(wb)
    ids: dict[str, int] = {}
    events, seen = [], set()
    for half, w in enumerate((wa, wb)):
        for e in w.events[:HALF]:
            t = ids.setdefault(global_label(e.query), len(ids))
            events.append(StreamEvent(step=len(events), query=e.query, topic_id=t,
                                      is_repeat=t in seen, is_exact=e.is_exact,
                                      is_hard_negative=e.is_hard_negative,
                                      session_id=e.session_id + half * 10 ** 6, turn=e.turn))
            seen.add(t)
    first = {e.topic_id for e in events[:HALF]}
    carry = float(np.mean([e.topic_id in first for e in events[HALF:]]))
    return Workload(events=events, meta={"kind": "shift", "first": a, "second": b, "half": HALF,
                                         "second_half_topics_seen_before_pct": 100 * carry})


def _correct(ev):
    return [bool(e["hit"] and e["correct"]) for e in ev]


def _false(ev):
    return [bool(e["hit"] and not e["correct"]) for e in ev]


def _worker(tasks: list[dict], n: int, vectors_path: str | None) -> list[dict]:
    warmup()
    vecs = {}
    if vectors_path:
        z = np.load(vectors_path, allow_pickle=True)
        vecs = dict(zip(z["queries"].tolist(), z["vectors"]))
    out: list[dict | None] = [None] * len(tasks)
    groups: dict[tuple, list[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault(tuple(t["stream"]), []).append(i)
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for key, idx in groups.items():
            if key[0] == "shift":
                wl, mseed = shifted_stream(key[1], key[2], n), key[1]
            else:
                wl, mseed = get_stream(key[1], n), key[1]
            metas = train_metas(seed=mseed, keys=(KEY,))
            for i in idx:
                t = tasks[i]
                r = serve(KEY, wl, make_backend(seed=mseed), metas, keep_events=True,
                          capacity=t["cap"], policy=build(t["policy"], t["cap"], mseed))
                c, f = _correct(r.per_event), _false(r.per_event)
                rec = _metrics(r.stats)
                if key[0] == "shift":
                    rec.update({"first_half_correct_pct": 100 * float(np.mean(c[:HALF])),
                                "second_half_correct_pct": 100 * float(np.mean(c[HALF:])),
                                "recovery_correct_pct": 100 * float(np.mean(c[HALF:HALF + RECOVERY])),
                                "first_half_false_pct": 100 * float(np.mean(f[:HALF])),
                                "second_half_false_pct": 100 * float(np.mean(f[HALF:])),
                                "recovery_false_pct": 100 * float(np.mean(f[HALF:HALF + RECOVERY]))})
                out[i] = rec
    return out


def runner(workers: int, vectors_path: str | None):
    def run(tasks, n):
        if workers <= 1:
            return _worker(tasks, n, vectors_path)
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        order = sorted(range(len(tasks)), key=lambda i: tasks[i]["stream"])
        size = max(1, len(tasks) // (workers * 3))
        chunks = [order[i:i + size] for i in range(0, len(order), size)]
        out = [None] * len(tasks)
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_worker, [tasks[i] for i in ch], n, vectors_path): ch for ch in chunks}
            done = 0
            for fu in futs:
                for i, r in zip(futs[fu], fu.result()):
                    out[i] = r
                done += len(futs[fu])
                print(f"    {done}/{len(tasks)} runs", flush=True)
        return out
    return run


def main(n: int = 2000, tag: str = "", run=None, extra: dict | None = None,
         exp18_file: str | None = None, exp24_file: str | None = None) -> dict:
    banner("EXP 28 | Workload shift; LeCaR as a learning-augmented baseline")
    run = run or runner(1, None)
    # ---- (a) shift --------------------------------------------------------
    tasks = [{"stream": ["shift", a, b], "cap": c, "policy": p}
             for a, b in PAIRS for c in SHIFT_CAPS for p in SHIFT_POLICIES]
    got = run(tasks, n)
    shift = {p: {str(c): {} for c in SHIFT_CAPS} for p in SHIFT_POLICIES}
    for t, r in zip(tasks, got):
        shift[t["policy"]][str(t["cap"])][f"{t['stream'][1]}->{t['stream'][2]}"] = r
    pairs = [f"{a}->{b}" for a, b in PAIRS]
    metrics = ("correct_hit_ratio_pct", "first_half_correct_pct", "second_half_correct_pct", "recovery_correct_pct",
               "first_half_false_pct", "second_half_false_pct", "recovery_false_pct")
    shift_summary = {p: {c: {m: seed_ci([shift[p][c][k][m] for k in pairs]) for m in metrics}
                         for c in map(str, SHIFT_CAPS)} for p in SHIFT_POLICIES}
    shift_tests = {}
    for c in map(str, SHIFT_CAPS):
        for p in SHIFT_SUBJECTS:
            for ref in SHIFT_REFS:
                if ref == p:
                    continue
                for m in ("second_half_correct_pct", "recovery_correct_pct", "second_half_false_pct"):
                    shift_tests[f"{p}|{ref}|{c}|{m}"] = paired_seed_test(
                        [shift[p][c][k][m] for k in pairs], [shift[ref][c][k][m] for k in pairs])
    adj = holm({k: v.get("t_p") for k, v in shift_tests.items()})
    for k, v in shift_tests.items():
        v["holm_p"] = adj.get(k)
    for c in map(str, SHIFT_CAPS):
        print(f"  shift, capacity {c}: " + "  ".join(
            f"{p} {shift_summary[p][c]['second_half_correct_pct']['mean']:.1f}/"
            f"{shift_summary[p][c]['recovery_correct_pct']['mean']:.1f}/"
            f"{shift_summary[p][c]['second_half_false_pct']['mean']:.2f}" for p in SHIFT_POLICIES), flush=True)

    # ---- (b) LeCaR on the standard protocol ------------------------------
    tasks = [{"stream": ["std", s], "cap": c, "policy": p} for s in EVAL_SEEDS for c in CAPACITIES
             for p in STD_POLICIES]
    got = run(tasks, n)
    std = {p: {str(c): {} for c in CAPACITIES} for p in STD_POLICIES}
    for t, r in zip(tasks, got):
        std[t["policy"]][str(t["cap"])][t["stream"][1]] = r
    std_tests, std_summary = {}, {}
    e18 = json.load(open(exp18_file, encoding="utf8"))["per_seed"] if exp18_file else None
    e24 = json.load(open(exp24_file, encoding="utf8"))["per_seed"] if exp24_file else None
    m = "correct_hit_ratio_pct"
    for c in map(str, CAPACITIES):
        std_summary[c] = {p: seed_ci([std[p][c][s][m] for s in EVAL_SEEDS]) for p in STD_POLICIES}
        refs = {}
        if e18:
            refs.update({r: [e18[r][c][str(s)][m] for s in EVAL_SEEDS] for r in ("LRU", "LFU", "SphereLFU")})
        if e24:
            refs["Admission"] = [e24["SD-Admit"][c][str(s)][m] for s in EVAL_SEEDS]
        for r, v in refs.items():
            std_summary[c][r] = seed_ci(v)
        for p in STD_POLICIES:
            mine = [std[p][c][s][m] for s in EVAL_SEEDS]
            for r, v in refs.items():
                std_tests[f"{p}|{r}|{c}"] = paired_seed_test(mine, v)
    adj = holm({k: v.get("t_p") for k, v in std_tests.items()})
    for k, v in std_tests.items():
        v["holm_p"] = adj.get(k)
    for p in STD_POLICIES:
        print(f"  {p}, standard streams: " + "  ".join(f"{c}:{std_summary[c][p]['mean']:.2f}"
                                                      for c in map(str, CAPACITIES)), flush=True)

    out = {"manifest": hardware_manifest({"n_stream": n, "config": KEY, "shift_pairs": [list(p) for p in PAIRS],
                                          "half": HALF, "recovery_window": RECOVERY, "latency_valid": False,
                                          "sd_admit": SD_ADMIT, "sd_tuned": SD_TUNED, **(extra or {})}),
           "shift": {"per_stream": shift, "summary": shift_summary, "tests": shift_tests},
           "standard": {"per_seed": std, "summary": std_summary, "tests": std_tests}}
    save_result(out, "exp28_shift_lecar", tag=tag)
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--exp18", default=str(REVISION_DIR / "exp18_eviction_m2.json"))
    ap.add_argument("--exp24", default=None)
    a = ap.parse_args()
    main(a.n, a.tag, run=runner(a.workers, a.vectors), extra={"workers": a.workers, "vectors": a.vectors},
         exp18_file=a.exp18, exp24_file=a.exp24)
