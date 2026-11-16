"""
Experiment 16 -- is a served cached answer actually substitutable? (E9)

Hit correctness elsewhere is decided by QQP equivalence labels. Here an LLM
judge reads the user's question, the cached answer the cache would return and
a freshly generated answer to the same question (as a reference), and decides
whether returning the cached answer would be acceptable.

Served hits are sampled per configuration, stratified by label (label-correct
and label-false hits), and each unique (question, cached question) pair is
judged once. Verbatim repeats are counted but not judged: the cached answer
was generated for the identical string.

Outputs: judge-vs-label agreement (Cohen's kappa), substitutable rate within
each label stratum, and a judge-based precision per configuration obtained by
reweighting the strata to the run's actual label mix.

Run: python -m semcache.experiments.exp16_answer_audit --per-stratum 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import config as C
from ..encoders import warmup
from ._common import (REVISION_DIR, add_run_args, banner, ensure_pool,
                      get_stream, hardware_manifest, make_backend, save_result,
                      serve, stream_args, stream_params, train_metas)

KEYS = ("C1_bi_only", "C2_bi_geo_lgb", "C4_full_stacking")
JUDGE_MODEL = os.environ.get("SEMCACHE_JUDGE_MODEL", "gpt-6-luna")

JUDGE_PROMPT_VERSION = 2
JUDGE_SYSTEM = (
    "You evaluate a semantic cache for a question-answering assistant. A user "
    "asked a question. The cache returned an answer that was originally "
    "generated for a different but similar question. Decide whether giving the "
    "user the cached answer is as acceptable as generating a new answer for "
    "their question.\n\n"
    "Both answers were produced under the same output limit of "
    f"{C.LLM_MAX_COMPLETION_TOKENS} tokens, so either may stop mid-sentence. "
    "Do not count truncation against the cached answer.\n\n"
    "A reference answer generated for the user's question is provided as a "
    "guide to what the question asks. It is not a checklist: the cached answer "
    "need not cover the same points or match its wording.\n\n"
    "Answer not_substitutable only if the cached answer (a) addresses a "
    "different question or a different scope than the user asked about, "
    "(b) contains content that is wrong or inapplicable for this user, for "
    "example details tied to a person, place, quantity or condition that "
    "differs from the user's question, or (c) omits the specific thing the "
    "user asked for. Otherwise answer substitutable.\n\n"
    'Reply with JSON only: {"verdict": "substitutable" | "not_substitutable", '
    '"reason": "<one sentence>"}')


def _judge_prompt(item: dict) -> str:
    return (f"User question:\n{item['query']}\n\n"
            f"Cached answer (returned to the user):\n{item['cached_answer']}\n\n"
            f"Reference answer for the user's question:\n{item['fresh_answer']}")


def judge_one(client, item: dict) -> dict:
    kwargs = dict(model=JUDGE_MODEL,
                  messages=[{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": _judge_prompt(item)}],
                  response_format={"type": "json_object"},
                  max_completion_tokens=500)
    try:
        r = client.chat.completions.create(**kwargs)
        d = json.loads(r.choices[0].message.content or "{}")
        v = d.get("verdict")
        return {"verdict": v if v in ("substitutable", "not_substitutable") else None,
                "reason": d.get("reason", ""),
                "usage": [r.usage.prompt_tokens, r.usage.completion_tokens]}
    except Exception as e:                                  # pragma: no cover
        return {"verdict": None, "reason": f"error: {type(e).__name__}: {e}"}


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement_of(pairs: list[dict]) -> dict:
    pairs = [j for j in pairs if j.get("verdict")]
    lab = [j["label_correct"] for j in pairs]
    jud = [j["verdict"] == "substitutable" for j in pairs]
    return {"n": len(pairs),
            "agreement_pct": 100.0 * float(np.mean(np.equal(lab, jud))) if pairs else None,
            "cohen_kappa": cohen_kappa(lab, jud) if pairs else None,
            "label_false_but_substitutable": int(sum((not l) and j for l, j in zip(lab, jud))),
            "label_correct_but_not_substitutable": int(sum(l and (not j) for l, j in zip(lab, jud)))}


def rejudge(path: str, seed: int = C.SEED, tag: str = "") -> dict:
    """Judge the pairs of an earlier judgements file again with the current
    prompt, so prompt versions are compared on identical items."""
    from openai import OpenAI
    keep = ("step", "query", "cached_query", "cached_answer", "fresh_answer",
            "label_correct", "verbatim", "is_hard_negative")
    old = [json.loads(l) for l in open(path, encoding="utf8")]
    items = [{k: o[k] for k in keep} for o in old]
    client = OpenAI()
    with ThreadPoolExecutor(max_workers=8) as ex:
        verdicts = list(ex.map(lambda it: judge_one(client, it), items))
    judged = [{**it, **v} for it, v in zip(items, verdicts)]
    changed = sum(o.get("verdict") != j["verdict"] for o, j in zip(old, judged))
    REVISION_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVISION_DIR / f"exp16_judgements_seed{seed}{'_' + tag if tag else ''}.jsonl",
              "w", encoding="utf8") as fh:
        for j in judged:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    usage = np.array([j.get("usage", [0, 0]) for j in judged])
    out = {"manifest": hardware_manifest({"judge_model": JUDGE_MODEL,
                                          "judge_prompt_version": JUDGE_PROMPT_VERSION,
                                          "rejudged_from": str(path)}, seed=seed),
           "agreement_previous_prompt": agreement_of(old),
           "agreement": agreement_of(judged),
           "verdicts_changed": changed,
           "judge_tokens": {"prompt": int(usage[:, 0].sum()),
                            "completion": int(usage[:, 1].sum())}}
    print(f"  agreement {out['agreement_previous_prompt']['agreement_pct']:.1f}% -> "
          f"{out['agreement']['agreement_pct']:.1f}%, kappa "
          f"{out['agreement_previous_prompt']['cohen_kappa']:.2f} -> "
          f"{out['agreement']['cohen_kappa']:.2f}, verdicts changed {changed}/{len(judged)}")
    save_result(out, "exp16_rejudge", seed=seed, tag=tag)
    return out


def collect_hits(seed, n, params, fetch, keys):
    wl = get_stream(seed, n, **params)
    cov = ensure_pool(wl.queries(), live=fetch)
    metas = train_metas(seed=seed)
    backend = make_backend(seed=seed)
    hits, run_stats = {}, {}
    for key in keys:
        r = serve(key, wl, make_backend(seed=seed), metas, log_responses=True)
        run_stats[key] = r.stats
        hits[key] = []
        for e in r.per_event:
            if not e["hit"]:
                continue
            fresh = backend.pool.get(e["query"])
            hits[key].append({
                "step": e["step"], "query": e["query"],
                "cached_query": e["matched_query"],
                "cached_answer": e["matched_response"],
                "fresh_answer": fresh.answer if fresh else None,
                "label_correct": bool(e["correct"]),
                "verbatim": e["query"] == e["matched_query"],
                "is_hard_negative": e["is_hard_negative"]})
    return wl, cov, hits, run_stats


def main(seed: int = C.SEED, n: int = 2000, params: dict | None = None,
         fetch: bool = True, per_stratum: int = 150, calibrate_only: int = 0,
         keys=KEYS, tag: str = "", collect_only: bool = False) -> dict:
    banner("EXP 16 | Answer substitutability audit (LLM judge)")
    params = params or {}
    warmup(include_cross=True)
    wl, cov, hits, run_stats = collect_hits(seed, n, params, fetch, keys)

    rng = random.Random(seed)
    sample, by_pair = {}, {}
    for key in keys:
        nv = [h for h in hits[key] if not h["verbatim"] and h["fresh_answer"]]
        strata = {True: [h for h in nv if h["label_correct"]],
                  False: [h for h in nv if not h["label_correct"]]}
        sample[key] = {lab: rng.sample(v, min(per_stratum, len(v)))
                       for lab, v in strata.items()}
        for lab in (True, False):
            for h in sample[key][lab]:
                by_pair.setdefault((h["query"], h["cached_query"]), h)

    items = list(by_pair.values())
    if calibrate_only:
        items = rng.sample(items, min(calibrate_only, len(items)))
    if collect_only:        # serving and sampling only (no API key needed); judge later with judge_collected
        REVISION_DIR.mkdir(parents=True, exist_ok=True)
        dump = {"seed": seed, "n": n, "params": params, "per_stratum": per_stratum, "keys": list(keys),
                "workload_meta": wl.meta, "pool": cov, "run_stats": run_stats, "hits": hits,
                "sample": {k: {str(lab).lower(): [[h["query"], h["cached_query"]] for h in v]
                               for lab, v in sample[k].items()} for k in keys},
                "items": items}
        path = REVISION_DIR / f"exp16_collect_seed{seed}{'_' + tag if tag else ''}.json"
        path.write_text(json.dumps(dump, ensure_ascii=False), encoding="utf8")
        print(f"  collected {len(items)} pairs to judge -> {path}", flush=True)
        return dump
    return _judge_stage(seed, n, params, per_stratum, calibrate_only, keys, wl.meta, cov, hits, run_stats,
                        sample, items, tag)


def judge_collected(path: str, tag: str = "") -> dict:
    """Judge the pairs of a collect-only dump (runs where the API key lives) and write the usual result."""
    d = json.loads(open(path, encoding="utf8").read())
    keys = tuple(d["keys"])
    index = {k: {(h["query"], h["cached_query"]): h for h in d["hits"][k]} for k in keys}
    sample = {k: {lab == "true": [index[k][tuple(pr)] for pr in d["sample"][k][lab]] for lab in ("true", "false")}
              for k in keys}
    return _judge_stage(d["seed"], d["n"], d["params"], d["per_stratum"], 0, keys, d["workload_meta"], d["pool"],
                        d["hits"], d["run_stats"], sample, d["items"], tag)


def _judge_stage(seed, n, params, per_stratum, calibrate_only, keys, wl_meta, cov, hits, run_stats,
                 sample, items, tag) -> dict:
    from openai import OpenAI
    client = OpenAI()
    with ThreadPoolExecutor(max_workers=8) as ex:
        verdicts = list(ex.map(lambda it: judge_one(client, it), items))
    judged = {(it["query"], it["cached_query"]): {**it, **v}
              for it, v in zip(items, verdicts)}

    per_config = {}
    for key in keys:
        all_h = hits[key]
        n_hits = len(all_h)
        n_verb = sum(h["verbatim"] for h in all_h)
        nv = [h for h in all_h if not h["verbatim"]]
        share_true = np.mean([h["label_correct"] for h in nv]) if nv else 0.0
        strata = {}
        for lab in (True, False):
            js = [judged.get((h["query"], h["cached_query"])) for h in sample[key][lab]]
            js = [j for j in js if j and j["verdict"]]
            sub = [j["verdict"] == "substitutable" for j in js]
            strata["label_correct" if lab else "label_false"] = {
                "n_judged": len(js),
                "substitutable_pct": 100.0 * float(np.mean(sub)) if sub else None}
        st, sf = strata["label_correct"]["substitutable_pct"], strata["label_false"]["substitutable_pct"]
        judge_prec_nv = None
        if st is not None and sf is not None:
            judge_prec_nv = share_true * st + (1 - share_true) * sf
        judge_prec = None
        if judge_prec_nv is not None and n_hits:
            judge_prec = (n_verb * 100.0 + len(nv) * judge_prec_nv) / n_hits
        per_config[key] = {
            "hits": n_hits, "verbatim_hits": n_verb,
            "label_precision_pct": run_stats[key]["precision_of_hits_pct"],
            "judge_precision_pct": judge_prec,
            "judge_precision_non_verbatim_pct": judge_prec_nv,
            "strata": strata}

    agreement = agreement_of(list(judged.values()))
    usage = np.array([j.get("usage", [0, 0]) for j in judged.values()])

    REVISION_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVISION_DIR / f"exp16_judgements_seed{seed}{'_' + tag if tag else ''}.jsonl",
              "w", encoding="utf8") as fh:
        for j in judged.values():
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")

    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "judge_model": JUDGE_MODEL,
                                          "judge_prompt_version": JUDGE_PROMPT_VERSION,
                                          "per_stratum": per_stratum,
                                          "calibrate_only": calibrate_only}, seed=seed),
           "workload": wl_meta, "pool": cov, "per_config": per_config,
           "agreement": agreement,
           "judge_tokens": {"prompt": int(usage[:, 0].sum()) if len(usage) else 0,
                            "completion": int(usage[:, 1].sum()) if len(usage) else 0}}
    save_result(out, "exp16_answer_audit", seed=seed, tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.add_argument("--per-stratum", type=int, default=150)
    ap.add_argument("--calibrate-only", type=int, default=0,
                    help="judge only this many pairs (calibration pass)")
    ap.add_argument("--rejudge", default="",
                    help="re-judge the pairs of an existing judgements .jsonl")
    ap.add_argument("--collect-only", action="store_true", help="serve and sample only; write the pairs to judge")
    ap.add_argument("--judge-collected", default="", help="judge the pairs of a --collect-only dump")
    a = ap.parse_args()
    if a.judge_collected:
        judge_collected(a.judge_collected, a.tag)
        raise SystemExit
    if a.rejudge:
        rejudge(a.rejudge, a.seed, a.tag)
        raise SystemExit
    main(a.seed, a.n, stream_params(a), fetch=not a.no_fetch,
         per_stratum=a.per_stratum, calibrate_only=a.calibrate_only, tag=a.tag,
         collect_only=a.collect_only)
