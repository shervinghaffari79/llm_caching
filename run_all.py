#!/usr/bin/env python
"""
Run the whole study end to end.

    python run_all.py                     # full run (needs OPENAI_API_KEY)
    python run_all.py --quick             # small sizes, no API calls
    python run_all.py --only exp3 exp8    # subset
    python run_all.py --llm-mode sim      # skip the API entirely

Ordering matters: exp8 joins exp1's latencies to exp3's F1 numbers, and exp9
reads everything. The runner enforces the order and keeps going if one stage
fails, so a missing API key does not cost you the classification results.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    """Read .env if present. Never committed; see .env.example."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small sample sizes, no API calls")
    ap.add_argument("--llm-mode", choices=["replay", "live", "sim"], default="replay")
    ap.add_argument("--stream", type=int, default=2000)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--verify-live", type=int, default=0,
                    help="re-run N requests against the live API to validate replay")
    args = ap.parse_args()

    _load_dotenv()
    if args.quick:
        args.llm_mode = "sim"
        args.stream = min(args.stream, 500)

    if args.llm_mode in ("replay", "live") and not os.environ.get("OPENAI_API_KEY"):
        print("! OPENAI_API_KEY not set -- falling back to --llm-mode sim.\n"
              "  Set it (export OPENAI_API_KEY=... or create .env) for real "
              "token and latency numbers.", file=sys.stderr)
        args.llm_mode = "sim"

    from semcache.experiments import (exp0_data_audit, exp1_latency_breakdown,
                                      exp2_end_to_end, exp3_ablation,
                                      exp4_meta_classifier, exp5_dynamic_cache,
                                      exp6_baselines, exp7_eviction,
                                      exp8_cheap_vs_expensive, exp9_report)

    n_lat = 60 if args.quick else 300
    n_meta = 3000 if args.quick else None

    stages = [
        ("exp0", "data audit", lambda: exp0_data_audit.main()),
        ("exp1", "latency breakdown", lambda: exp1_latency_breakdown.main(n_lat)),
        ("exp2", "end-to-end", lambda: exp2_end_to_end.main(
            args.stream, args.llm_mode, verify_live=args.verify_live)),
        ("exp3", "ablation", lambda: exp3_ablation.main()),
        ("exp4", "meta-classifier", lambda: exp4_meta_classifier.main(n_meta)),
        ("exp5", "dynamic cache", lambda: exp5_dynamic_cache.main(
            llm_mode="sim" if args.llm_mode == "sim" else "replay")),
        ("exp6", "baselines", lambda: exp6_baselines.main(
            args.stream, args.llm_mode)),
        ("exp7", "eviction", lambda: exp7_eviction.main(
            args.stream, llm_mode="sim", sweep=not args.quick)),
        ("exp8", "cheap vs expensive", lambda: exp8_cheap_vs_expensive.main()),
        ("exp9", "report", lambda: exp9_report.main()),
    ]

    selected = [s for s in stages
                if (args.only is None or s[0] in args.only) and s[0] not in args.skip]

    results, failures = {}, []
    t_all = time.perf_counter()
    for key, name, fn in selected:
        t0 = time.perf_counter()
        try:
            results[key] = fn()
            print(f"\n[{key}] {name}: done in {time.perf_counter()-t0:.1f}s")
        except Exception:
            failures.append(key)
            print(f"\n[{key}] {name}: FAILED after {time.perf_counter()-t0:.1f}s",
                  file=sys.stderr)
            traceback.print_exc()

    print("\n" + "=" * 78)
    print(f"completed {len(results)}/{len(selected)} stages in "
          f"{time.perf_counter()-t_all:.1f}s")
    if failures:
        print(f"failed: {', '.join(failures)}")
    print(f"artefacts: {ROOT / 'results'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
