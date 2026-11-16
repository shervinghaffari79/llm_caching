"""
Holm-Bonferroni correction for the families of paired tests in the revision.

    python scripts/holm_report.py

Writes results/raw/revision/holm_adjusted.json and prints, per family, how
many tests are significant at 0.05 before and after adjustment and which
claims change.
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semcache.stats import holm  # noqa: E402

RAW = ROOT / "results" / "raw" / "revision"


def load(n):
    return json.load(open(RAW / n, encoding="utf8"))


def families() -> dict[str, dict[str, float | None]]:
    fam = {}
    c = load("exp11_seeds_geforcertx4090_real.json")["c1_vs_c2_across_seeds"]
    fam["exp11 C1 vs C2 (metrics)"] = {k: v.get("t_p") for k, v in c.items()}
    from semcache.stats import paired_seed_test
    ps = load("exp11_seeds_geforcertx4090_real.json")["per_seed"]
    ras = lambda k, lam: [ps[s]["stats"][k]["risk_adjusted_saving_pct"][lam] for s in ps]
    fam["exp11 default-threshold RAS vs C1 (tab:risk_adjusted)"] = {
        f"{k}-C1|lambda {lam}": paired_seed_test(ras(k, lam), ras("C1_bi_only", lam)).get("t_p")
        for k in ("C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking") for lam in ("1", "2", "5", "10")}
    t19 = load("exp19_baselines_seeds_m2.json")["paired_vs_C"]
    fam["exp19 baselines vs C1/C2"] = {f"{ref}|{sys}|{m}": v[m].get("t_p")
                                       for ref, d in t19.items() for sys, v in d.items() for m in v}
    t18 = load("exp18_eviction_m2.json")["tests"]
    fam["exp18 eviction policies vs LRU/LFU"] = {f"{p}|{ref}|{c}": v.get("t_p")
                                                 for p, d in t18.items() for ref, cc in d.items() for c, v in cc.items()}
    t24 = load("exp24_sd_admit_cpu.json")["tests"]
    fam["exp24 admission vs references"] = {f"{var}|{ref}|{c}": v.get("t_p")
                                            for c, dv in t24.items() for var, dr in dv.items() if var == "SD-Admit"
                                            for ref, v in dr.items()}
    t26 = load("exp26_risk_priced_cpu.json")["tests"]
    fam["exp26 risk-priced vs references"] = {f"{lam}|{c}|{ref}": v.get("t_p")
                                              for lam, dc in t26.items() for c, dr in dc.items() for ref, v in dr.items()}
    return fam


def main():
    out = {}
    for name, ps in families().items():
        adj = holm(ps)
        raw_sig = {k for k, p in ps.items() if p is not None and p < 0.05}
        adj_sig = {k for k, p in adj.items() if p < 0.05}
        out[name] = {"raw": ps, "holm": adj, "n_tests": len(adj),
                     "significant_raw": len(raw_sig), "significant_holm": len(adj_sig),
                     "lost_after_holm": sorted(raw_sig - adj_sig)}
        print(f"{name}: {len(adj)} tests, significant {len(raw_sig)} -> {len(adj_sig)} after Holm")
        for k in sorted(raw_sig - adj_sig):
            print(f"    no longer significant: {k} (p {ps[k]:.4f} -> {adj[k]:.4f})")
    (RAW / "holm_adjusted.json").write_text(json.dumps(out, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
