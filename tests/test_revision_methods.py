"""
Unit tests for the methods and metrics added in the revision.

    PYTHONPATH=. python -m pytest tests/test_revision_methods.py -q

Small hand-checkable cases; no models, streams or network.
"""
import math

import numpy as np
import pytest
from scipy.stats import norm

from semcache.cache import (CacheEntry, RiskPricedPolicy, SemanticAdmissionPolicy,
                            SphereLFUPolicy, StreamStats)
from semcache.stats import holm, paired_seed_test


def entry(i):
    return CacheEntry(eid=i, query=str(i), response="", prompt_tokens=0,
                      completion_tokens=0, created_at=0, last_access=0)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ statistics
def test_holm_matches_textbook_example():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03)          # 3 * 0.01
    assert adj["c"] == pytest.approx(0.06)          # 2 * 0.03
    assert adj["b"] == pytest.approx(0.06)          # max(1 * 0.04, 0.06): monotone
    assert "d" not in adj


def test_paired_seed_test_sign_is_a_minus_b():
    r = paired_seed_test([3, 4, 5, 6, 7], [1, 2, 3, 4, 6])
    assert r["delta_mean"] == pytest.approx(1.8)
    assert r["t_p"] < 0.05


# ------------------------------------------------------------------ metrics
def test_risk_adjusted_saving_and_break_even():
    s = StreamStats()
    s.tokens_served_correct, s.tokens_served_false = 400, 100
    s.tokens_served_by_cache, s.tokens_generated = 500, 500
    s.n, s.hits, s.misses, s.true_hits, s.false_hits = 10, 5, 5, 4, 1
    s.latencies_s, s.overheads_ms = [1.0] * 10, [1.0] * 10
    s.hit_latencies_s, s.miss_latencies_s = [0.01] * 5, [2.0] * 5
    out = s.summary()
    total = out["total_tokens_no_cache_equivalent"]
    assert out["token_saving_correct_only_pct"] == pytest.approx(100 * 400 / total)
    assert out["risk_adjusted_saving_pct"]["0"] == pytest.approx(out["token_saving_correct_only_pct"])
    assert out["risk_adjusted_saving_pct"]["2"] == pytest.approx(100 * (400 - 2 * 100) / total)
    assert out["break_even_false_hit_cost"] == pytest.approx(4.0)   # saving reaches 0 at lambda = 4


# ------------------------------------------------------------------ SphereLFU
def test_sphere_lfu_spreads_one_unit_inside_radius_only():
    p = SphereLFUPolicy(cos_threshold=0.9, capacity=100)
    a, b, far = unit([1, 0, 0]), unit([1, 0.1, 0]), unit([0, 0, 1])
    for i, v in enumerate((a, b, far)):
        p.on_insert(entry(i), v)
    before = dict(p.count)
    p.on_query(unit([1, 0.05, 0]))
    gain = {e: p.count[e] - before[e] for e in before}
    assert sum(gain.values()) == pytest.approx(1.0)
    assert gain[2] == 0.0                              # outside the hit radius
    assert gain[0] > 0 and gain[1] > 0


def test_sphere_lfu_never_evicts_the_entry_being_inserted():
    p = SphereLFUPolicy(cos_threshold=0.9, capacity=2)
    p.on_insert(entry(0), unit([1, 0]))
    p.on_query(unit([1, 0]))                           # entry 0 now above 1
    p.on_insert(entry(1), unit([0, 1]))
    assert p.choose_victim(None, None) == 0 or p.count[0] > p.count[1]
    p2 = SphereLFUPolicy(cos_threshold=0.9, capacity=2)
    p2.on_insert(entry(0), unit([1, 0]))
    p2.on_insert(entry(1), unit([0, 1]))
    assert p2.choose_victim(None, None) == 0           # tie: oldest, never the newest


# ------------------------------------------------------------------ admission
def test_admission_rejects_newcomer_on_tie_and_without_evidence():
    p = SemanticAdmissionPolicy(beta=0.0, admission=True, period=1000)
    for i in range(3):
        p.on_query(unit([1, i, 0]))
        p.on_insert(entry(i), unit([1, i, 0]))
    assert p.choose_victim(None, None) == 2            # all score 1: newcomer rejected


def test_admission_off_never_picks_the_newcomer():
    p = SemanticAdmissionPolicy(beta=0.0, admission=False, period=1000)
    for i, v in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):    # no neighbours
        p.on_query(unit(v))
        p.on_insert(entry(i), unit(v))
    assert p.choose_victim(None, None) == 0            # oldest resident instead


def test_admission_tie_break_prefers_an_entry_whose_topic_stays_covered():
    p = SemanticAdmissionPolicy(beta=0.0, admission=False, period=1000)
    vecs = ([1, 0, 0], [0, 1, 0], [0, 1, 0.05], [0, 0, 1])       # 1 and 2 are near-duplicates
    for i, v in enumerate(vecs):
        p.on_query(unit(v))
        p.on_insert(entry(i), unit(v))
    assert p.choose_victim(None, None) in (1, 2)       # covered entry before the oldest (0)


def test_admission_keeps_a_hit_entry_over_a_cold_one():
    p = SemanticAdmissionPolicy(beta=0.0, admission=True, period=1000)
    p.on_query(unit([1, 0, 0])); p.on_insert(entry(0), unit([1, 0, 0]))
    p.on_query(unit([0, 1, 0])); p.on_insert(entry(1), unit([0, 1, 0]))
    p.on_hit(entry(0), None)
    p.on_query(unit([0, 0, 1])); p.on_insert(entry(2), unit([0, 0, 1]))
    assert p.choose_victim(None, None) == 2            # 1 and 2 tie at 1: newcomer goes


# ------------------------------------------------------------------ RiskPriced
def test_risk_priced_values_hits_and_purges_negative_entries():
    p = RiskPricedPolicy(lam=5.0, prior=1.0, purge_negative=True, min_hits=2, period=1000)
    p.on_insert(entry(0), None)
    p.on_hit_prob(0, 0.6)                              # 0.6 - 5 * 0.4 = -1.4 -> value -0.4
    assert p.value[0] == pytest.approx(1.0 - 1.4)
    assert p.purge() == []                             # needs min_hits
    p.on_hit_prob(0, 0.6)
    assert p.purge() == [0]


def test_risk_priced_confident_hits_behave_like_frequency():
    p = RiskPricedPolicy(lam=10.0, prior=1.0, admission=False, purge_negative=False)
    p.on_insert(entry(0), None); p.on_insert(entry(1), None); p.on_insert(entry(2), None)
    p.on_hit_prob(0, 1.0); p.on_hit_prob(0, 1.0); p.on_hit_prob(1, 1.0)
    assert p.choose_victim(None, None) == 1            # 2 is newest and protected without admission


# ------------------------------------------------------------------ vCache
def _vcache_tau(obs, s, delta=0.02):
    from semcache.baselines import VCacheBaseline
    vc = VCacheBaseline.__new__(VCacheBaseline)
    vc.delta, vc._fit = delta, {}
    vc.eps = np.linspace(1e-6, 1 - 1e-6, 50)
    vc.z = norm.ppf(1 - vc.eps)
    vc.obs = {1: obs}
    return vc._tau(s, 1)


def test_vcache_tau_falls_as_similarity_rises():
    obs = [(0.0, 0), (1.0, 1), (0.8, 1), (0.82, 1), (0.6, 0), (0.7, 0), (0.9, 1), (0.75, 1), (0.65, 0)]
    taus = [_vcache_tau(obs, s) for s in (0.6, 0.75, 0.9, 0.99)]
    assert all(t1 >= t2 for t1, t2 in zip(taus, taus[1:]))
    assert 0.0 <= taus[-1] <= taus[0] <= 1.0


def test_vcache_tau_tightens_with_smaller_error_bound():
    obs = [(0.0, 0), (1.0, 1), (0.8, 1), (0.82, 1), (0.6, 0), (0.7, 0), (0.9, 1), (0.75, 1), (0.65, 0)]
    assert _vcache_tau(obs, 0.9, delta=0.01) >= _vcache_tau(obs, 0.9, delta=0.05)


def test_vcache_matches_official_formula_at_one_point():
    # tau = min_eps 1 - delta / (1 - (1-eps) * sigmoid(gamma (s - t'(eps)))), t' = t_hat + z sqrt(var)
    obs = [(0.0, 0), (1.0, 1), (0.8, 1), (0.82, 1), (0.6, 0), (0.7, 0), (0.9, 1), (0.75, 1), (0.65, 0)]
    from semcache.baselines import VCacheBaseline
    vc = VCacheBaseline.__new__(VCacheBaseline)
    vc.delta, vc._fit, vc.obs = 0.02, {}, {1: obs}
    vc.eps = np.linspace(1e-6, 1 - 1e-6, 50)
    vc.z = norm.ppf(1 - vc.eps)
    t_hat, gamma, var_t = vc._params(1)
    eps = np.linspace(1e-6, 1 - 1e-6, 50)
    tp = np.clip(t_hat + norm.ppf(1 - eps) * math.sqrt(var_t), 0, 1)
    lik = 1 / (1 + np.exp(-gamma * (0.85 - tp)))
    expected = float(np.clip(np.min(1 - 0.02 / (1 - (1 - eps) * lik)), 0, 1))
    assert vc._tau(0.85, 1) == pytest.approx(expected)


# ------------------------------------------------------------------ LeCaR
def test_lecar_penalises_the_expert_whose_victim_returns():
    from semcache.cache import LeCaRPolicy
    p = LeCaRPolicy(cos_threshold=0.9, capacity=2, seed=0)
    p.w[:] = [0.999999, 1e-6]                         # force the LRU expert
    for i, v in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        p.on_query(unit(v))
        p.on_insert(entry(i), unit(v))
    victim = p.choose_victim(None, None)
    assert victim == 0                                 # least recently used, not the newcomer
    p.on_evict(victim)
    w_lru_before = p.w[0]
    p.on_query(unit([1, 0.01, 0]))                     # a request near the evicted entry
    assert p.w[0] < w_lru_before                       # LRU blamed
    assert p.hist[0] == []                             # the remembered victim is consumed


def test_lecar_lfu_expert_evicts_least_frequent():
    from semcache.cache import LeCaRPolicy
    p = LeCaRPolicy(cos_threshold=0.9, capacity=3, seed=0)
    p.w[:] = [1e-6, 0.999999]                         # force the LFU expert
    for i, v in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0])):
        p.on_query(unit(v))
        p.on_insert(entry(i), unit(v))
    p.on_hit(entry(0), None); p.on_hit(entry(2), None)
    assert p.choose_victim(None, None) == 1


# ------------------------------------------------------------------ SD-RR (redesigned Semantic Decay)
def test_sdrr_without_extra_terms_is_lfu():
    from semcache.cache import RedundancyRiskDecayPolicy
    p = RedundancyRiskDecayPolicy(alpha=0.0, beta=0.0, half_life=None)
    for i, v in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0])):
        p.on_query(unit(v)); p.on_insert(entry(i), unit(v))
    p.on_hit(entry(0), None); p.on_hit(entry(2), None)
    assert p.choose_victim(None, None) == 1            # least frequent, oldest; newcomer 3 protected


def test_sdrr_evicts_the_redundant_duplicate_before_a_unique_entry():
    from semcache.cache import RedundancyRiskDecayPolicy
    p = RedundancyRiskDecayPolicy(alpha=1.0, beta=0.0, radius=0.9)
    vecs = ([1, 0, 0], [0, 1, 0], [0, 1, 0.05], [0, 0, 1])   # 1 and 2 are near-duplicates
    for i, v in enumerate(vecs):
        p.on_query(unit(v)); p.on_insert(entry(i), unit(v))
    p.on_hit(entry(1), None)                           # 1 earns more, so 2 is the redundant copy
    p.on_hit(entry(0), None)
    assert p.choose_victim(None, None) == 2


def test_sdrr_risk_term_evicts_the_uncertain_entry():
    from semcache.cache import RedundancyRiskDecayPolicy
    p = RedundancyRiskDecayPolicy(alpha=0.0, beta=4.0)
    for i, v in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        p.on_query(unit(v)); p.on_insert(entry(i), unit(v))
    for e, prob in ((0, 0.55), (1, 0.99)):
        p.on_hit(entry(e), None); p.on_hit_prob(e, prob)
    assert p.choose_victim(None, None) == 0            # equal counts; 0 was served at borderline confidence


def test_sdrr_decay_lets_recent_hits_outweigh_old_ones():
    from semcache.cache import RedundancyRiskDecayPolicy
    p = RedundancyRiskDecayPolicy(alpha=0.0, beta=0.0, half_life=10)
    for i, v in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        p.on_query(unit(v)); p.on_insert(entry(i), unit(v))
    for _ in range(3):
        p.on_hit(entry(0), None)                       # old popularity
    for _ in range(100):
        p.on_query(unit([0, 0, 1]))
    p.on_hit(entry(1), None)                           # one recent hit
    assert p.choose_victim(None, None) == 0


# ------------------------------------------------------------------ SemDemand (Semantic Decay with semantic admission)
def _feed(p, vecs, start=0):
    for i, v in enumerate(vecs, start):
        p.on_query(unit(v)); p.on_insert(entry(i), unit(v))


def test_semdemand_rejects_a_one_off_newcomer_when_residents_are_in_demand():
    from semcache.cache import SemanticDemandPolicy
    p = SemanticDemandPolicy(radius=0.9, half_life=None, admission=True)
    _feed(p, ([1, 0, 0], [0, 1, 0]))
    for v in ([1, 0, 0], [0, 1, 0]):                   # both residents requested again
        p.on_query(unit(v))
    _feed(p, ([0, 0, 1],), start=2)                    # unseen question
    assert p.choose_victim(None, None) == 2            # newcomer dropped


def test_semdemand_admits_a_newcomer_whose_paraphrase_was_seen():
    from semcache.cache import SemanticDemandPolicy
    sem = SemanticDemandPolicy(radius=0.9, half_life=None, admission=True)
    exact = SemanticDemandPolicy(radius=0.9999, half_life=None, admission=True)
    for p in (sem, exact):
        _feed(p, ([1, 0, 0], [0, 1, 0]))
        p.on_query(unit([0, 0, 1]))                    # first request of a new topic, not cached
        p.on_query(unit([0, 0.05, 1]))                 # paraphrase arrives and misses
        p.on_insert(entry(2), unit([0, 0.05, 1]))
    assert sem.choose_victim(None, None) in (0, 1)     # semantic demand 2 beats residents at 1
    assert exact.choose_victim(None, None) == 2        # exact-key counting sees no repeat


def test_semdemand_decay_lets_a_fresh_query_replace_a_stale_resident():
    from semcache.cache import SemanticDemandPolicy
    p = SemanticDemandPolicy(radius=0.9, half_life=10, admission=True)
    _feed(p, ([1, 0, 0], [0, 1, 0]))
    for _ in range(50):
        p.on_query(unit([1, 0, 0]))                    # resident 0 stays popular, 1 goes stale
    _feed(p, ([0, 0, 1],), start=2)
    assert p.choose_victim(None, None) == 1


def test_semdemand_without_admission_never_drops_the_newcomer():
    from semcache.cache import SemanticDemandPolicy
    p = SemanticDemandPolicy(radius=0.9, half_life=None, admission=False)
    _feed(p, ([1, 0, 0], [0, 1, 0]))
    for v in ([1, 0, 0], [0, 1, 0]):
        p.on_query(unit(v))
    _feed(p, ([0, 0, 1],), start=2)
    assert p.choose_victim(None, None) in (0, 1)


def test_semdemand_second_sighting_keeps_first_sightings_out():
    from semcache.cache import SemanticDemandPolicy
    p = SemanticDemandPolicy(radius=0.9, half_life=10, admission="second")
    _feed(p, ([1, 0, 0], [0, 1, 0]))
    for _ in range(50):
        p.on_query(unit([1, 0, 0]))                    # resident 1 goes stale
    _feed(p, ([0, 0, 1],), start=2)                    # first sighting: no prior demand
    assert p.choose_victim(None, None) == 2
    q = SemanticDemandPolicy(radius=0.9, half_life=10, admission="second")
    _feed(q, ([1, 0, 0], [0, 1, 0]))
    for _ in range(50):
        q.on_query(unit([1, 0, 0]))
    q.on_query(unit([0, 0, 1]))                        # seen once, not cached
    _feed(q, ([0, 0.05, 1],), start=2)                 # its paraphrase misses
    assert q.choose_victim(None, None) == 1            # admitted, replacing the stale resident


def test_tinylfu_halving_rejects_ties_and_admits_after_a_reset():
    from semcache.cache import SemanticDemandPolicy
    p = SemanticDemandPolicy(radius=0.9999, aging="halve", period=10, admission="compete")
    _feed(p, ([1, 0, 0], [0, 1, 0]))
    _feed(p, ([0, 0, 1],), start=2)                    # count 1 vs victim 1: tie, rejected
    assert p.choose_victim(None, None) == 2
    p.on_evict(2)
    for _ in range(10):
        p.on_query(unit([1, 0.3, 0.3]))                # unrelated traffic crosses a reset
    _feed(p, ([0, 0, 1],), start=3)
    assert p.choose_victim(None, None) in (0, 1)       # residents halved to 0, newcomer at 1 wins


# ------------------------------------------------------------------ windowed semantic admission
def _wfeed(p, vecs, start=0):
    for i, v in enumerate(vecs, start):
        p.on_query(unit(v)); p.on_insert(entry(i), unit(v))


def test_window_drops_a_one_off_leaving_the_window():
    from semcache.cache import WindowSemanticDemandPolicy
    p = WindowSemanticDemandPolicy(capacity=3, window_frac=0.34, radius=0.9, half_life=None)
    _wfeed(p, ([1, 0, 0], [0, 1, 0], [0, 0, 1]))           # 0, 1 move to main; 2 stays in the window
    p.on_query(unit([1, 0, 0])); p.on_query(unit([0, 1, 0]))   # main entries gain evidence
    _wfeed(p, ([1, 1, 0],), start=3)                       # window overflows: 2 competes with 0 evidence
    assert p.choose_victim(None, None) == 2


def test_window_promotes_a_paraphrased_query_over_a_stale_main_entry():
    from semcache.cache import WindowSemanticDemandPolicy
    sem = WindowSemanticDemandPolicy(capacity=3, window_frac=0.34, radius=0.9, half_life=None)
    exact = WindowSemanticDemandPolicy(capacity=3, window_frac=0.34, radius=0.9999, half_life=None)
    for p in (sem, exact):
        _wfeed(p, ([1, 0, 0], [0, 1, 0], [0, 0, 1]))
        p.on_query(unit([1, 0, 0]))                        # 0 keeps evidence, 1 has none
        p.on_query(unit([0, 0.05, 1]))                     # paraphrase of the window entry 2
        _wfeed(p, ([1, 1, 0],), start=3)
    assert sem.choose_victim(None, None) == 1              # 2 promoted, stale main entry 1 leaves
    assert exact.choose_victim(None, None) == 2            # exact counting misses the paraphrase


def test_window_serves_new_entries_before_any_competition():
    from semcache.cache import WindowSemanticDemandPolicy
    p = WindowSemanticDemandPolicy(capacity=4, window_frac=0.5, radius=0.9, half_life=None)
    _wfeed(p, ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]))
    assert len(p.window) == 2 and set(p.D) == {0, 1, 2, 3}
