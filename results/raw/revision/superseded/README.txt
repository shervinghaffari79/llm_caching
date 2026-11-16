Run on 2026-10-04 on Clore server #110175 (RTX 4090, i7-12th gen, 20 threads), before:
- meta-classifier inference was fixed to one thread (LightGBM n_jobs=-1 contended with the encoders), and
- top-k candidates were scored in one batched LightGBM call instead of one call each.
Decisions are unchanged by either fix; only measured overheads change.
Replaced by the runs on server #100123 (RTX 4090, Ryzen 9 7950X3D).

Also superseded (2026-10-05):
- exp11_seeds_geforcertx4090.json: virtual-time replay, before the LightGBM thread fix; replaced by exp11_seeds_geforcertx4090_real.json (real waits).
- exp2_end_to_end_geforcertx4090_seed42.json: virtual-time replay; replaced by exp2_end_to_end_geforcertx4090_seed42_real.json.
- exp2_end_to_end_geforcertx4090_seed42_realsleep.json: 300-request real-wait check from the first host.
- exp12_threshold_risk_geforcertx4090_seed42.json: before vCache was added; replaced by exp12_threshold_risk_m2_seed42.json.
- exp18_eviction_geforcertx4090.json: before SphereLFU was added; replaced by exp18_eviction_m2.json.
Decisions in these files are identical to their replacements; only timings or the set of systems differ.
