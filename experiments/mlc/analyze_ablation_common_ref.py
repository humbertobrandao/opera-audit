"""Ablation root gap with a common reference (Section 6.6): the best
incumbent of the cell across the variants. Reproduces: indicators with a
smaller root gap in 21 of the 22 unproven cells, median reduction of 37%.

Usage: analyze_ablation_common_ref.py results/ablation_hard_mediamill_R2.json
"""
import json, sys, statistics as st

d = json.load(open(sys.argv[1]))
hard = {k: c for k, c in d['cases'].items()
        if not c['F3_agg_estrutural']['optimal'] and not c['F4_agg_indicator']['optimal']}
rows, menor = [], 0
for c in hard.values():
    ref = max(v['U_int'] for v in c.values() if isinstance(v, dict))
    g3 = c['F3_agg_estrutural']['root_bound'] - ref
    g4 = c['F4_agg_indicator']['root_bound'] - ref
    if g4 < g3 - 1e-9: menor += 1
    if g3 > 0: rows.append(100 * (g3 - g4) / g3)
print(f"cells unproven by both aggregated variants: {len(hard)}")
print(f"indicators with a smaller gap: {menor}/{len(hard)}; median reduction {st.median(rows):.1f}%")
print(f"proofs: F3 {sum(1 for c in d['cases'].values() if c['F3_agg_estrutural']['optimal'])}/40 | "
      f"F4 {sum(1 for c in d['cases'].values() if c['F4_agg_indicator']['optimal'])}/40")
