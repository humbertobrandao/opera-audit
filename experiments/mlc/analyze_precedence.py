"""Summarizes the precedence-experiment JSON into the published numbers
(Section 6.6, paragraph 'Measured effect of the precedence inequalities'):
proofs 18->22/40; on the 18 unproven cells, smaller root gap in 18/18 with
a median reduction of 74% (common reference = best incumbent across the
variants); root <= U_mon in 40/40; median of 13,384 inequalities per cell;
0.014 s (without) vs 0.020 s (with) on the cells proved by both.

Usage: analyze_precedence.py records/r1_mediamill_R2.json
"""
import json, sys, statistics as st

d = json.load(open(sys.argv[1]))
cells = d['cells']; n = len(cells)
f4 = lambda c: c['F4_agg_indicator']; f5 = lambda c: c['F5_agg_ind_prec']
opt4 = sum(1 for c in cells.values() if f4(c)['optimal'])
opt5 = sum(1 for c in cells.values() if f5(c)['optimal'])
print(f"proofs within the budget: without precedence {opt4}/{n} | with {opt5}/{n}")
both = [c for c in cells.values() if f4(c)['optimal'] and f5(c)['optimal']]
assert all(f4(c)['U_int'] == f5(c)['U_int'] for c in both), "optima diverge"
print(f"identical optima on the {len(both)} cells proved by both; median t_solver "
      f"{st.median(f4(c)['t_solver'] for c in both):.3f}s (without) vs "
      f"{st.median(f5(c)['t_solver'] for c in both):.3f}s (with)")
hard = [c for c in cells.values() if not f4(c)['optimal'] and not f5(c)['optimal']]
red, menor = [], 0
for c in hard:
    inc = max(f4(c)['U_int'], f5(c)['U_int'])
    g4, g5 = f4(c)['root_bound'] - inc, f5(c)['root_bound'] - inc
    if g5 < g4 - 1e-9: menor += 1
    if g4 > 0: red.append(100 * (g4 - g5) / g4)
print(f"unproven cells: {len(hard)}; smaller root gap in {menor}/{len(hard)}; "
      f"median reduction {st.median(red):.1f}% (min {min(red):.1f}, max {max(red):.1f})")
san = sum(1 for c in cells.values() if f5(c)['root_bound'] <= c['U_mon_int'] + 1.0)
print(f"sanity root(F5) <= U_mon: {san}/{n} (recorded violations: {d['violacoes_sanidade']})")
print(f"inequalities per cell: median {st.median(f5(c)['n_prec'] for c in cells.values()):.0f}")
